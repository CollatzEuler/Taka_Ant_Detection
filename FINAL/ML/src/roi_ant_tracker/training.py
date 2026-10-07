from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import CocoBoxDataset, collate_detection_batch
from .model import build_detector, choose_device


def train_detector(
    images_dir: str,
    annotations_path: str,
    output_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    architecture: str,
    min_size: int,
    max_size: int,
    initialization: str,
    device_name: str,
    workers: int,
    resume_checkpoint: str | None = None,
    box_detections_per_img: int = 300,
    lr_patience: int = 3,
    lr_factor: float = 0.5,
    min_learning_rate: float = 1e-7,
    plateau_delta: float = 0.001,
    log_every: int = 10,
    max_grad_norm: float = 1.0,
    max_empty_fraction: float = 1.0,
    reset_training_history: bool = False,
) -> None:
    if lr_patience < 1:
        raise ValueError("lr_patience must be at least 1.")
    if not 0.0 < lr_factor < 1.0:
        raise ValueError("lr_factor must be between 0 and 1.")
    if min_learning_rate <= 0:
        raise ValueError("min_learning_rate must be positive.")
    if plateau_delta < 0:
        raise ValueError("plateau_delta must be non-negative.")
    if max_grad_norm < 0:
        raise ValueError("max_grad_norm must be non-negative. Use 0 to disable clipping.")
    if not 0.0 <= max_empty_fraction <= 1.0:
        raise ValueError("max_empty_fraction must be in [0, 1].")
    device = choose_device(device_name)
    dataset = CocoBoxDataset(
        images_dir,
        annotations_path,
        training=True,
        max_empty_fraction=max_empty_fraction,
    )
    if len(dataset) == 0:
        raise ValueError("No images were found in the COCO annotation file.")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_detection_batch,
    )

    starting_epoch = 0
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model_config = checkpoint["model_config"]
        model = build_detector(**model_config)
        model.load_state_dict(checkpoint["model_state"])
        if reset_training_history:
            starting_epoch = 0
            initialization_description = f"resume={resume_checkpoint} reset_history"
        else:
            starting_epoch = int(checkpoint.get("epoch", 0))
            initialization_description = f"resume={resume_checkpoint}"
    else:
        model_config = {
            "architecture": architecture,
            "num_classes": 2,
            "min_size": min_size,
            "max_size": max_size,
            "pretrained_backbone": False,
            "pretrained_detector": False,
            "box_detections_per_img": box_detections_per_img,
        }
        if initialization not in {"detector", "backbone", "random"}:
            raise ValueError(f"Unsupported initialization: {initialization}")
        model = build_detector(
            **{
                **model_config,
                "pretrained_detector": initialization == "detector",
                "pretrained_backbone": initialization == "backbone",
            }
        )
        initialization_description = f"initialization={initialization}"

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        threshold=plateau_delta,
        threshold_mode="rel",
        min_lr=min_learning_rate,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_output = output.with_name(output.stem + "_best" + output.suffix)
    log_path = output.with_name(output.stem + "_training_log.csv")
    history = checkpoint.get("training_history", []) if resume_checkpoint and not reset_training_history else []
    best_loss = min((float(row["mean_loss"]) for row in history), default=float("inf"))
    plateau_epochs = 0
    write_training_log(log_path, history)
    print(
        f"[train] images={len(dataset)} device={device} architecture={model_config['architecture']} "
        f"positive_images={dataset.positive_image_count} empty_images={dataset.empty_image_count} "
        f"{initialization_description} lr={current_lr(optimizer):.3g} "
        f"plateau_patience={lr_patience} lr_factor={lr_factor} max_grad_norm={max_grad_norm}"
    )
    final_epoch = starting_epoch + epochs
    for epoch in range(starting_epoch + 1, final_epoch + 1):
        started = time.time()
        running_loss = 0.0
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                loss_values = {
                    name: float(value.detach().cpu()) if torch.isfinite(value.detach()).all() else "nonfinite"
                    for name, value in losses.items()
                }
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch} batch={batch_index}. "
                    f"Stopping before optimizer step. losses={loss_values}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = None
            if max_grad_norm > 0:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                grad_norm = float(grad_norm_tensor.detach().cpu())
                if not torch.isfinite(grad_norm_tensor):
                    raise RuntimeError(
                        f"Non-finite gradient norm at epoch={epoch} batch={batch_index}. "
                        "Stopping before optimizer step."
                    )
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            if log_every > 0 and (batch_index % log_every == 0 or batch_index == len(loader)):
                grad_text = "" if grad_norm is None else f" grad_norm={grad_norm:.4f}"
                print(
                    f"[train] epoch={epoch}/{final_epoch} batch={batch_index}/{len(loader)} "
                    f"loss={running_loss / batch_index:.4f} lr={current_lr(optimizer):.3g}{grad_text}"
                )
        mean_loss = running_loss / len(loader)
        previous_lr = current_lr(optimizer)
        improved = mean_loss < best_loss * (1.0 - plateau_delta)
        if improved:
            best_loss = mean_loss
            plateau_epochs = 0
        else:
            plateau_epochs += 1
        scheduler.step(mean_loss)
        new_lr = current_lr(optimizer)
        lr_changed = new_lr < previous_lr
        history.append(
            {
                "epoch": epoch,
                "mean_loss": mean_loss,
                "best_loss": best_loss,
                "learning_rate": new_lr,
                "plateau_epochs": plateau_epochs,
                "lr_reduced": int(lr_changed),
                "seconds": time.time() - started,
            }
        )
        write_training_log(log_path, history)
        checkpoint_payload = {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "epoch": epoch,
            "training_history": history,
            "optimizer_config": {
                "learning_rate": learning_rate,
                "lr_patience": lr_patience,
                "lr_factor": lr_factor,
                "min_learning_rate": min_learning_rate,
                "plateau_delta": plateau_delta,
                "max_grad_norm": max_grad_norm,
                "reset_training_history": reset_training_history,
            },
        }
        torch.save(checkpoint_payload, output)
        if improved:
            torch.save(checkpoint_payload, best_output)
        status = "improved" if improved else f"plateau={plateau_epochs}/{lr_patience}"
        lr_note = " lr_reduced" if lr_changed else ""
        best_note = f" best={best_output}" if improved else ""
        print(
            f"[train] saved={output} epoch={epoch} mean_loss={mean_loss:.4f} "
            f"best_loss={best_loss:.4f} lr={new_lr:.3g} {status}{lr_note} "
            f"log={log_path}{best_note} seconds={time.time() - started:.1f}"
        )


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def write_training_log(path: Path, history: list[dict[str, object]]) -> None:
    fields = ["epoch", "mean_loss", "best_loss", "learning_rate", "plateau_epochs", "lr_reduced", "seconds"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field, "") for field in fields})
