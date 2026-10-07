# Detector and tracker

The code in `src/roi_ant_tracker/` trains, evaluates, and runs the ant detector.
Use `../pipeline.py` as the entry point so the local package is loaded without
an editable install. Run all commands from the `Taka` folder with the tested
`python_env` (see [`../README.md`](../README.md)).

```powershell
$python = "$HOME\miniconda3\envs\python_env\python.exe"
```

## Train

Start with a COCO export from [`../Annotation/README.md`](../Annotation/README.md).
For an unbiased evaluation, split it into temporal train and validation blocks
before training. For example:

```powershell
& $python .\FINAL\pipeline.py train-detector `
  --images .\FINAL\runs\Type1_Vid1_labels\selected_frames `
  --annotations .\FINAL\runs\Type1_Vid1_labels\annotations_coco.json `
  --output .\FINAL\checkpoints\ant_detector_new.pt `
  --epochs 30 --device auto
```

Training with `--initialization detector` may fetch pretrained torchvision
weights on the first run. The copied `checkpoints/` already contains trained
MobileNet detector weights, including
`ant_detector_mobilenet_round2_reset_best.pt`.

## Track and evaluate

```powershell
& $python .\FINAL\pipeline.py track-video `
  --video .\Videos\Type1_Vid1.mp4 `
  --roi-json .\FINAL\runs\Type1_Vid1_roi.json `
  --checkpoint .\FINAL\checkpoints\ant_detector_mobilenet_round2_reset_best.pt `
  --output .\FINAL\outputs\Type1_Vid1_new `
  --device auto --duration-seconds 10
```

Tracking writes detections, trajectories, review events, and a run manifest.
`evaluate-detector` compares a checkpoint to held-out COCO labels and writes
summary and per-image metrics. Use `--device cpu` if CUDA is unavailable.

## Tests

```powershell
& $python -m pytest .\FINAL\ML\tests -q
```

The test configuration imports this copy's `ML/src`, so it does not depend on
another copy of `roi_ant_tracker` being installed.
