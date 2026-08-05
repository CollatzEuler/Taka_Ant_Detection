# ROI Ant ML Pipeline

This folder replaces the failed background-subtraction path with an ROI-first
detector pipeline. The source video stays untouched. For each video, draw one
or more rectangles on a representative middle frame; training samples and
inference are then limited to those rectangles.

There is no preprocessing trim stage here: no background subtraction, no noisy
blob filtering, and no cropped/trimmed output video. ROI crops are written only
as annotation/training images, and inference offsets detections back into the
original full-frame coordinate system.

For a step-by-step annotator onboarding guide, including sampling choices, GUI
controls, saving and resuming, label QA, COCO export, and temporal splitting,
see [`Annotation/README.md`](Annotation/README.md).

## Environment

From the Taka project root:

```powershell
$python = "$HOME\miniconda3\envs\python_env\python.exe"
& $python .\MLPipeline\pipeline.py --help
```

To create the tested environment from the project-level Conda file instead:

```powershell
conda env create -f .\environment.yaml
conda activate taka-ant-ml
python .\MLPipeline\pipeline.py --help
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

To update an existing environment after `environment.yaml` changes:

```powershell
conda env update -f .\environment.yaml --prune
```

On this computer, install the CUDA PyTorch build first, then the remaining
requirements. The GTX 1650 Ti works with the CUDA 12.8 wheel:

```powershell
& $python -m pip install torch==2.11.0 torchvision==0.26.0 `
  --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install -r .\MLPipeline\requirements.txt
& $python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Workflow

### 1. Draw Search ROIs

The default frame is the middle of the video.

```powershell
& $python .\MLPipeline\pipeline.py select-roi -- `
  --video ".\Videos\Type1_Vid1.mp4" `
  --output ".\MLPipeline\runs\Type1_Vid1_roi.json"
```

Controls in the OpenCV window:

- drag: add ROI rectangle
- `u`: undo last rectangle
- `c`: clear all rectangles
- `s`: save
- `q` or Esc: quit without saving

For non-interactive use:

```powershell
& $python .\MLPipeline\pipeline.py select-roi -- `
  --video ".\Videos\Type1_Vid1.mp4" `
  --output ".\MLPipeline\runs\Type1_Vid1_roi.json" `
  --rect 120,180,420,300 `
  --rect 700,210,360,280
```

The command writes the ROI JSON, a representative frame PNG, and a preview PNG.

### 1A. Black Out Forbidden Areas And Sample Random Training ROIs

For a long, static-camera video, you do not need to choose one fixed location
for annotation crops. Paint only the places that random training ROIs must
avoid. Everything not black remains eligible:

```powershell
& $python .\MLPipeline\pipeline.py select-exclusions -- `
  --video ".\Videos\long_video.mp4" `
  --output ".\MLPipeline\runs\long_video_exclusions.json" `
  --frame middle
```

Controls:

- left-drag: paint an excluded area black
- right-drag: restore an allowed area
- `[` / `]`: decrease/increase the brush size
- `u`: undo the last stroke
- `c`: clear the mask
- `s`: save
- `q` or Esc: quit without saving

The mask is appropriate when the camera and arena remain fixed. Repaint it if
the camera moves, zooms, or changes resolution.

Next, reserve some selections from the beginning and choose the rest from
stratified random times later in the video. Candidate positions are random but
must avoid the black mask. Crop dimensions are chosen from the reference ROIs
with only the requested size jitter:

```powershell
& $python .\MLPipeline\pipeline.py sample-random-roi -- `
  --video ".\Videos\long_video.mp4" `
  --reference-roi-json ".\MLPipeline\runs\Type1_Vid1_roi.json" `
  --exclusion-json ".\MLPipeline\runs\long_video_exclusions.json" `
  --output ".\MLPipeline\runs\long_video_random_labels" `
  --count 400 `
  --beginning-count 150 `
  --beginning-seconds 600 `
  --candidate-multiplier 6 `
  --patches-per-frame 8 `
  --size-jitter 0.08 `
  --max-excluded-fraction 0.01 `
  --motion-gap-frames 6 `
  --motion-guided-fraction 0.50
```

This example selects 150 crops from the first ten minutes and 250 from random,
time-stratified locations after that. `--size-jitter 0.08` restricts each crop
to 92–108% of one of the reference ROI sizes. The default references here are
300×270 and 282×265 pixels.

The fast selection stage centers half its proposals on localized motion and
leaves half spatially random, then combines motion, focus, contrast, exposure,
OpenCV PCA, and OpenCV k-means. It keeps diverse cluster representatives while
slightly preferring high-quality patches. It does not assume every selected
crop contains an ant: empty and ant-like negative crops are necessary training
data. Review `selection_preview.jpg` before annotating. If reflections dominate,
black them out or reduce `--quality-weight`; if useful rare events are missing,
increase `--candidate-multiplier` from 6 to 8 or 10.

The output is directly compatible with the existing annotation GUI:

```powershell
& $python .\MLPipeline\pipeline.py annotate -- `
  --images-dir ".\MLPipeline\runs\long_video_random_labels\selected_frames" `
  --manifest ".\MLPipeline\runs\long_video_random_labels\frame_manifest.csv" `
  --project ".\MLPipeline\runs\long_video_random_labels\annotations.json"
```

Random ROIs are for building spatially diverse training data. During final
tracking, search every fixed region where an ant must be detected; randomly
checking only part of the arena would miss ants by construction.

### 2. Sample ROI Frames For Annotation

```powershell
& $python .\MLPipeline\pipeline.py sample-roi -- `
  --videos ".\Videos\Type1_Vid1.mp4" `
  --roi-jsons ".\MLPipeline\runs\Type1_Vid1_roi.json" `
  --output ".\MLPipeline\runs\Type1_Vid1_labels" `
  --count 300 `
  --frame-step 90 `
  --duration-seconds 300
```

For training, you can pass multiple ROI JSONs for the same video to sample
different regions:

```powershell
--roi-jsons ".\MLPipeline\runs\roi_a.json" ".\MLPipeline\runs\roi_b.json"
```

Use `--start-seconds`, `--end-seconds`, or `--duration-seconds` to limit how
much of a long source video is scanned for annotation frames. Progress lines
include percent complete, rate, and ETA.

After a first rough detector is trained, use it to create a better annotation
round with suggested boxes:

```powershell
& $python .\MLPipeline\pipeline.py sample-suggested -- `
  --video ".\Videos\Type1_Vid1.mp4" `
  --roi-json ".\MLPipeline\runs\Type1_Vid1_roi.json" `
  --checkpoint ".\MLPipeline\checkpoints\ant_detector_mobilenet_best.pt" `
  --output ".\MLPipeline\runs\Type1_Vid1_suggested_round2" `
  --count 200 `
  --frame-step 30 `
  --duration-seconds 300 `
  --score-threshold 0.12
```

Then open the suggested project, delete false positives, add missed ants, and
export COCO:

```powershell
& $python .\MLPipeline\pipeline.py annotate -- `
  --images-dir ".\MLPipeline\runs\Type1_Vid1_suggested_round2\selected_frames" `
  --manifest ".\MLPipeline\runs\Type1_Vid1_suggested_round2\frame_manifest.csv" `
  --project ".\MLPipeline\runs\Type1_Vid1_suggested_round2\annotations.json"
```

### 3. Annotate Ant Boxes

```powershell
& $python .\MLPipeline\pipeline.py annotate -- `
  --images-dir ".\MLPipeline\runs\Type1_Vid1_labels\selected_frames" `
  --manifest ".\MLPipeline\runs\Type1_Vid1_labels\frame_manifest.csv" `
  --project ".\MLPipeline\runs\Type1_Vid1_labels\annotations.json"
```

Use the GUI to draw boxes around ants inside each ROI image. Press `e` or click
`Export COCO` to write `annotations_coco.json`, or export explicitly:

```powershell
& $python .\MLPipeline\pipeline.py export-coco -- `
  --project ".\MLPipeline\runs\Type1_Vid1_labels\annotations.json" `
  --output ".\MLPipeline\runs\Type1_Vid1_labels\annotations_coco.json"
```

Keep correctly annotated empty frames. They teach the detector that marks,
scratches, and an empty second ROI are not ants. Use `--skip-empty` only for a
temporary positive-only export, not the final training or evaluation set.

Before training on a new video, reserve temporally separate validation blocks.
This prevents nearly identical adjacent frames from appearing in both sets:

```powershell
& $python .\MLPipeline\pipeline.py split-coco -- `
  --annotations ".\MLPipeline\runs\new_video_labels\annotations_coco.json" `
  --train-output ".\MLPipeline\runs\new_video_labels\train_coco.json" `
  --validation-output ".\MLPipeline\runs\new_video_labels\validation_coco.json" `
  --validation-fraction 0.20 `
  --block-frames 600
```

For a small update from the new video, continue from the current best model
with a low learning rate and then evaluate only on `validation_coco.json`:

```powershell
& $python .\MLPipeline\pipeline.py train-detector `
  --images ".\MLPipeline\runs\long_video_random_labels\selected_frames" `
  --annotations ".\MLPipeline\runs\long_video_random_labels\train_coco.json" `
  --output ".\MLPipeline\checkpoints\ant_detector_long_video.pt" `
  --resume-checkpoint ".\MLPipeline\checkpoints\ant_detector_mobilenet_round2_reset_best.pt" `
  --epochs 5 `
  --learning-rate 1e-5 `
  --batch-size 2 `
  --max-empty-fraction 0.50 `
  --device cuda
```

### 4. Train The Detector

MobileNet is now the default lightweight model:

```powershell
& $python .\MLPipeline\pipeline.py train-detector `
  --images ".\MLPipeline\runs\Type1_Vid1_labels\selected_frames" `
  --annotations ".\MLPipeline\runs\Type1_Vid1_labels\annotations_coco.json" `
  --output ".\MLPipeline\checkpoints\ant_detector_mobilenet.pt" `
  --epochs 30 `
  --lr-patience 3 `
  --lr-factor 0.5 `
  --log-every 10 `
  --max-grad-norm 1.0 `
  --max-empty-fraction 0.25
```

The heavier Fed-style ResNet model is still available:

```powershell
--architecture resnet50_fpn_v2 --min-size 1024 --max-size 1440
```

Training prints batch loss with the current learning rate, then an epoch
summary with mean loss, best loss, plateau count, and whether LR was reduced.
It also writes `ant_detector_mobilenet_training_log.csv` and a separate
`ant_detector_mobilenet_best.pt` beside the checkpoint. Non-finite loss now
aborts training before the optimizer step, so a NaN batch cannot overwrite the
last clean epoch.

### 5. Track A Video In The Saved ROIs

```powershell
& $python .\MLPipeline\pipeline.py track-video `
  --video ".\Videos\Type1_Vid1.mp4" `
  --roi-json ".\MLPipeline\runs\Type1_Vid1_roi.json" `
  --checkpoint ".\MLPipeline\checkpoints\ant_detector_mobilenet.pt" `
  --output ".\MLPipeline\outputs\Type1_Vid1" `
  --duration-seconds 300
```

The current two-ROI workload fits four ROI crops per CUDA batch on the 4 GB
GPU. For the current checkpoint, FP16 is about 2.3 times faster than CPU:

```powershell
& $python .\MLPipeline\pipeline.py track-video `
  --video ".\Videos\Type1_Vid1.mp4" `
  --roi-json ".\MLPipeline\runs\Type1_Vid1_roi.json" `
  --checkpoint ".\MLPipeline\checkpoints\ant_detector_mobilenet_round2_reset_best.pt" `
  --output ".\MLPipeline\outputs\Type1_Vid1_gpu" `
  --device cuda `
  --amp fp16 `
  --frame-batch-size 2 `
  --tile-batch-size 4 `
  --score-threshold 0.30 `
  --duplicate-center-distance 12 `
  --duplicate-overlap-threshold 0.65
```

For a long 60-fps video, `--frame-step 15` analyzes 4 frames per source second
and is approximately real-time on this GPU. Use a smaller step when motion is
fast, or `--frame-step 1` for every source frame. The tracker scales its motion
gate by the source-frame gap, timestamps remain source-video timestamps, and
the manifest records speed and the last processed source frame.
It also records `requested_window_completed`; a false value means the video
decoder reached end-of-stream before the frame count advertised in the file.

```powershell
& $python .\MLPipeline\pipeline.py track-video `
  --video ".\Videos\long_video.mp4" `
  --roi-json ".\MLPipeline\runs\long_video_roi.json" `
  --checkpoint ".\MLPipeline\checkpoints\ant_detector_mobilenet_round2_reset_best.pt" `
  --output ".\MLPipeline\outputs\long_video_gpu" `
  --device cuda --amp fp16 `
  --frame-step 15 `
  --frame-batch-size 2 --tile-batch-size 4 `
  --score-threshold 0.30 `
  --no-overlay
```

Omit `--no-overlay` for ROI-cropped QC videos. `--roi-padding 12` or `16` can
provide context when ants are clipped at a rectangle edge; detections are still
kept only when their centers fall inside the original ROI. Recheck accuracy
after changing padding. Do not lower `--inference-min-size` to 512 with the
current checkpoint: measured recall collapsed because it was trained at 768.
Use `--score-threshold 0.43` only for a precision-first export; `0.30` retains
more detections and is safer for track continuity.
If a larger ROI, padding, or a heavier model causes a CUDA out-of-memory error,
reduce `--tile-batch-size` from `4` to `2`.

The same long-video settings are packaged in a launcher. Because script
execution is disabled by the machine-wide Windows policy, invoke it with a
per-process bypass:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\MLPipeline\run_long_video_gpu.ps1 `
  -Video .\Videos\long_video.mp4 `
  -RoiJson .\MLPipeline\runs\long_video_roi.json `
  -FrameStep 15
```

For a later slice, combine `--start-seconds` with `--duration-seconds`, for
example `--start-seconds 600 --duration-seconds 120`. Frame-based controls
(`--start-frame`, `--end-frame`) are still available.

Outputs:

- `detections.csv`: model detections in original full-frame coordinates
- `raw_trajectories.csv`: all provisional tracks
- `trajectories.csv`: tracks with at least `--minimum-track-hits`
- `review_events.csv`: frames likely to need manual checking
- `tracking_overlay.mp4`: full-frame QC video with ROI rectangles and tracks
- `tracking_manifest.json`: run settings and ROI metadata

### 6. Measure Detector Accuracy

Evaluate against COCO boxes that were held out before training:

```powershell
& $python .\MLPipeline\pipeline.py evaluate-detector `
  --images ".\MLPipeline\runs\new_video_labels\selected_frames" `
  --annotations ".\MLPipeline\runs\new_video_labels\validation_coco.json" `
  --checkpoint ".\MLPipeline\checkpoints\ant_detector_mobilenet_new_best.pt" `
  --output ".\MLPipeline\outputs\new_video_validation" `
  --device cuda --amp fp16 --batch-size 2 `
  --score-threshold 0.30
```

`summary.json` contains precision, recall, F1, AP at the requested IoU,
best-F1 score threshold, speed, and separate results for each ROI.

The existing combined labels were already used for training. Evaluation on
them is useful as a fit check, but it is not an unbiased generalization result.

## Do I Need To Annotate And Train First?

Yes, unless you already have a trained checkpoint for the same kind of video.
The normal first run is: draw ROI, sample ROI frames, annotate ant boxes, export
COCO, train the detector, then track new video with that checkpoint.

## Notes

- Train on ROI images; infer on matching ROI rectangles.
- Use multiple ROIs per video when ants appear in separate static regions.
- Keep the ROI tight enough that ants remain large after resizing, but include
  enough edge context that a full ant can be labeled.
- Sample hard cases: ROI edges, glare, shadows, stationary ants, crossings, and
  empty frames containing ant-like marks.
- Balance annotation effort by actual ant appearances per ROI. An ROI that is
  always empty is useful negative data but cannot provide a recall estimate.
- The first run with `--initialization detector` may download torchvision weights.
