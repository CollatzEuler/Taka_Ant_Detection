# Ant Annotation Guide

This guide covers the annotation portion of the ROI ant-detection pipeline: preparing ROI crops, drawing or correcting ant boxes, saving work, exporting COCO labels, and preparing those labels for training. It is written for someone who already understands bounding-box annotation but has not used this repository.

Run all commands from the `Taka` project root. The examples use PowerShell.

## What this annotator produces

The annotation tool is a local Tkinter GUI for a single detection class: `ant`.

The workflow uses three related files:

- `selected_frames/`: PNG crops presented to the annotator.
- `frame_manifest.csv`: the source video, source frame, ROI, crop origin, and image dimensions for every PNG.
- `annotations.json`: the editable annotation project and resume file.

The final training label file is `annotations_coco.json`. Boxes are stored as COCO `[x, y, width, height]` values in **ROI-crop coordinates**, not full-video coordinates. The manifest and project preserve the ROI offset needed to relate a crop to the source video.

Do not rename or move a run directory after creating its manifest or annotation project. These files contain absolute paths to the images and source data.

## The order of operations

Use this sequence for a new dataset:

1. Set up and test the Python environment.
2. Create an ROI definition that establishes the crop size and/or fixed search area.
3. Sample annotation crops using one of the routes below.
4. Review the sampled images before labeling.
5. Open the GUI and annotate every crop, including confirming true empty crops.
6. Save the editable project.
7. Export COCO and perform label QA.
8. Split the dataset into temporal training and validation blocks.
9. Optionally merge multiple independently exported datasets.

Model-suggested annotation is a later-round route. It requires an existing trained checkpoint and does not replace the first manually labeled dataset.

## 1. Set up the environment

The tested environment is defined in the repository-level `environment.yaml`:

```powershell
conda env create -f .\environment.yaml
conda activate taka-env
python .\MLPipeline\pipeline.py --help
```

If the environment already exists:

```powershell
conda env update -f .\environment.yaml --prune
conda activate taka-env
```

The annotation steps need OpenCV, NumPy, and Tkinter. A successful `pipeline.py --help` checks the main entry point; opening the annotation GUI later also confirms that Tkinter is available.

Use a new output directory for each annotation round. The samplers write files into the requested directory and are not intended to version or protect an existing project.

## 2. Define the ROI

An ROI is a rectangular crop from the source video. Fixed-ROI sampling uses the rectangle as both the annotation and inference search area. Random-ROI sampling uses one or more reference rectangles primarily to establish suitable crop dimensions.

Create one or more rectangles on a representative video frame:

```powershell
python .\MLPipeline\pipeline.py select-roi -- `
  --video ".\Videos\Type1_Vid1.mp4" `
  --output ".\MLPipeline\runs\Type1_Vid1_roi.json"
```

ROI window controls:

- Left-drag: add a rectangle.
- `u`: remove the most recently added rectangle.
- `c`: clear every rectangle.
- `s`: save the ROI JSON, representative frame, and preview image.
- `q` or `Esc`: close without saving.

Review `Type1_Vid1_roi_preview.png`. Each fixed ROI should cover every place in which ants must be detected, remain tight enough that an ant is not made unnecessarily small during model resizing, and leave enough context to draw a complete box near its edges.

## 3. Sample annotation images

Choose exactly one sampling route for an annotation round.

### Route A: fixed ROI crops

Use this route when the final detector will search known, static rectangles.

```powershell
python .\MLPipeline\pipeline.py sample-roi -- `
  --videos ".\Videos\Type1_Vid1.mp4" `
  --roi-jsons ".\MLPipeline\runs\Type1_Vid1_roi.json" `
  --output ".\MLPipeline\runs\Type1_Vid1_labels" `
  --count 300 `
  --frame-step 90 `
  --duration-seconds 300
```

The sampler scans every 90th source frame in the selected five-minute window, clusters the ROI crops for visual diversity, and writes up to 300 representative crops. `--count` is the total across all videos and ROIs, not a per-ROI count.

Useful time-window options are `--start-seconds`, `--end-seconds`, and `--duration-seconds`. If no ROI JSON is passed, the sampler produces full-frame images. With one video, multiple ROI JSON files may be supplied; with multiple videos, supply either one shared ROI JSON or one ROI JSON per video.

### Route B: random allowed-area crops

Use this route for a long, static-camera recording when training crops should cover many locations instead of only fixed rectangles. The final tracker must still search all required fixed areas; random spatial sampling is only a training-data strategy.

First paint places that must never be sampled:

```powershell
python .\MLPipeline\pipeline.py select-exclusions -- `
  --video ".\Videos\long_video.mp4" `
  --output ".\MLPipeline\runs\long_video_exclusions.json" `
  --frame middle
```

Exclusion-mask controls:

- Left-drag: paint an excluded area black.
- Right-drag: restore an allowed area.
- `[` and `]`: decrease or increase brush size.
- `u`: undo the last stroke.
- `c`: clear the mask.
- `s`: save.
- `q` or `Esc`: close without saving.

Then sample crops. Their dimensions remain close to the sizes in the reference ROI JSON:

```powershell
python .\MLPipeline\pipeline.py sample-random-roi -- `
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

This example reserves 150 crops for the first ten minutes and distributes the remaining 250 over later, stratified times. Half the candidates are motion-guided and half use random allowed locations. Empty crops and hard ant-like negatives are expected.

Review `selection_preview.jpg` before beginning annotation. If bad regions dominate, improve the exclusion mask and sample into a **new** run directory. If important rare appearances are absent, increase `--candidate-multiplier` or widen the time window.

### Route C: model-suggested boxes

Use this only after training a preliminary detector. It selects high-value fixed-ROI frames and creates `annotations.json` with proposed boxes already present.

```powershell
python .\MLPipeline\pipeline.py sample-suggested -- `
  --video ".\Videos\Type1_Vid1.mp4" `
  --roi-json ".\MLPipeline\runs\Type1_Vid1_roi.json" `
  --checkpoint ".\MLPipeline\checkpoints\ant_detector_mobilenet_best.pt" `
  --output ".\MLPipeline\runs\Type1_Vid1_suggested_round2" `
  --count 200 `
  --frame-step 30 `
  --duration-seconds 300 `
  --score-threshold 0.12
```

The low suggestion threshold is intentional: the annotator must remove false positives, add missed ants, and redraw inaccurate boxes. Do not treat a proposed box as accepted merely because it is present.

## 4. Inspect the sampled run

Before launching the GUI, check the run directory:

```text
MLPipeline/runs/Type1_Vid1_labels/
|-- selected_frames/
|-- frame_manifest.csv
`-- sampler_manifest.json
```

Random sampling also produces `selection_preview.jpg`. Suggested sampling produces `annotations.json` and `suggested_manifest.json` immediately.

Confirm the following before annotation begins:

- The images open and show the intended regions.
- Ants are large enough to box consistently.
- The set covers the relevant times, ROIs, lighting, focus, glare, shadows, arena edges, crossings, stationary ants, and empty scenes.
- The number of rows in `frame_manifest.csv` matches the expected image count.
- The output directory is dedicated to this round and is backed up if annotation has already begun.

Sampling diversity is not proof of class balance. Browse enough crops to check that useful positive examples actually occur in each important ROI.

## 5. Launch the annotation GUI

For fixed or random sampling, the first launch creates the project in memory from `frame_manifest.csv`:

```powershell
python .\MLPipeline\pipeline.py annotate -- `
  --images-dir ".\MLPipeline\runs\Type1_Vid1_labels\selected_frames" `
  --manifest ".\MLPipeline\runs\Type1_Vid1_labels\frame_manifest.csv" `
  --project ".\MLPipeline\runs\Type1_Vid1_labels\annotations.json"
```

For a suggested round, use that round's pre-created project:

```powershell
python .\MLPipeline\pipeline.py annotate -- `
  --images-dir ".\MLPipeline\runs\Type1_Vid1_suggested_round2\selected_frames" `
  --manifest ".\MLPipeline\runs\Type1_Vid1_suggested_round2\frame_manifest.csv" `
  --project ".\MLPipeline\runs\Type1_Vid1_suggested_round2\annotations.json"
```

If `--project` already exists, the GUI loads it as-is. It does not rebuild it from a changed manifest, and it always reopens at image 1. Use a different project filename for a deliberately separate annotation version.

## 6. Annotate each crop

The code supports only one class, `ant`. Unless the study's written labeling protocol says otherwise, use one tight axis-aligned box around the visible extent of each recognizable ant. Clip boxes at image boundaries rather than estimating pixels outside the crop. Do not label glare, stains, debris, shadows, or other ant-like marks. Preserve examples containing only these distractors as empty negative images.

The software does not encode a policy for severe occlusion, fragments, ambiguous objects, or identity continuity. Agree on those cases before production annotation and apply the decision consistently. The exporter always writes `iscrowd: 0`, so a single crowd box should not be used as a substitute for individual ant boxes.

GUI controls:

| Action | Control |
| --- | --- |
| Draw a new box | Left-drag on the image |
| Select an existing box | Click its row in the **Boxes** list |
| Delete the selected box | `Delete` or **Delete Box** |
| Remove the most recently added box | `u` or **Undo** |
| Next image | `n` or **Next** |
| Previous image | `p` or **Prev** |
| Assign active identity 1-9 | Press `1` through `9` |
| Clear the active identity | Press `0` |
| Save the editable project | `s` or **Save** |
| Export COCO | `e` or **Export COCO** |
| Save and quit | `q` or close the window |

Manual boxes are green, model-suggested boxes are blue, and the selected box is yellow. The GUI cannot drag or resize an existing box; delete an inaccurate box and redraw it.

Identity is optional metadata. Pressing a digit sets the ID for subsequently drawn boxes; if a box is selected, it also changes that box's ID. All identities still export as the same `ant` detection class and do not affect detector training. Use IDs only if the annotation protocol requires them.

For every image:

1. Inspect the complete crop, including all four borders.
2. In a suggested project, evaluate every blue box and delete false positives.
3. Draw one box for every missed recognizable ant.
4. Delete and redraw boxes with incorrect extent.
5. Leave the image with zero boxes if it is truly empty.
6. Move to the next image with `n`.
7. Save regularly with `s`.

Important GUI behavior:

- Moving with `n` or `p` does **not** save the project to disk.
- `q` and the window close button save before exiting.
- Exporting COCO does not first save `annotations.json`; press `s` before `e` so the editable project and export represent the same state.
- The status bar's `Labeled images` value counts images containing at least one box. Correctly reviewed empty images remain shown as unlabeled because the project format has no separate reviewed flag. Keep an external completion log if multiple annotators need auditable empty-image review.
- Very small drags are ignored. The minimum accepted rectangle is four displayed pixels in both dimensions.

## 7. Save, resume, and export

To resume, rerun the same `annotate` command with the same project path. The GUI starts at image 1, so record the last completed image filename or index outside the tool.

After the final review, press `s`, then press `e`. The GUI writes `annotations_coco.json` next to `annotations.json`.

The equivalent explicit export is:

```powershell
python .\MLPipeline\pipeline.py export-coco -- `
  --project ".\MLPipeline\runs\Type1_Vid1_labels\annotations.json" `
  --output ".\MLPipeline\runs\Type1_Vid1_labels\annotations_coco.json"
```

Do not use `--skip-empty` for the final training or evaluation dataset. Empty images teach the detector that background structures and ant-like artifacts are not ants. `--skip-empty` is only useful for a temporary positive-only diagnostic export.

## 8. Label Count Check

Quick PowerShell count check:

```powershell
$projectData = Get-Content ".\MLPipeline\runs\Type1_Vid1_labels\annotations.json" -Raw | ConvertFrom-Json
$cocoData = Get-Content ".\MLPipeline\runs\Type1_Vid1_labels\annotations_coco.json" -Raw | ConvertFrom-Json
$projectBoxCount = ($projectData.images.annotations | Measure-Object).Count
[pscustomobject]@{
  ProjectImages = $projectData.images.Count
  CocoImages = $cocoData.images.Count
  ProjectBoxes = $projectBoxCount
  CocoBoxes = $cocoData.annotations.Count
}
```

The image and box counts should agree unless a diagnostic export deliberately used `--skip-empty`.