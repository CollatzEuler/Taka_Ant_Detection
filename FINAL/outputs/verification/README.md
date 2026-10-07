# Verification run

This folder records small checks run from the `FINAL` entry point against
`Videos/Type1_Vid1.mp4` with the existing `python_env` environment:

- `roi.json` and its PNGs: non-interactive preprocessing ROI selection.
- `exclusions.json` and its PNGs: saved and previewed allowed-area mask.
- `labels/`: two sampled ROI frames and a frame manifest.
- `random_labels/`: two random ROI crops sampled with the exclusion mask.
- `labels/gui_smoke_annotations.json` and its COCO export: the Tk/Pillow GUI
  loaded the sampled images in `carina_venv`, saved a project, and exported one
  test box.
- `annotations_coco.json`: export of the copied editable annotation project
  (250 images, 47 boxes).
- `train_coco.json` and `validation_coco.json`: temporal split of that export
  (200 and 50 images).
- `tracking/`: short CPU tracking run with the copied checkpoint (two frames).
- `evaluation/`: detector evaluation output for the 50-image split.
- `training_smoke_coco.json` and the training log: one CPU epoch on two
  annotated images with random initialization.

The copied checkpoint was trained on these source labels, so the evaluation
numbers are a functional check, not a held-out accuracy estimate.
