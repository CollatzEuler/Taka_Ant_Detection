# Preprocessing

Preprocessing defines where the detector should search. The original video is
never rewritten. Run these commands from the `Taka` folder with the tested
`python_env` (see [`../README.md`](../README.md)).

```powershell
$python = "$HOME\miniconda3\envs\python_env\python.exe"
```

## Fixed search ROIs

```powershell
& $python .\FINAL\pipeline.py select-roi -- `
  --video .\Videos\Type1_Vid1.mp4 `
  --output .\FINAL\runs\Type1_Vid1_roi.json
```

Drag a rectangle for each region, press `s` to save, `u` to undo, or `q` to
leave without saving. For a scripted selection, add one or more
`--rect X,Y,WIDTH,HEIGHT` arguments. The JSON stores full-frame coordinates;
the two companion PNGs show the chosen frame and rectangle preview. Use
`preview-roi` to review a saved definition against the video.

## Allowed-area mask for random annotation crops

```powershell
& $python .\FINAL\pipeline.py select-exclusions -- `
  --video .\Videos\Type1_Vid1.mp4 `
  --output .\FINAL\runs\Type1_Vid1_exclusions.json
```

Paint forbidden areas with the left mouse button and restore them with the
right. Press `s` to save. `sample-random-roi` uses the saved mask to avoid
forbidden areas; it still needs a reference ROI JSON for crop dimensions.
Recreate the mask if the camera moves or the video resolution changes.

The copied ROI and mask definitions in `../runs/` point to the copied FINAL
artifacts. The video itself remains in `../../Videos/`.
