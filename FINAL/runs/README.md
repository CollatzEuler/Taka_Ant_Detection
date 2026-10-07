# Saved runs

This directory preserves the datasets and run artifacts copied from
`MLPipeline/runs`. References inside the copied JSON and CSV files now point to
`FINAL/runs`. Keep each new annotation round in a separate subdirectory so its
manifest, images, editable project, and COCO export stay together.

| Item | Contents |
| --- | --- |
| `Type1_Vid1_roi.json` | Fixed search regions for `Videos/Type1_Vid1.mp4`; companion PNGs preview them. |
| `Type1_Vid1_labels/` | Sampled ROI images, frame manifest, editable labels, and COCO export. |
| `Type1_Vid1_suggested_round2/` | Model-suggested annotation round. |
| `long_video_random_labels/` | Historical random ROI annotation set; its source `long_video.mp4` is not in `Videos/`. |
| `*_smoke/`, `benchmark_*/`, `*_track_*/`, `*_eval_*/` | Historical checks and measurements. |

Use the existing `Type1_Vid1_labels/selected_frames` and
`Type1_Vid1_labels/annotations_coco.json` directly for training or examples.
For a new dataset, follow [`../Annotation/README.md`](../Annotation/README.md).
