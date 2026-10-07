from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "Annotation" / "split_coco.py"
SPEC = importlib.util.spec_from_file_location("split_coco", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SplitCocoTests(unittest.TestCase):
    def test_keeps_rois_from_the_same_temporal_block_together(self) -> None:
        images = []
        annotations = []
        next_id = 1
        for frame in (0, 100, 700, 800, 1300, 1400):
            for roi in ("left", "right"):
                images.append(
                    {
                        "id": next_id,
                        "file_name": f"{next_id}.png",
                        "video_path": "video.mp4",
                        "source_frame_index": frame,
                        "roi_id": roi,
                    }
                )
                annotations.append({"id": next_id, "image_id": next_id, "bbox": [0, 0, 5, 5]})
                next_id += 1
        train, validation = MODULE.split_coco_payload(
            {"images": images, "annotations": annotations, "categories": []},
            validation_fraction=0.34,
            block_frames=600,
            seed=3,
        )

        train_blocks = {image["source_frame_index"] // 600 for image in train["images"]}
        validation_blocks = {image["source_frame_index"] // 600 for image in validation["images"]}
        self.assertFalse(train_blocks.intersection(validation_blocks))
        self.assertEqual(len(train["annotations"]) + len(validation["annotations"]), len(annotations))


if __name__ == "__main__":
    unittest.main()
