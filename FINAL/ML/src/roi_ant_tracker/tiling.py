from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def tile_origins(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or tile_size >= length:
        return [0]
    stride = max(1, tile_size - overlap)
    origins = list(range(0, max(1, length - tile_size + 1), stride))
    final_origin = length - tile_size
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return origins


def iter_tiles(
    frame: np.ndarray, tile_size: int, overlap: int
) -> Iterator[tuple[np.ndarray, int, int]]:
    height, width = frame.shape[:2]
    effective_tile = max(height, width) if tile_size <= 0 else tile_size
    for y in tile_origins(height, effective_tile, overlap):
        for x in tile_origins(width, effective_tile, overlap):
            yield frame[y : min(y + effective_tile, height), x : min(x + effective_tile, width)], x, y
