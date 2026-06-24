from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from planner.f110_self_play.backend import TrackMap


def load_track_map(
    map_path: str,
    map_ext: str = ".png",
    num_beams: int = 1080,
    fov: float = 4.7,
    max_range: float = 30.0,
    eps: float = 0.0001,
    theta_dis: int = 2000,
    car_length: float = 0.58,
    car_width: float = 0.31,
) -> tuple[TrackMap, float, float]:
    map_yaml = Path(map_path)
    map_img = map_yaml.with_suffix(map_ext)

    import yaml

    with open(map_yaml) as f:
        meta = cast(dict[str, Any], yaml.safe_load(f))

    resolution = float(meta["resolution"])
    origin = [float(v) for v in cast(list[float], meta["origin"])]

    img = Image.open(map_img).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    bitmap = np.asarray(img, dtype=np.float32)
    bitmap = np.where(bitmap <= 128, 0.0, 255.0)

    import scipy.ndimage

    dt = np.asarray(scipy.ndimage.distance_transform_edt(bitmap), dtype=np.float32)
    dt *= resolution

    from numpy.typing import NDArray

    dt_flat: NDArray[np.float32] = dt.astype(np.float32).ravel()

    track = TrackMap()
    track.height = int(dt.shape[0])
    track.width = int(dt.shape[1])
    track.resolution = float(resolution)
    track.orig_x = float(origin[0])
    track.orig_y = float(origin[1])
    track.orig_c = float(math.cos(origin[2]))
    track.orig_s = float(math.sin(origin[2]))
    track.dt = dt_flat.tolist()
    track.theta_dis = int(theta_dis)
    track.num_beams = int(num_beams)
    track.fov = float(fov)
    track.max_range = float(max_range)
    track.eps = float(eps)
    track.compute_scan_tables()

    dist_sides = car_width / 2.0
    dist_fr = car_length / 2.0
    scan_ang_incr = fov / float(num_beams - 1)
    side_distances = np.empty((num_beams,), dtype=np.float32)
    for i in range(num_beams):
        angle = -fov / 2.0 + float(i) * scan_ang_incr
        if angle > 0.0:
            if angle < math.pi / 2.0:
                to_side = dist_sides / math.sin(angle)
                to_fr = dist_fr / math.cos(angle)
                side_distances[i] = float(min(to_side, to_fr))
            else:
                to_side = dist_sides / math.cos(angle - math.pi / 2.0)
                to_fr = dist_fr / math.sin(angle - math.pi / 2.0)
                side_distances[i] = float(min(to_side, to_fr))
        else:
            if angle > -math.pi / 2.0:
                to_side = dist_sides / math.sin(-angle)
                to_fr = dist_fr / math.cos(-angle)
                side_distances[i] = float(min(to_side, to_fr))
            else:
                to_side = dist_sides / math.cos(-angle - math.pi / 2.0)
                to_fr = dist_fr / math.sin(-angle - math.pi / 2.0)
                side_distances[i] = float(min(to_side, to_fr))

    track.side_distances = side_distances.tolist()
    track.ttc_thresh = 0.005

    return track, car_length, car_width
