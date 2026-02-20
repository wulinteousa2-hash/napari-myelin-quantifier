# src/napari_myelin_quantifier/_axon_quant.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max


@dataclass
class QuantParams:
    """
    Parameters for myelin ring instance labeling + quantification from a 2D binary mask.

    Assumptions:
      - input is a 2D mask (Image or Labels) where myelin pixels are >0
      - background is 0

    Outputs:
      - instance label image (1..N)
      - dataframe with centroid/bbox/areas/euler/touches_border
    """

    px_size_um: float = 0.0  # set >0 to compute µm² columns (e.g. 0.02)

    min_ring_area: int = 80
    close_radius: int = 1
    open_radius: int = 0

    lumen_min_area: int = 50
    seed_dilation: int = 2  # used for expand_labels when use_peak_seeds=False

    use_peak_seeds: bool = False
    peak_min_distance: int = 10

    euler_connectivity: int = 2  # 2 -> 8-connectivity in 2D

    exclude_border_objects: bool = (
        False  # if True, drop objects touching border
    )


def quantify_myelin_rings_from_mask(
    mask_2d: np.ndarray,
    params: QuantParams,
    euler_keep: Sequence[int] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    arr = np.asarray(mask_2d)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask. Got shape: {arr.shape}")

    mask = arr > 0

    # Cleanup
    mask = morphology.remove_small_objects(mask, params.min_ring_area)
    if params.close_radius > 0:
        mask = morphology.closing(mask, morphology.disk(params.close_radius))
    if params.open_radius > 0:
        mask = morphology.opening(mask, morphology.disk(params.open_radius))
    mask = mask.astype(bool)

    # Lumen seeds (holes)
    filled = ndi.binary_fill_holes(mask)
    lumen = filled & (~mask)
    lumen = morphology.remove_small_objects(lumen, params.lumen_min_area)
    lumen_labels = measure.label(lumen, connectivity=params.euler_connectivity)

    # Markers
    if params.use_peak_seeds:
        dist = ndi.distance_transform_edt(mask)
        coords = peak_local_max(
            dist,
            min_distance=params.peak_min_distance,
            labels=mask,
        )
        markers = np.zeros(mask.shape, dtype=np.int32)
        if coords.size > 0:
            markers[coords[:, 0], coords[:, 1]] = np.arange(
                1, coords.shape[0] + 1, dtype=np.int32
            )
    else:
        if params.seed_dilation > 0:
            markers = segmentation.expand_labels(
                lumen_labels, distance=int(params.seed_dilation)
            )
        else:
            markers = lumen_labels.astype(np.int32)
        markers = (markers * mask.astype(np.int32)).astype(np.int32)

    if int(markers.max()) == 0:
        raise RuntimeError(
            "No seeds found. Try lowering lumen_min_area, increasing seed_dilation, "
            "or set use_peak_seeds=True."
        )

    # Watershed instance split
    dist = ndi.distance_transform_edt(mask)
    inst = segmentation.watershed(-dist, markers=markers, mask=mask).astype(np.int32)

    H, W = mask.shape
    props = measure.regionprops(inst)

    euler_keep_set = set(euler_keep) if euler_keep is not None else None

    kept_props = []
    for p in props:
        rid = int(p.label)
        if rid == 0:
            continue

        minr, minc, maxr, maxc = p.bbox
        touches_border = (minr == 0) or (minc == 0) or (maxr == H) or (maxc == W)
        if params.exclude_border_objects and touches_border:
            continue

        # IMPORTANT: compute on cropped object image (fast), not full HxW mask
        obj_img = p.image  # boolean array within bbox
        chi = int(measure.euler_number(obj_img, connectivity=params.euler_connectivity))
        if euler_keep_set is not None and chi not in euler_keep_set:
            continue

        cy, cx = p.centroid
        kept_props.append((rid, float(cy), float(cx), chi, touches_border, p))

    # Sort by centroid (same as your behavior)
    kept_props.sort(key=lambda t: (t[1], t[2]))

    # Vectorized relabel: LUT remap instead of N full-image comparisons
    max_id = int(inst.max())
    lut = np.zeros(max_id + 1, dtype=np.int32)
    for new_id, (old_id, *_rest) in enumerate(kept_props, start=1):
        lut[old_id] = np.int32(new_id)

    inst_keep = lut[inst]  # one pass over the image

    # Build dataframe without re-running regionprops and without full-image masks
    rows = []
    for new_id, (old_id, cy, cx, chi, touches_border, p) in enumerate(kept_props, start=1):
        minr, minc, maxr, maxc = p.bbox

        ring_area_px = int(p.area)

        # Fill holes on cropped image (fast)
        obj_filled = ndi.binary_fill_holes(p.image)
        filled_area_px = int(obj_filled.sum())
        lumen_area_px = int(filled_area_px - ring_area_px)

        row = {
            "ring_id": int(new_id),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "bbox_x0": int(minc),
            "bbox_y0": int(minr),
            "bbox_x1": int(maxc),
            "bbox_y1": int(maxr),
            "ring_area_px": ring_area_px,
            "lumen_area_px": lumen_area_px,
            "filled_area_px": filled_area_px,
            "euler": int(chi),
            "touches_border": bool(touches_border),
        }

        if params.px_size_um and params.px_size_um > 0:
            px2 = float(params.px_size_um) ** 2
            row["ring_area_um2"] = float(ring_area_px) * px2
            row["lumen_area_um2"] = float(lumen_area_px) * px2
            row["filled_area_um2"] = float(filled_area_px) * px2

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("ring_id").reset_index(drop=True)
    return inst_keep, df
