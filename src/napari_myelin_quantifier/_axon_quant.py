# src/napari_myelin_quantifier/_axon_quant.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import morphology
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border


@dataclass
class QuantParams:
    px_size_um: float = 0.02

    v_thresh: float = 0.15
    min_myelin_obj: int = 30
    close_radius: int = 3
    open_radius: int = 1

    min_axon_area: int = 150
    min_solidity: float = 0.60

    remove_border_objects: bool = True
    edge_margin_px: int = 8
    auto_crop: bool = True

    # Optional filter (your previous idea)
    require_single_myelin_component: bool = True
    myelin_contact_dilate: int = 2  # dilation radius to test myelin neighbors

    # ✅ Robust filter: require an enclosing myelin ring around candidate axon
    ring_width_px: int = 3
    min_ring_coverage: float = 0.70


def _find_content_crop(
    img_rgb: np.ndarray, v_thresh: float = 0.15, frac_thresh: float = 0.15
):
    """Auto-crop away UI/borders by finding dense colored region."""
    hsv = rgb2hsv(img_rgb / 255.0)
    v = hsv[..., 2]
    mask = v > v_thresh

    col_frac = mask.mean(axis=0)
    row_frac = mask.mean(axis=1)

    xs = np.where(col_frac > frac_thresh)[0]
    ys = np.where(row_frac > frac_thresh)[0]

    if len(xs) < 10 or len(ys) < 10:
        return img_rgb, (0, 0)

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return img_rgb[y0:y1, x0:x1], (y0, x0)


def _touches_margin(prop, H: int, W: int, margin: int) -> bool:
    minr, minc, maxr, maxc = prop.bbox
    return (
        (minr <= margin)
        or (minc <= margin)
        or (maxr >= H - margin)
        or (maxc >= W - margin)
    )


def _count_touching_myelin_components(
    axon_mask: np.ndarray,
    myelin_labels: np.ndarray,
    dilate_radius: int = 2,
) -> int:
    """
    Count how many distinct connected myelin components touch this axon region.
    True axons should touch exactly ONE myelin ring component.

    NOTE: This can fail when closing merges nearby rings into one component.
    Kept for compatibility, but ring-coverage is the main robust filter.
    """
    if dilate_radius > 0:
        dil = morphology.binary_dilation(
            axon_mask, morphology.disk(dilate_radius)
        )
    else:
        dil = axon_mask

    touched = np.unique(myelin_labels[dil])
    touched = touched[touched != 0]
    return int(len(touched))


def _ring_coverage_ok(
    obj_mask: np.ndarray,
    myelin_mask: np.ndarray,
    ring_width_px: int,
    min_ring_coverage: float,
) -> bool:
    """
    Validate that a candidate axon is surrounded by myelin.

    Compute a thin ring around the object (dilate - object),
    and measure fraction of ring pixels that are myelin.
    """
    if ring_width_px <= 0:
        return True

    selem = morphology.disk(ring_width_px)
    dil = morphology.binary_dilation(obj_mask, selem)
    ring = dil & (~obj_mask)

    denom = int(ring.sum())
    if denom == 0:
        return False

    coverage = float((myelin_mask & ring).sum()) / float(denom)
    return coverage >= float(min_ring_coverage)


def quantify_myelinated_axons(img_rgb: np.ndarray, params: QuantParams):
    """
    Returns:
      labels_full: int32 label image (axon_id labels) [cropped if auto_crop True]
      overlay_full: uint8 RGB overlay with axon boundaries in white [cropped if auto_crop True]
      df: measurements table (filtered + axon_id)
      offsets: (y_off, x_off) from autocrop
    """
    img = img_rgb

    # optional crop
    if params.auto_crop:
        img_c, (y_off, x_off) = _find_content_crop(
            img, v_thresh=params.v_thresh, frac_thresh=0.15
        )
    else:
        img_c, (y_off, x_off) = img, (0, 0)

    hsv = rgb2hsv(img_c / 255.0)
    v = hsv[..., 2]

    # myelin mask (binary)
    myelin = v > params.v_thresh
    myelin = morphology.remove_small_objects(myelin, params.min_myelin_obj)

    if params.close_radius > 0:
        myelin = morphology.closing(
            myelin, morphology.disk(params.close_radius)
        )
    if params.open_radius > 0:
        myelin = morphology.opening(
            myelin, morphology.disk(params.open_radius)
        )

    # label myelin components (used by optional filter)
    myelin_lab = label(myelin)

    # fill holes (axon interiors become "holes" inside myelin)
    filled = ndi.binary_fill_holes(myelin)

    # axon interior candidates
    axon = filled & (~myelin)
    axon = morphology.remove_small_objects(axon, params.min_axon_area)

    if params.remove_border_objects:
        axon = clear_border(axon)

    lab = label(axon)
    props = regionprops(lab)

    H, W = axon.shape
    rows = []

    # We'll keep only accepted old labels
    accepted_old_labels: list[int] = []

    for p in props:
        if params.edge_margin_px > 0 and _touches_margin(
            p, H, W, params.edge_margin_px
        ):
            continue

        # basic filters
        if p.area < params.min_axon_area:
            continue
        if p.solidity < params.min_solidity:
            continue

        # build region mask in full cropped image coords (H,W)
        minr, minc, maxr, maxc = p.bbox
        region_mask = np.zeros((H, W), dtype=bool)
        region_mask[minr:maxr, minc:maxc] = p.image

        # optional old filter (kept)
        if params.require_single_myelin_component:
            n_touch = _count_touching_myelin_components(
                region_mask,
                myelin_lab,
                dilate_radius=params.myelin_contact_dilate,
            )
            if n_touch != 1:
                continue

        # ✅ robust filter: must be surrounded by myelin
        if not _ring_coverage_ok(
            region_mask,
            myelin_mask=myelin,
            ring_width_px=params.ring_width_px,
            min_ring_coverage=params.min_ring_coverage,
        ):
            continue

        # measurements
        area_px = float(p.area)
        area_um2 = area_px * (params.px_size_um**2)
        eq_diam_um = float(p.equivalent_diameter) * params.px_size_um

        cy, cx = p.centroid
        rows.append(
            {
                "raw_label": int(p.label),
                "area_px": area_px,
                "area_um2": area_um2,
                "eq_diam_um": eq_diam_um,
                "eccentricity": float(p.eccentricity),
                "solidity": float(p.solidity),
                "cy_crop": float(cy),
                "cx_crop": float(cx),
                "cy_img": float(cy + y_off),
                "cx_img": float(cx + x_off),
            }
        )
        accepted_old_labels.append(int(p.label))

    df = pd.DataFrame(rows)

    # rebuild labels to be 1..N in sorted spatial order
    labels_full = np.zeros_like(lab, dtype=np.int32)

    if len(df) > 0:
        df = df.sort_values(["cy_img", "cx_img"]).reset_index(drop=True)
        df.insert(0, "axon_id", np.arange(1, len(df) + 1, dtype=int))

        old_to_new = {
            int(r.raw_label): int(r.axon_id)
            for r in df.itertuples(index=False)
        }
        for old_lbl, new_lbl in old_to_new.items():
            labels_full[lab == old_lbl] = new_lbl
    else:
        df.insert(0, "axon_id", [])

    # overlay boundaries
    axon_kept = labels_full > 0
    boundary = morphology.binary_dilation(
        axon_kept, morphology.disk(1)
    ) ^ morphology.binary_erosion(axon_kept, morphology.disk(1))
    overlay = img_c.copy()
    overlay[boundary] = [255, 255, 255]

    return labels_full, overlay, df, (y_off, x_off)
