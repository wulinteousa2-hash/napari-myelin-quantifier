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

    # Myelin extraction / cleanup
    v_thresh: float = 0.15
    min_myelin_obj: int = 30
    close_radius: int = 3
    open_radius: int = 1

    # Axon interior filtering
    min_axon_area: int = 150
    min_solidity: float = 0.60

    # Edge handling
    remove_border_objects: bool = True
    edge_margin_px: int = 8

    # Optional crop for screenshots with UI borders
    auto_crop: bool = True
    crop_frac_thresh: float = 0.15  # fraction threshold used for content crop


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


def _touches_margin(bbox, H: int, W: int, margin: int) -> bool:
    minr, minc, maxr, maxc = bbox
    return (
        (minr <= margin)
        or (minc <= margin)
        or (maxr >= H - margin)
        or (maxc >= W - margin)
    )


def quantify_myelinated_axons(
    img_rgb: np.ndarray,
    params: QuantParams,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, tuple[int, int]]:
    """
    Returns:
      labels_full: 2D int32 label image in full image coordinates (axon_id 1..N; 0=bg)
      overlay_full: 3D uint8 RGB overlay in full image coordinates (boundary drawn white)
      df: measurements table; axon_id matches labels/text
      (y_off, x_off): crop offset applied
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] not in (3, 4):
        raise ValueError(
            f"Expected 2D RGB/RGBA image with shape (Y, X, 3/4). Got: {img_rgb.shape}"
        )

    if img_rgb.shape[2] == 4:
        img_rgb = img_rgb[..., :3]

    img_rgb = img_rgb.astype(np.uint8)

    # Optional crop (useful for screenshots like your example)
    if params.auto_crop:
        img_c, (y_off, x_off) = _find_content_crop(
            img_rgb,
            v_thresh=params.v_thresh,
            frac_thresh=params.crop_frac_thresh,
        )
    else:
        img_c, (y_off, x_off) = img_rgb, (0, 0)

    Hc, Wc, _ = img_c.shape

    # -----------------------
    # MYELIN MASK
    # -----------------------
    hsv = rgb2hsv(img_c / 255.0)
    v = hsv[..., 2]
    myelin = v > params.v_thresh

    myelin = morphology.remove_small_objects(myelin, params.min_myelin_obj)
    myelin = morphology.binary_closing(
        myelin, morphology.disk(params.close_radius)
    )
    myelin = morphology.binary_opening(
        myelin, morphology.disk(params.open_radius)
    )

    # -----------------------
    # FILL HOLES (NO AREA CAP)
    # -----------------------
    filled = ndi.binary_fill_holes(myelin)
    axon = filled & (~myelin)

    axon = morphology.remove_small_objects(axon, params.min_axon_area)

    if params.remove_border_objects:
        axon = clear_border(axon)

    # -----------------------
    # LABEL + MEASURE
    # -----------------------
    lab_raw = label(axon)
    props = regionprops(lab_raw)

    rows = []
    for p in props:
        if params.edge_margin_px > 0 and _touches_margin(
            p.bbox, axon.shape[0], axon.shape[1], params.edge_margin_px
        ):
            continue

        area_px = float(p.area)
        area_um2 = area_px * (params.px_size_um**2)
        eq_diam_um = float(p.equivalent_diameter) * params.px_size_um

        rows.append(
            {
                "raw_label": int(p.label),
                "area_px": area_px,
                "area_um2": area_um2,
                "eq_diam_um": eq_diam_um,
                "eccentricity": float(p.eccentricity),
                "solidity": float(p.solidity),
                "cy_crop": float(p.centroid[0]),
                "cx_crop": float(p.centroid[1]),
                "cy_img": float(p.centroid[0] + y_off),
                "cx_img": float(p.centroid[1] + x_off),
            }
        )

    df = pd.DataFrame(rows)

    # Filter by shape (your “verified axons/myelin” gate)
    if len(df) > 0:
        df_f = df[
            (df["area_px"] >= params.min_axon_area)
            & (df["solidity"] >= params.min_solidity)
        ].copy()
        df_f = df_f.sort_values(["cy_img", "cx_img"]).reset_index(drop=True)
        df_f.insert(0, "axon_id", np.arange(1, len(df_f) + 1))
    else:
        df_f = df.copy()
        df_f.insert(0, "axon_id", [])

    # Relabel into axon_id space (1..N) inside crop coords
    labels_crop = np.zeros((Hc, Wc), dtype=np.int32)
    if len(df_f) > 0:
        raw_to_new = {
            int(r.raw_label): int(r.axon_id)
            for r in df_f.itertuples(index=False)
        }
        # Map raw labels -> new ids
        # Fast remap via LUT
        max_raw = int(lab_raw.max())
        lut = np.zeros(max_raw + 1, dtype=np.int32)
        for k, v in raw_to_new.items():
            if 0 <= k <= max_raw:
                lut[k] = v
        labels_crop = lut[lab_raw].astype(np.int32)

    # Boundary overlay (crop coords)
    axon_crop_mask = labels_crop > 0
    boundary = morphology.dilation(
        axon_crop_mask, morphology.disk(1)
    ) ^ morphology.erosion(axon_crop_mask, morphology.disk(1))
    overlay_crop = img_c.copy()
    overlay_crop[boundary] = [255, 255, 255]

    # Paste back into full-size arrays
    H, W, _ = img_rgb.shape
    labels_full = np.zeros((H, W), dtype=np.int32)
    overlay_full = img_rgb.copy()

    labels_full[y_off : y_off + Hc, x_off : x_off + Wc] = labels_crop
    overlay_full[y_off : y_off + Hc, x_off : x_off + Wc] = overlay_crop

    return labels_full, overlay_full, df_f, (y_off, x_off)
