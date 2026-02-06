from __future__ import annotations

from contextlib import suppress

import napari
import numpy as np
import pandas as pd
from magicgui import magic_factory
from qtpy.QtWidgets import QFileDialog

from ._axon_quant import QuantParams, quantify_myelinated_axons


def _as_uint8_rgb(data: np.ndarray) -> np.ndarray:
    """Convert napari image data into uint8 RGB (Y,X,3)."""
    arr = np.asarray(data)

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            a = a - np.nanmin(a)
            denom = (np.nanmax(a) - np.nanmin(a)) + 1e-9
            a = a / denom
            a = (a * 255.0).clip(0, 255).astype(np.uint8)
            arr = a
        return arr[..., :3]

    raise ValueError(
        f"Expected RGB/RGBA image data. Got shape={arr.shape}, dtype={arr.dtype}"
    )


@magic_factory(
    call_button="Run quantification",
    v_thresh={
        "label": "V-threshold (myelin brightness)",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    min_axon_area={
        "label": "Min axon area (px)",
        "min": 0,
        "max": 20000,
        "step": 10,
    },
    min_solidity={
        "label": "Min solidity",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    close_radius={"label": "Closing radius", "min": 0, "max": 20, "step": 1},
    open_radius={"label": "Opening radius", "min": 0, "max": 20, "step": 1},
    edge_margin_px={
        "label": "Edge margin exclusion (px)",
        "min": 0,
        "max": 200,
        "step": 1,
    },
    px_size_um={
        "label": "Pixel size (µm/px)",
        "min": 0.0,
        "max": 10.0,
        "step": 0.001,
    },
    text_size={"label": "Label font size", "min": 1, "max": 30, "step": 1},
)
def myelin_quantifier_widget(
    viewer: napari.Viewer,
    image_layer: napari.layers.Image,  # <-- IMPORTANT: resolvable string
    v_thresh: float = 0.15,
    min_axon_area: int = 150,
    min_solidity: float = 0.60,
    close_radius: int = 3,
    open_radius: int = 1,
    edge_margin_px: int = 8,
    px_size_um: float = 0.02,
    remove_border_objects: bool = True,
    auto_crop_screenshot: bool = True,
    export_csv: bool = True,
    text_size: int = 5,
    text_scale_with_zoom: bool = True,
) -> None:
    """Runs myelinated axon quantification on an RGB image layer."""
    if image_layer is None:
        raise ValueError("Select an RGB image layer.")

    img_rgb = _as_uint8_rgb(image_layer.data)

    params = QuantParams(
        px_size_um=px_size_um,
        v_thresh=v_thresh,
        min_axon_area=min_axon_area,
        min_solidity=min_solidity,
        close_radius=close_radius,
        open_radius=open_radius,
        remove_border_objects=remove_border_objects,
        edge_margin_px=edge_margin_px,
        auto_crop=auto_crop_screenshot,
    )

    labels_full, overlay_full, df, _ = quantify_myelinated_axons(
        img_rgb, params
    )

    base = image_layer.name

    viewer.add_labels(labels_full, name=f"{base} | axon_ids")
    viewer.add_image(overlay_full, name=f"{base} | axon_overlay", opacity=0.8)

    if len(df) > 0:
        points = df[["cy_img", "cx_img"]].to_numpy(dtype=float)  # (y,x)
        features = pd.DataFrame(
            {"axon_id": df["axon_id"].astype(str).to_numpy()}
        )

        pts = viewer.add_points(
            points,
            name=f"{base} | axon_id_text",
            size=0,
            features=features,
        )

        pts.text = {
            "string": "{axon_id}",
            "size": float(text_size),
            "color": "yellow",
            "anchor": "center",
        }

        # napari-version-safe scaling toggle
        with suppress(AttributeError):
            pts.text.scaling = bool(text_scale_with_zoom)

    if export_csv:
        default_name = f"{base}_axon_measurements.csv"
        save_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save axon measurements CSV",
            default_name,
            "CSV files (*.csv)",
        )
        if save_path:
            df.to_csv(save_path, index=False)

    viewer.status = f"Myelin Quantifier: detected {len(df)} axons (filtered)."
