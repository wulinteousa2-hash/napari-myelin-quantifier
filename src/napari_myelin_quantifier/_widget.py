# src/napari_myelin_quantifier/_widget.py
from __future__ import annotations

from contextlib import suppress

import napari
import numpy as np
import pandas as pd
from magicgui import magic_factory
from qtpy.QtWidgets import (
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._axon_quant import QuantParams, quantify_myelin_rings_from_mask


def _as_2d_mask(data: np.ndarray) -> np.ndarray:
    """Ensure the input is a 2D array (napari Image/Labels data)."""
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask layer (Image or Labels). Got shape={arr.shape}. "
            "If you loaded an RGB screenshot, convert it to a binary mask first."
        )
    return arr


def _parse_euler_keep(text: str) -> list[int] | None:
    """
    Parse a comma-separated Euler list: e.g. "-1,0,1".
    Returns None for "any/all/*" meaning no filtering.
    """
    t = (text or "").strip().lower()
    if t in ("", "any", "all", "*"):
        return None
    parts = [p.strip() for p in t.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError as e:
            raise ValueError(
                f"Invalid Euler list item: {p!r}. Use e.g. '-1,0,1' or 'any'."
            ) from e
    return out


def _get_or_create_labels_layer(
    viewer: napari.Viewer, name: str, data: np.ndarray
) -> napari.layers.Labels:
    """Create or overwrite a Labels layer with a stable name."""
    if name in viewer.layers:
        layer = viewer.layers[name]
        if not isinstance(layer, napari.layers.Labels):
            raise TypeError(
                f"Layer name '{name}' exists but is not a Labels layer."
            )
        layer.data = data
        layer.visible = True
        return layer
    return viewer.add_labels(data, name=name)


def _get_or_create_points_layer(
    viewer: napari.Viewer,
    name: str,
    points: np.ndarray,
    features: pd.DataFrame,
    text_size: float,
    text_scale_with_zoom: bool,
) -> napari.layers.Points:
    """Create or overwrite a Points layer with ring_id text."""
    if name in viewer.layers:
        layer = viewer.layers[name]
        if not isinstance(layer, napari.layers.Points):
            raise TypeError(
                f"Layer name '{name}' exists but is not a Points layer."
            )
        layer.data = points
        layer.features = features
        layer.visible = True
    else:
        layer = viewer.add_points(points, name=name, size=0, features=features)

    layer.text = {
        "string": "{ring_id}",
        "size": float(text_size),
        "color": "yellow",
        "anchor": "center",
    }
    with suppress(AttributeError):
        layer.text.scaling = bool(text_scale_with_zoom)
    return layer


@magic_factory(
    call_button="Run myelin ring quantification",
    min_ring_area={
        "label": "Min ring area (px)",
        "min": 0,
        "max": 200000,
        "step": 10,
    },
    close_radius={"label": "Closing radius", "min": 0, "max": 20, "step": 1},
    open_radius={"label": "Opening radius", "min": 0, "max": 20, "step": 1},
    lumen_min_area={
        "label": "Min lumen area (px)",
        "min": 0,
        "max": 200000,
        "step": 10,
    },
    seed_dilation={
        "label": "Seed dilation (px)",
        "min": 0,
        "max": 50,
        "step": 1,
    },
    peak_min_distance={
        "label": "Peak min distance",
        "min": 1,
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
    mask_layer: napari.layers.Layer,
    min_ring_area: int = 80,
    close_radius: int = 1,
    open_radius: int = 0,
    lumen_min_area: int = 50,
    seed_dilation: int = 2,
    use_peak_seeds: bool = False,
    peak_min_distance: int = 10,
    exclude_border_objects: bool = False,
    myelin_euler_keep: str = "any",
    px_size_um: float = 0.0,
    add_text_labels: bool = True,
    text_size: int = 5,
    text_scale_with_zoom: bool = True,
    export_csv: bool = True,
) -> None:
    if mask_layer is None:
        raise ValueError("Select a 2D mask layer (Image or Labels).")

    mask = _as_2d_mask(mask_layer.data)
    base = mask_layer.name

    params = QuantParams(
        px_size_um=px_size_um,
        min_ring_area=min_ring_area,
        close_radius=close_radius,
        open_radius=open_radius,
        lumen_min_area=lumen_min_area,
        seed_dilation=seed_dilation,
        use_peak_seeds=use_peak_seeds,
        peak_min_distance=peak_min_distance,
        exclude_border_objects=exclude_border_objects,
    )

    e_keep = _parse_euler_keep(myelin_euler_keep)
    inst_keep, df = quantify_myelin_rings_from_mask(
        mask, params, euler_keep=e_keep
    )

    # --- stable layer names; overwrite on rerun ---
    lbl_name = f"{base} | myelin_instances"
    _get_or_create_labels_layer(viewer, lbl_name, inst_keep)

    pts_name = f"{base} | myelin_ring_IDs"
    if add_text_labels and len(df) > 0:
        pts = df[["centroid_y", "centroid_x"]].to_numpy(dtype=float)
        features = pd.DataFrame(
            {"ring_id": df["ring_id"].astype(str).to_numpy()}
        )
        _get_or_create_points_layer(
            viewer,
            pts_name,
            pts,
            features,
            text_size=float(text_size),
            text_scale_with_zoom=bool(text_scale_with_zoom),
        )
    else:
        # If user disables labels, just hide the points layer if it exists
        with suppress(KeyError):
            viewer.layers[pts_name].visible = False

    if export_csv and len(df) > 0:
        default_name = f"{base}_myelin_rings.csv"
        save_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save myelin ring quantification CSV",
            default_name,
            "CSV files (*.csv)",
        )
        if save_path:
            df.to_csv(save_path, index=False)

    viewer.status = (
        f"Myelin Quantifier: rings={len(df)} (Euler keep: {myelin_euler_keep})"
    )


@magic_factory(call_button="Locate ring_id")
def myelin_ring_locator_widget(
    viewer: napari.Viewer,
    instances_layer: napari.layers.Labels,
    ring_id: int = 1,
    zoom_multiplier: float = 1.5,
    min_zoom: float = 0.1,
    max_zoom: float = 500.0,
) -> None:
    if instances_layer is None:
        raise ValueError(
            "Select the *Labels* layer: '... | myelin_instances'."
        )

    lab = np.asarray(instances_layer.data)
    rid = int(ring_id)

    mask = lab == rid
    if not mask.any():
        raise ValueError(
            f"ring_id={rid} not found in '{instances_layer.name}'."
        )

    yy, xx = np.nonzero(mask)
    cy = float(yy.mean())
    cx = float(xx.mean())

    viewer.camera.center = (cy, cx)

    # clamp zoom so repeated clicks don't explode
    if zoom_multiplier and zoom_multiplier > 0:
        new_zoom = float(viewer.camera.zoom) * float(zoom_multiplier)
        new_zoom = max(float(min_zoom), min(float(max_zoom), new_zoom))
        viewer.camera.zoom = new_zoom

    viewer.layers.selection.active = instances_layer
    viewer.status = f"Located ring_id={rid} at (y={cy:.1f}, x={cx:.1f})"


def myelin_locator_dashboard(viewer=None, **kwargs) -> QWidget:
    if viewer is None:
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No active napari viewer found.")

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.addWidget(QLabel("Myelin Locator Dashboard (Step 2)"))

    step2 = myelin_ring_locator_widget()
    step2.viewer.value = viewer
    with suppress(Exception):
        step2.viewer.visible = False

    btn = QPushButton("Use latest myelin_instances layer")

    def _use_latest_instances() -> None:
        for lyr in reversed(list(viewer.layers)):
            if getattr(lyr, "name", "").endswith(
                " | myelin_instances"
            ) and isinstance(lyr, napari.layers.Labels):
                with suppress(Exception):
                    step2.instances_layer.value = lyr
                viewer.status = f"Selected: {lyr.name}"
                return
        viewer.status = (
            "No '* | myelin_instances' Labels layer found. Run Step 1 first."
        )

    btn.clicked.connect(_use_latest_instances)

    layout.addWidget(btn)
    layout.addWidget(step2.native)
    return container
