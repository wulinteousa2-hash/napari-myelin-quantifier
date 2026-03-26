# src/napari_myelin_quantifier/_widget.py
from __future__ import annotations

from contextlib import suppress

import napari
import numpy as np
import pandas as pd
from magicgui import magic_factory
from skimage.filters import threshold_otsu
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._axon_quant import QuantParams, quantify_myelin_rings_from_mask

_PANEL_MIN_WIDTH = 420
_PANEL_MAX_WIDTH = 520


def _to_gray_2d(data: np.ndarray) -> np.ndarray:
    """Convert supported image data into a single-channel 2D image."""
    arr = np.asarray(data)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0].astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return arr[..., :3].astype(np.float32).mean(axis=-1)
    raise ValueError(
        f"Expected 2D, RGB, or RGBA image data. Got shape={arr.shape}."
    )


def _invert_image(data: np.ndarray) -> np.ndarray:
    """Invert an image while preserving its current data range."""
    arr = np.asarray(data)
    if arr.dtype == bool:
        return ~arr

    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("Image has no finite pixels to invert.")

    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    return (hi + lo) - arr


def _gray_to_binary_mask(data: np.ndarray) -> np.ndarray:
    """Convert a grayscale-like image into a binary mask with Otsu threshold."""
    gray = _to_gray_2d(data)
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        raise ValueError("Image has no finite pixels for thresholding.")
    if np.allclose(finite.min(), finite.max()):
        raise ValueError("Image is uniform; unable to compute a binary mask.")

    threshold = float(threshold_otsu(finite))
    return gray > threshold


def _as_2d_mask(data: np.ndarray) -> np.ndarray:
    """Convert supported napari layer data into a 2D binary mask."""
    arr = np.asarray(data)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a binary 2D mask layer. Got shape={arr.shape}. "
            "Use Quick Mask Prep first for RGB/RGBA or grayscale images."
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
def _myelin_quantifier_controls(
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
        f"Myelin Rings: quantified {len(df)} rings "
        f"(Euler filter: {myelin_euler_keep})"
    )


def myelin_quantifier_widget(viewer=None, **kwargs) -> QWidget:
    if viewer is None:
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No active napari viewer found.")

    container = QWidget()
    layout = QVBoxLayout(container)
    container.setMinimumWidth(_PANEL_MIN_WIDTH)
    container.setMaximumWidth(_PANEL_MAX_WIDTH)

    prep_group = QGroupBox("Quick Mask Prep (Optional)")
    prep_layout = QFormLayout(prep_group)
    prep_note = QLabel(
        "For already-segmented RGB/RGBA or grayscale masks. "
        "Use: 1-Channel -> Invert if needed -> Binary."
    )
    prep_note.setWordWrap(True)
    prep_layout.addRow(prep_note)

    source_combo = QComboBox()
    prep_layout.addRow(QLabel("Source:"), source_combo)

    btn_refresh = QPushButton("Refresh")
    prep_layout.addRow(btn_refresh)

    gray_row = QWidget()
    gray_layout = QHBoxLayout(gray_row)
    btn_gray = QPushButton("1-Channel")
    btn_invert_gray = QPushButton("Invert")
    gray_layout.addWidget(btn_gray)
    gray_layout.addWidget(btn_invert_gray)
    prep_layout.addRow(gray_row)

    btn_binary = QPushButton("Binary (Otsu)")
    prep_layout.addRow(btn_binary)

    prep_status = QLabel(
        "Optional helper only. Use it when the mask is not already binary."
    )
    prep_status.setWordWrap(True)
    prep_layout.addRow(prep_status)
    layout.addWidget(prep_group)

    quant = _myelin_quantifier_controls()
    quant.viewer.value = viewer
    with suppress(Exception):
        quant.viewer.visible = False

    btn_latest = QPushButton("Use Latest Mask")

    def _refresh_sources() -> None:
        current = source_combo.currentText()
        source_combo.clear()
        for lyr in viewer.layers:
            if isinstance(lyr, (napari.layers.Image, napari.layers.Labels)):
                source_combo.addItem(lyr.name)
        if current:
            idx = source_combo.findText(current)
            if idx >= 0:
                source_combo.setCurrentIndex(idx)

    def _get_source_layer():
        name = source_combo.currentText().strip()
        if not name:
            return None
        with suppress(KeyError):
            return viewer.layers[name]
        return None

    def _set_quant_mask(layer) -> None:
        with suppress(Exception):
            quant.mask_layer.reset_choices()
        with suppress(Exception):
            quant.mask_layer.value = layer
            return
        with suppress(Exception):
            quant.mask_layer.value = layer.name

    def _create_labels_layer(data: np.ndarray, suffix: str) -> None:
        layer = _get_source_layer()
        if layer is None:
            raise ValueError("Select an Image or Labels layer to prepare.")

        prepared = viewer.add_labels(
            (np.asarray(data) > 0).astype(np.uint8),
            name=f"{layer.name} | {suffix}",
            scale=layer.scale,
            translate=layer.translate,
        )
        _refresh_sources()
        source_combo.setCurrentText(prepared.name)
        _set_quant_mask(prepared)
        viewer.status = f"Prepared binary mask: {prepared.name}"

    def _create_gray_layer(invert: bool = False) -> None:
        layer = _get_source_layer()
        if layer is None:
            raise ValueError("Select an Image or Labels layer to prepare.")

        gray = _to_gray_2d(layer.data)
        suffix = "gray"
        if invert:
            gray = _invert_image(gray)
            suffix = "inv"

        viewer.add_image(
            gray,
            name=f"{layer.name} | {suffix}",
            scale=layer.scale,
            translate=layer.translate,
        )
        _refresh_sources()
        source_combo.setCurrentText(f"{layer.name} | {suffix}")
        viewer.status = f"Created helper layer: {layer.name} | {suffix}"

    def _create_binary_mask() -> None:
        layer = _get_source_layer()
        if layer is None:
            raise ValueError("Select an Image or Labels layer to prepare.")
        mask = _gray_to_binary_mask(np.asarray(layer.data))
        _create_labels_layer(mask, "mask")

    def _use_latest_prepared() -> None:
        for lyr in reversed(list(viewer.layers)):
            if isinstance(lyr, napari.layers.Labels) and (
                lyr.name.endswith(" | mask")
                or lyr.name.endswith(" | myelin_instances")
            ):
                _refresh_sources()
                source_combo.setCurrentText(lyr.name)
                _set_quant_mask(lyr)
                viewer.status = f"Selected mask for quantification: {lyr.name}"
                return
        viewer.status = "No prepared binary mask found."

    def _run_safely(fn) -> None:
        try:
            fn()
        except Exception as e:
            viewer.status = str(e)
            prep_status.setText(str(e))
        else:
            prep_status.setText(
                "Optional helper only. Use it when the mask is not already binary."
            )

    btn_refresh.clicked.connect(_refresh_sources)
    btn_gray.clicked.connect(lambda: _run_safely(_create_gray_layer))
    btn_invert_gray.clicked.connect(
        lambda: _run_safely(lambda: _create_gray_layer(invert=True))
    )
    btn_binary.clicked.connect(lambda: _run_safely(_create_binary_mask))
    btn_latest.clicked.connect(_use_latest_prepared)

    _refresh_sources()
    with suppress(Exception):
        quant.mask_layer.reset_choices()
    layout.addWidget(btn_latest)
    layout.addWidget(quant.native)
    return container


@magic_factory(call_button="Locate Ring ID")
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
            "Select the quantified rings labels layer ('... | myelin_instances')."
        )

    lab = np.asarray(instances_layer.data)
    rid = int(ring_id)

    mask = lab == rid
    if not mask.any():
        raise ValueError(
            f"Ring ID {rid} was not found in '{instances_layer.name}'."
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
    viewer.status = f"Located Ring ID {rid} at (y={cy:.1f}, x={cx:.1f})"


def myelin_locator_dashboard(viewer=None, **kwargs) -> QWidget:
    if viewer is None:
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No active napari viewer found.")

    container = QWidget()
    layout = QVBoxLayout(container)
    container.setMinimumWidth(_PANEL_MIN_WIDTH)
    container.setMaximumWidth(_PANEL_MAX_WIDTH)
    layout.addWidget(QLabel("Myelin Rings: Locate by ID"))

    step2 = myelin_ring_locator_widget()
    step2.viewer.value = viewer
    with suppress(Exception):
        step2.viewer.visible = False

    btn = QPushButton("Use Latest Quantified Rings Layer")

    def _use_latest_instances() -> None:
        for lyr in reversed(list(viewer.layers)):
            if getattr(lyr, "name", "").endswith(
                " | myelin_instances"
            ) and isinstance(lyr, napari.layers.Labels):
                with suppress(Exception):
                    step2.instances_layer.value = lyr
                viewer.status = f"Selected quantified rings layer: {lyr.name}"
                return
        viewer.status = (
            "No quantified rings layer found. Run 'Myelin Rings: Quantify' first."
        )

    btn.clicked.connect(_use_latest_instances)

    layout.addWidget(btn)
    layout.addWidget(step2.native)
    return container
