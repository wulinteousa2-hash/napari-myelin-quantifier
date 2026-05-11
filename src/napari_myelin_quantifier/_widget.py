# src/napari_myelin_quantifier/_widget.py
from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ._axon_quant import QuantParams, quantify_myelin_rings_from_mask
from .csv_quantification import (
    load_measurement_csv,
    process_csv_file,
    save_combined_summary,
    validate_required_columns,
)

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


def threshold_autogenerate_widget(
    image: np.ndarray, threshold: float
) -> np.ndarray:
    """Legacy example helper retained for the starter-template tests."""
    return np.asarray(image) > float(threshold)


@magic_factory(call_button="Threshold")
def threshold_magic_widget(
    image: napari.layers.Image, threshold: float = 0.5
):
    """Legacy example magicgui widget retained for test compatibility."""
    return np.asarray(image.data) > float(threshold)


class ImageThreshold(QWidget):
    """Small legacy threshold widget retained for test compatibility."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self._image_layer_combo = SimpleNamespace(value=None)
        self._threshold_slider = SimpleNamespace(value=0.5)

    def _threshold_im(self) -> None:
        layer = self._image_layer_combo.value
        if layer is None:
            raise ValueError("Select an image layer.")
        thresholded = np.asarray(layer.data) > float(self._threshold_slider.value)
        self.viewer.add_labels(thresholded.astype(np.uint8), name="thresholded")


class ExampleQWidget(QWidget):
    """Small legacy example widget retained for test compatibility."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

    def _on_click(self) -> None:
        print(f"napari has {len(self.viewer.layers)} layers")


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


def csv_quantification_widget(viewer=None, **kwargs) -> QWidget:
    if viewer is None:
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No active napari viewer found.")

    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
    except ImportError:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

    container = QWidget()
    layout = QVBoxLayout(container)
    container.setMinimumWidth(_PANEL_MIN_WIDTH)
    container.setMaximumWidth(_PANEL_MAX_WIDTH)

    group = QGroupBox("CSV Quantification")
    group_layout = QVBoxLayout(group)

    button_row = QWidget()
    button_layout = QHBoxLayout(button_row)
    btn_import = QPushButton("Import CSV")
    btn_process = QPushButton("Process CSV")
    button_layout.addWidget(btn_import)
    button_layout.addWidget(btn_process)
    group_layout.addWidget(button_row)

    table = QTableWidget(0, 8)
    table.setHorizontalHeaderLabels(
        [
            "File name",
            "Rows",
            "Required columns",
            "Status",
            "Output path",
            "Mean G-ratio",
            "Mean thickness",
            "Mean axon diameter",
        ]
    )
    table.setSelectionBehavior(QTableWidget.SelectRows)
    table.setSelectionMode(QTableWidget.SingleSelection)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.horizontalHeader().setStretchLastSection(True)
    group_layout.addWidget(table)

    summary_label = QLabel("Select a processed CSV file to view a summary.")
    summary_label.setWordWrap(True)
    group_layout.addWidget(summary_label)

    plot_row = QWidget()
    plot_layout = QHBoxLayout(plot_row)
    plot_combo = QComboBox()
    btn_plot = QPushButton("Generate Plot")
    plot_combo.addItems(
        [
            "G-ratio histogram",
            "Axon diameter histogram",
            "Myelin thickness histogram",
            "G-ratio vs Axon Diameter",
            "G-ratio vs Myelin Thickness",
            "G-ratio by axon size class",
        ]
    )
    plot_layout.addWidget(plot_combo)
    plot_layout.addWidget(btn_plot)
    group_layout.addWidget(plot_row)

    figure = Figure(figsize=(4.6, 3.1), tight_layout=True)
    canvas = FigureCanvasQTAgg(figure)
    group_layout.addWidget(canvas)

    status_label = QLabel("Import one or more measurement CSV files.")
    status_label.setWordWrap(True)
    group_layout.addWidget(status_label)
    layout.addWidget(group)

    records: list[dict] = []

    def _format_float(value) -> str:
        if pd.isna(value):
            return ""
        return f"{float(value):.4g}"

    def _summary_value(summary: pd.DataFrame, column: str):
        if summary is None or summary.empty or column not in summary.columns:
            return np.nan
        return summary.iloc[0][column]

    def _set_item(row: int, column: int, value) -> None:
        table.setItem(row, column, QTableWidgetItem(str(value)))

    def _refresh_table() -> None:
        table.setRowCount(len(records))
        for row, record in enumerate(records):
            summary = record.get("summary")
            _set_item(row, 0, Path(record["path"]).name)
            _set_item(row, 1, record.get("rows", ""))
            _set_item(row, 2, "Yes" if record.get("has_required") else "No")
            _set_item(row, 3, record.get("status", "Imported"))
            _set_item(row, 4, record.get("output_path", ""))
            _set_item(
                row,
                5,
                _format_float(_summary_value(summary, "Mean G-ratio")),
            )
            _set_item(
                row,
                6,
                _format_float(
                    _summary_value(summary, "Mean myelin thickness")
                ),
            )
            _set_item(
                row,
                7,
                _format_float(_summary_value(summary, "Mean axon diameter")),
            )
        table.resizeColumnsToContents()

    def _selected_record() -> dict | None:
        row = table.currentRow()
        if row < 0 or row >= len(records):
            return None
        return records[row]

    def _update_summary() -> None:
        record = _selected_record()
        if record is None:
            summary_label.setText("Select a processed CSV file to view a summary.")
            return
        summary = record.get("summary")
        if summary is None or summary.empty:
            summary_label.setText(
                f"{Path(record['path']).name}\n"
                f"Status: {record.get('status', 'Imported')}"
            )
            return

        row = summary.iloc[0]
        summary_label.setText(
            f"File name: {Path(record['path']).name}\n"
            f"Total objects: {int(row['Total object count'])}\n"
            f"Valid objects: {int(row['Valid object count'])}\n"
            f"Mean G-ratio: {_format_float(row['Mean G-ratio'])}\n"
            f"Mean myelin thickness: "
            f"{_format_float(row['Mean myelin thickness'])}\n"
            f"Mean axon diameter: {_format_float(row['Mean axon diameter'])}"
        )

    def _import_csv() -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            container,
            "Import measurement CSV files",
            "",
            "CSV files (*.csv)",
        )
        if not paths:
            return

        seen = {str(record["path"]) for record in records}
        added = 0
        for path in paths:
            if path in seen:
                continue
            record = {
                "path": path,
                "rows": "",
                "has_required": False,
                "status": "Imported",
                "output_path": "",
                "processed_data": None,
                "summary": None,
            }
            try:
                df = load_measurement_csv(path)
                record["rows"] = len(df)
                record["has_required"] = validate_required_columns(df)
                if not record["has_required"]:
                    record["status"] = "Missing columns"
            except Exception as e:
                record["status"] = f"Import error: {e}"
            records.append(record)
            added += 1

        _refresh_table()
        if records:
            table.selectRow(len(records) - 1)
            _update_summary()
        status_label.setText(f"Imported {added} CSV file(s).")

    def _process_csv() -> None:
        if not records:
            status_label.setText("Import CSV files before processing.")
            return

        processed_summaries: list[pd.DataFrame] = []
        done_count = 0
        for record in records:
            if not record.get("has_required"):
                record["status"] = "Skipped: missing columns"
                continue
            try:
                result = process_csv_file(record["path"])
            except Exception as e:
                record["status"] = f"Error: {e}"
                continue

            record["status"] = "Done"
            record["output_path"] = str(result["output_path"])
            record["processed_data"] = result["processed_data"]
            record["summary"] = result["summary"]
            processed_summaries.append(result["summary"])
            done_count += 1

        if done_count > 1:
            first_dir = Path(records[0]["path"]).parent
            combined_path = first_dir / "combined_myelin_quantification_summary.xlsx"
            save_combined_summary(processed_summaries, combined_path)
            status_label.setText(
                f"Processed {done_count} CSV file(s). Combined summary: "
                f"{combined_path}"
            )
        else:
            status_label.setText(f"Processed {done_count} CSV file(s).")

        _refresh_table()
        _update_summary()

    def _plot_selected() -> None:
        record = _selected_record()
        if record is None or record.get("processed_data") is None:
            status_label.setText("Select a processed CSV file before plotting.")
            return

        df = record["processed_data"]
        valid_df = df[df["Valid Measurement"]]
        if valid_df.empty:
            status_label.setText("Selected file has no valid measurements to plot.")
            return

        figure.clear()
        ax = figure.add_subplot(111)
        plot_name = plot_combo.currentText()

        if plot_name == "G-ratio histogram":
            ax.hist(valid_df["G-ratio"].dropna(), bins=20, color="#3f7f93")
            ax.set_xlabel("G-ratio")
            ax.set_ylabel("Object count")
        elif plot_name == "Axon diameter histogram":
            ax.hist(
                valid_df["Axon Diameter µm"].dropna(),
                bins=20,
                color="#5f8f4e",
            )
            ax.set_xlabel("Axon Diameter µm")
            ax.set_ylabel("Object count")
        elif plot_name == "Myelin thickness histogram":
            ax.hist(
                valid_df["thickness (Myelin)"].dropna(),
                bins=20,
                color="#9b6b3f",
            )
            ax.set_xlabel("thickness (Myelin)")
            ax.set_ylabel("Object count")
        elif plot_name == "G-ratio vs Axon Diameter":
            ax.scatter(
                valid_df["Axon Diameter µm"],
                valid_df["G-ratio"],
                s=16,
                alpha=0.75,
                color="#3f7f93",
            )
            ax.set_xlabel("Axon Diameter µm")
            ax.set_ylabel("G-ratio")
        elif plot_name == "G-ratio vs Myelin Thickness":
            ax.scatter(
                valid_df["thickness (Myelin)"],
                valid_df["G-ratio"],
                s=16,
                alpha=0.75,
                color="#6d6098",
            )
            ax.set_xlabel("thickness (Myelin)")
            ax.set_ylabel("G-ratio")
        else:
            classes = ["Thin", "Medium", "Thick"]
            values = [
                valid_df.loc[
                    valid_df["Axon Size Class"] == size_class, "G-ratio"
                ].dropna()
                for size_class in classes
            ]
            ax.boxplot(values, labels=classes, showmeans=True)
            ax.set_xlabel("Axon Size Class")
            ax.set_ylabel("G-ratio")

        ax.set_title(plot_name)
        canvas.draw()
        status_label.setText(f"Generated plot: {plot_name}")

    btn_import.clicked.connect(_import_csv)
    btn_process.clicked.connect(_process_csv)
    btn_plot.clicked.connect(_plot_selected)
    table.itemSelectionChanged.connect(_update_summary)

    return container
