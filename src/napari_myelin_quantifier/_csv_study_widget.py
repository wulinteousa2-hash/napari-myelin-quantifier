from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import napari
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .csv_quantification import (
    load_measurement_csv,
    load_processed_excel,
    validate_required_columns,
)
from .csv_study_analysis import (
    PCA_DEFAULT_FEATURES,
    add_sample_metadata,
    available_groups,
    build_combined_object_table,
    build_sample_summary_table,
    export_study_workbook,
    prepare_pca_input,
    process_csv_file_with_metadata,
    sample_id_for_path,
    sample_id_for_index,
    summarize_sample_features,
)

_PANEL_MIN_WIDTH = 520
_PANEL_MAX_WIDTH = 760


def _format_float(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.4g}"
    except Exception:
        return str(value)


def _table_item(text: Any, editable: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem(str(text))
    flags = item.flags()
    if editable:
        item.setFlags(flags | Qt.ItemIsEditable)
    else:
        item.setFlags(flags & ~Qt.ItemIsEditable)
    return item


def _checkbox_item(checked: bool = True) -> QTableWidgetItem:
    item = QTableWidgetItem("")
    item.setFlags((item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
    item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
    return item


class CSVStudyAnalysisWidget(QWidget):
    """Study-level CSV quantification, group plotting, PCA, and k-means."""

    IMPORT_COLUMNS = [
        "Include",
        "Sample ID",
        "File name",
        "Animal ID",
        "Image ID",
        "Blind Group",
        "Final Group",
        "Rows",
        "Required columns",
        "Status",
        "Output path",
    ]

    SUMMARY_COLUMNS = [
        "Sample ID",
        "Blind Group",
        "Final Group",
        "Valid object count",
        "Invalid object count",
        "Mean G-ratio",
        "Median G-ratio",
        "Mean myelin thickness",
        "Mean axon diameter",
        "Thin percent",
        "Medium percent",
        "Thick percent",
        "Status",
    ]

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.records: list[dict[str, Any]] = []
        self.combined_objects_df = pd.DataFrame()
        self.sample_summary_df = pd.DataFrame()
        self.pca_cluster_df = pd.DataFrame()
        self.selected_pca_features: list[str] = []
        self.last_kmeans_n: int | None = None
        self.output_dir: str = ""

        self._build_ui()
        self._refresh_all_tables()

    def _build_ui(self) -> None:
        self.setMinimumWidth(_PANEL_MIN_WIDTH)
        self.setMaximumWidth(_PANEL_MAX_WIDTH)
        layout = QVBoxLayout(self)

        note = QLabel(
            "Study-level myelin morphometry: import CSVs, assign blind/final "
            "groups, process samples, generate group plots, and run sample-level "
            "PCA/k-means. Object-level plots are descriptive; sample-level "
            "summaries should be used for biological group comparison."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._build_import_tab()
        self._build_qc_tab()
        self._build_plot_tab()
        self._build_compare_tab()
        self._build_pca_tab()
        self._build_export_tab()

        self.status_label = QLabel("Import one or more CSV files to start a study.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Tab 1
    # ------------------------------------------------------------------
    def _build_import_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        buttons = QWidget()
        b = QHBoxLayout(buttons)
        self.btn_import = QPushButton("Import CSV / Excel")
        self.btn_process_selected = QPushButton("Process Selected")
        self.btn_process_all = QPushButton("Process All")
        self.btn_output_dir = QPushButton("Set Output Folder")
        self.btn_clear = QPushButton("Clear Study")
        self.prefix_numeric_sample_ids = QCheckBox("Prefix numeric IDs with S")
        b.addWidget(self.btn_import)
        b.addWidget(self.btn_process_selected)
        b.addWidget(self.btn_process_all)
        b.addWidget(self.btn_output_dir)
        b.addWidget(self.btn_clear)
        layout.addWidget(buttons)
        layout.addWidget(self.prefix_numeric_sample_ids)

        self.output_dir_label = QLabel("Output folder: next to each source file")
        self.output_dir_label.setWordWrap(True)
        layout.addWidget(self.output_dir_label)

        self.import_table = QTableWidget(0, len(self.IMPORT_COLUMNS))
        self.import_table.setHorizontalHeaderLabels(self.IMPORT_COLUMNS)
        self.import_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.import_table.setSelectionMode(QTableWidget.SingleSelection)
        layout.addWidget(self.import_table)

        self.btn_import.clicked.connect(self._import_csvs)
        self.btn_process_selected.clicked.connect(self._process_selected)
        self.btn_process_all.clicked.connect(self._process_all)
        self.btn_output_dir.clicked.connect(self._select_output_dir)
        self.btn_clear.clicked.connect(self._clear_study)
        self.import_table.cellChanged.connect(self._sync_record_from_import_table)
        self.import_table.itemChanged.connect(self._sync_record_check_state)

        self.tabs.addTab(tab, "Import & Blind Setup")

    def _import_csvs(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import raw CSV or calculated Excel files",
            "",
            "Measurement files (*.csv *.xlsx *.xls);;CSV files (*.csv);;Excel workbooks (*.xlsx *.xls)",
        )
        if not paths:
            return

        seen = {str(record["path"]) for record in self.records}
        added = 0
        for path in paths:
            if path in seen:
                continue
            source = Path(path)
            sample_id = sample_id_for_path(
                source,
                prefix_numeric=self.prefix_numeric_sample_ids.isChecked(),
            )
            if not sample_id:
                sample_id = sample_id_for_index(len(self.records))
            record = {
                "include": True,
                "sample_id": sample_id,
                "path": str(source),
                "source_file_name": source.name,
                "animal_id": "",
                "image_id": "",
                "blind_group": "",
                "final_group": "",
                "rows": "",
                "has_required": False,
                "status": "Imported",
                "file_type": source.suffix.lower(),
                "output_dir": self.output_dir,
                "output_path": "",
                "processed_data": None,
                "summary": None,
                "sample_summary": None,
            }
            try:
                if source.suffix.lower() in {".xlsx", ".xls"}:
                    processed, summary = load_processed_excel(source)
                    record["rows"] = len(processed)
                    record["has_required"] = True
                    record["status"] = "Loaded calculated Excel"
                    record["output_path"] = str(source)
                    record["processed_data"] = add_sample_metadata(processed, record)
                    record["summary"] = summary
                    record["sample_summary"] = summarize_sample_features(
                        record["processed_data"], record
                    )
                else:
                    df = load_measurement_csv(source)
                    record["rows"] = len(df)
                    record["has_required"] = validate_required_columns(df)
                    if not record["has_required"]:
                        record["status"] = "Missing columns"
            except Exception as exc:
                record["status"] = f"Import error: {exc}"
            self.records.append(record)
            added += 1

        self._rebuild_combined_tables()
        self._refresh_all_tables()
        if self.records:
            self.import_table.selectRow(len(self.records) - 1)
        self.status_label.setText(f"Imported {added} measurement file(s).")

    def _select_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select output folder for calculated Excel files",
            self.output_dir or "",
        )
        if not directory:
            return
        self.output_dir = directory
        self.output_dir_label.setText(f"Output folder: {directory}")
        for record in self.records:
            record["output_dir"] = directory
        self.status_label.setText(f"Output folder set: {directory}")

    def _clear_study(self) -> None:
        self.records.clear()
        self.combined_objects_df = pd.DataFrame()
        self.sample_summary_df = pd.DataFrame()
        self.pca_cluster_df = pd.DataFrame()
        self.selected_pca_features = []
        self.last_kmeans_n = None
        self._refresh_all_tables()
        self._clear_plot_canvas(self.plot_figure, self.plot_canvas)
        self._clear_plot_canvas(self.compare_figure, self.compare_canvas)
        self._clear_plot_canvas(self.pca_figure, self.pca_canvas)
        self.status_label.setText("Study cleared.")

    def _sync_record_from_import_table(self, row: int, column: int) -> None:
        if row < 0 or row >= len(self.records):
            return
        record = self.records[row]
        item = self.import_table.item(row, column)
        if item is None:
            return
        text = item.text().strip()
        mapping = {
            1: "sample_id",
            3: "animal_id",
            4: "image_id",
            5: "blind_group",
            6: "final_group",
        }
        if column in mapping:
            record[mapping[column]] = text
            if record.get("processed_data") is not None:
                # Rebuild metadata-sensitive in-memory tables after user edits groups/IDs.
                # Do not re-read or overwrite the source Excel output just because labels changed.
                record["processed_data"] = add_sample_metadata(record["processed_data"], record)
                record["sample_summary"] = summarize_sample_features(record["processed_data"], record)
                self._rebuild_combined_tables()
                self._refresh_qc_table()

    def _sync_record_check_state(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        row = item.row()
        if row < 0 or row >= len(self.records):
            return
        self.records[row]["include"] = item.checkState() == Qt.Checked
        self._rebuild_combined_tables()
        self._refresh_qc_table()

    def _refresh_import_table(self) -> None:
        self.import_table.blockSignals(True)
        self.import_table.setRowCount(len(self.records))
        for row, record in enumerate(self.records):
            self.import_table.setItem(row, 0, _checkbox_item(record.get("include", True)))
            self.import_table.setItem(row, 1, _table_item(record.get("sample_id", ""), editable=True))
            self.import_table.setItem(row, 2, _table_item(record.get("source_file_name", Path(record["path"]).name)))
            self.import_table.setItem(row, 3, _table_item(record.get("animal_id", ""), editable=True))
            self.import_table.setItem(row, 4, _table_item(record.get("image_id", ""), editable=True))
            self.import_table.setItem(row, 5, _table_item(record.get("blind_group", ""), editable=True))
            self.import_table.setItem(row, 6, _table_item(record.get("final_group", ""), editable=True))
            self.import_table.setItem(row, 7, _table_item(record.get("rows", "")))
            self.import_table.setItem(row, 8, _table_item("Yes" if record.get("has_required") else "No"))
            self.import_table.setItem(row, 9, _table_item(record.get("status", "")))
            self.import_table.setItem(row, 10, _table_item(record.get("output_path", "")))
        self.import_table.resizeColumnsToContents()
        self.import_table.blockSignals(False)

    def _selected_record_index(self) -> int | None:
        row = self.import_table.currentRow()
        if row < 0 or row >= len(self.records):
            return None
        return row

    def _process_selected(self) -> None:
        row = self._selected_record_index()
        if row is None:
            self.status_label.setText("Select a row before processing.")
            return
        self._process_record_indices([row])

    def _process_all(self) -> None:
        if not self.records:
            self.status_label.setText("Import CSV files before processing.")
            return
        self._process_record_indices(list(range(len(self.records))))

    def _process_record_indices(self, indices: list[int]) -> None:
        done_count = 0
        for index in indices:
            record = self.records[index]
            if not record.get("include", True):
                record["status"] = "Skipped: not included"
                continue
            if not record.get("has_required"):
                record["status"] = "Skipped: missing columns"
                continue
            try:
                result = process_csv_file_with_metadata(record)
            except Exception as exc:
                record["status"] = f"Error: {exc}"
                continue
            record["status"] = "Done"
            record["output_path"] = str(result["output_path"])
            record["processed_data"] = result["processed_data"]
            record["summary"] = result["summary"]
            record["sample_summary"] = result["sample_summary"]
            done_count += 1

        self._rebuild_combined_tables()
        self._refresh_all_tables()
        self.status_label.setText(f"Processed {done_count} sample(s).")

    def _rebuild_combined_tables(self) -> None:
        self.combined_objects_df = build_combined_object_table(self.records)
        self.sample_summary_df = build_sample_summary_table(self.records)
        self._refresh_group_selectors()
        self._refresh_compare_selectors()

    # ------------------------------------------------------------------
    # Tab 2
    # ------------------------------------------------------------------
    def _build_qc_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        warning = QLabel(
            "Object-level measurements are descriptive. Use sample-level summaries "
            "for biological group comparison, PCA, and k-means."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)

        self.qc_text = QTextEdit()
        self.qc_text.setReadOnly(True)
        self.qc_text.setMaximumHeight(120)
        layout.addWidget(self.qc_text)

        self.qc_table = QTableWidget(0, len(self.SUMMARY_COLUMNS))
        self.qc_table.setHorizontalHeaderLabels(self.SUMMARY_COLUMNS)
        self.qc_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.qc_table)

        self.tabs.addTab(tab, "QC & Summary")

    def _refresh_qc_table(self) -> None:
        df = self.sample_summary_df
        self.qc_table.setRowCount(len(df))
        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, column in enumerate(self.SUMMARY_COLUMNS):
                value = row.get(column, "")
                if isinstance(value, float):
                    value = _format_float(value)
                self.qc_table.setItem(row_idx, col_idx, _table_item(value))
        self.qc_table.resizeColumnsToContents()

        imported = len(self.records)
        processed = sum(1 for r in self.records if r.get("processed_data") is not None)
        total_valid = int(df["Valid object count"].sum()) if "Valid object count" in df else 0
        total_invalid = int(df["Invalid object count"].sum()) if "Invalid object count" in df else 0
        blind = ", ".join(available_groups(df, "Blind Group")) or "none"
        final = ", ".join(available_groups(df, "Final Group")) or "none"
        self.qc_text.setPlainText(
            f"Imported files: {imported}\n"
            f"Processed files: {processed}\n"
            f"Total valid objects: {total_valid}\n"
            f"Total invalid objects: {total_invalid}\n"
            f"Blind groups: {blind}\n"
            f"Final groups: {final}"
        )

    # ------------------------------------------------------------------
    # Tab 3
    # ------------------------------------------------------------------
    def _build_plot_tab(self) -> None:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QGroupBox("Group plot controls")
        form = QFormLayout(controls)
        self.plot_scope_combo = QComboBox()
        self.plot_scope_combo.addItems(
            [
                "Selected sample",
                "Selected blind group",
                "Selected final group",
                "All samples / groups",
            ]
        )
        self.group_variable_combo = QComboBox()
        self.group_variable_combo.addItems(["Blind Group", "Final Group", "Sample ID"])
        self.group_value_combo = QComboBox()
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(
            [
                "G-ratio histogram by group",
                "Axon diameter histogram by group",
                "Myelin thickness histogram by group",
                "G-ratio vs Axon Diameter by group",
                "Myelin Thickness vs Axon Diameter by group",
                "G-ratio by axon size class and group",
                "Axon size composition by group",
                "Mean G-ratio by sample",
                "Mean myelin thickness by sample",
                "Mean axon diameter by sample",
            ]
        )
        self.btn_generate_group_plot = QPushButton("Generate Plot")
        form.addRow("Plot scope:", self.plot_scope_combo)
        form.addRow("Group variable:", self.group_variable_combo)
        form.addRow("Group value:", self.group_value_combo)
        form.addRow("Plot type:", self.plot_type_combo)
        form.addRow(self.btn_generate_group_plot)
        layout.addWidget(controls)

        self.plot_figure = Figure(figsize=(5.5, 3.8), tight_layout=True)
        self.plot_canvas = FigureCanvasQTAgg(self.plot_figure)
        layout.addWidget(self.plot_canvas)

        self.btn_generate_group_plot.clicked.connect(self._generate_group_plot)
        self.group_variable_combo.currentTextChanged.connect(self._refresh_group_selectors)
        self.plot_scope_combo.currentTextChanged.connect(self._refresh_group_selectors)

        self.tabs.addTab(tab, "Group Plots")

    def _refresh_group_selectors(self) -> None:
        if not hasattr(self, "group_value_combo"):
            return
        group_var = self.group_variable_combo.currentText()
        scope = self.plot_scope_combo.currentText()
        self.group_value_combo.clear()

        if scope == "Selected sample":
            values = available_groups(self.sample_summary_df, "Sample ID")
        elif scope == "Selected blind group":
            values = available_groups(self.sample_summary_df, "Blind Group")
        elif scope == "Selected final group":
            values = available_groups(self.sample_summary_df, "Final Group")
        else:
            values = available_groups(self.sample_summary_df, group_var)
        self.group_value_combo.addItems(values)

    def _plot_object_subset(self) -> pd.DataFrame:
        df = self.combined_objects_df.copy()
        if df.empty:
            return df
        df = df[df["Valid Measurement"]].copy()
        scope = self.plot_scope_combo.currentText()
        value = self.group_value_combo.currentText().strip()
        if scope == "Selected sample" and value:
            df = df[df["Sample ID"].astype(str) == value]
        elif scope == "Selected blind group" and value:
            df = df[df["Blind Group"].astype(str) == value]
        elif scope == "Selected final group" and value:
            df = df[df["Final Group"].astype(str) == value]
        return df

    def _sample_subset(self) -> pd.DataFrame:
        df = self.sample_summary_df.copy()
        if df.empty:
            return df
        scope = self.plot_scope_combo.currentText()
        value = self.group_value_combo.currentText().strip()
        if scope == "Selected sample" and value:
            df = df[df["Sample ID"].astype(str) == value]
        elif scope == "Selected blind group" and value:
            df = df[df["Blind Group"].astype(str) == value]
        elif scope == "Selected final group" and value:
            df = df[df["Final Group"].astype(str) == value]
        return df

    def _generate_group_plot(self) -> None:
        plot_name = self.plot_type_combo.currentText()
        group_col = self.group_variable_combo.currentText()
        self.plot_figure.clear()
        ax = self.plot_figure.add_subplot(111)

        if plot_name in (
            "Mean G-ratio by sample",
            "Mean myelin thickness by sample",
            "Mean axon diameter by sample",
            "Axon size composition by group",
        ):
            df = self._sample_subset()
        else:
            df = self._plot_object_subset()

        if df.empty:
            self.status_label.setText("No processed valid data available for this plot.")
            self.plot_canvas.draw()
            return

        if plot_name == "G-ratio histogram by group":
            self._plot_group_hist(ax, df, group_col, "G-ratio", "G-ratio")
        elif plot_name == "Axon diameter histogram by group":
            self._plot_group_hist(ax, df, group_col, "Axon Diameter µm", "Axon Diameter µm")
        elif plot_name == "Myelin thickness histogram by group":
            self._plot_group_hist(ax, df, group_col, "thickness (Myelin)", "Myelin thickness")
        elif plot_name == "G-ratio vs Axon Diameter by group":
            self._plot_group_scatter(ax, df, group_col, "Axon Diameter µm", "G-ratio")
        elif plot_name == "Myelin Thickness vs Axon Diameter by group":
            self._plot_group_scatter(ax, df, group_col, "Axon Diameter µm", "thickness (Myelin)")
        elif plot_name == "G-ratio by axon size class and group":
            self._plot_gratio_by_size_and_group(ax, df, group_col)
        elif plot_name == "Axon size composition by group":
            self._plot_size_composition(ax, df, group_col)
        elif plot_name == "Mean G-ratio by sample":
            self._plot_sample_metric(ax, df, "Mean G-ratio")
        elif plot_name == "Mean myelin thickness by sample":
            self._plot_sample_metric(ax, df, "Mean myelin thickness")
        elif plot_name == "Mean axon diameter by sample":
            self._plot_sample_metric(ax, df, "Mean axon diameter")

        ax.set_title(plot_name)
        with suppress(Exception):
            ax.legend(fontsize=8)
        self.plot_canvas.draw()
        self.status_label.setText(f"Generated plot: {plot_name}")

    def _group_values(self, df: pd.DataFrame, group_col: str) -> list[str]:
        if group_col not in df.columns:
            return ["All"]
        values = [str(v).strip() or "Unassigned" for v in df[group_col].dropna().unique()]
        return sorted(set(values)) or ["All"]

    def _plot_group_hist(self, ax, df: pd.DataFrame, group_col: str, value_col: str, xlabel: str) -> None:
        for group in self._group_values(df, group_col):
            subset = df if group == "All" else df[(df[group_col].astype(str).str.strip().replace("", "Unassigned")) == group]
            values = pd.to_numeric(subset[value_col], errors="coerce").dropna()
            if values.empty:
                continue
            ax.hist(values, bins=24, alpha=0.45, label=group, histtype="stepfilled")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Object count")

    def _plot_group_scatter(self, ax, df: pd.DataFrame, group_col: str, x_col: str, y_col: str) -> None:
        for group in self._group_values(df, group_col):
            subset = df if group == "All" else df[(df[group_col].astype(str).str.strip().replace("", "Unassigned")) == group]
            x = pd.to_numeric(subset[x_col], errors="coerce")
            y = pd.to_numeric(subset[y_col], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.any():
                ax.scatter(x[valid], y[valid], s=14, alpha=0.65, label=group)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    def _plot_gratio_by_size_and_group(self, ax, df: pd.DataFrame, group_col: str) -> None:
        classes = ["Thin", "Medium", "Thick"]
        data = []
        labels = []
        for group in self._group_values(df, group_col):
            group_df = df if group == "All" else df[(df[group_col].astype(str).str.strip().replace("", "Unassigned")) == group]
            for size_class in classes:
                values = pd.to_numeric(
                    group_df.loc[group_df["Axon Size Class"] == size_class, "G-ratio"],
                    errors="coerce",
                ).dropna()
                if not values.empty:
                    data.append(values.to_numpy())
                    labels.append(f"{group}\n{size_class}")
        if not data:
            self.status_label.setText("No G-ratio values available by axon size class.")
            return
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_ylabel("G-ratio")
        ax.tick_params(axis="x", labelrotation=45)

    def _plot_size_composition(self, ax, df: pd.DataFrame, group_col: str) -> None:
        percent_cols = ["Thin percent", "Medium percent", "Thick percent"]
        group_col = group_col if group_col in df.columns else "Sample ID"
        grouped = df.groupby(group_col, dropna=False)[percent_cols].mean(numeric_only=True)
        grouped.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean percentage of valid objects")
        ax.set_xlabel(group_col)
        ax.tick_params(axis="x", labelrotation=45)

    def _plot_sample_metric(self, ax, df: pd.DataFrame, metric_col: str) -> None:
        labels = df["Sample ID"].astype(str).tolist()
        values = pd.to_numeric(df[metric_col], errors="coerce")
        ax.bar(labels, values)
        ax.set_xlabel("Sample ID")
        ax.set_ylabel(metric_col)
        ax.tick_params(axis="x", labelrotation=45)

    # ------------------------------------------------------------------
    # Tab 4
    # ------------------------------------------------------------------
    def _build_compare_tab(self) -> None:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QGroupBox("Traditional comparison")
        grid = QGridLayout(controls)
        self.compare_level_combo = QComboBox()
        self.compare_level_combo.addItems(["Sample ID", "Blind Group", "Final Group"])
        self.compare_value_a = QComboBox()
        self.compare_value_b = QComboBox()
        self.compare_value_c = QComboBox()
        self.compare_metric_combo = QComboBox()
        self.compare_metric_combo.addItems(
            ["G-ratio", "Axon Diameter µm", "thickness (Myelin)"]
        )
        self.compare_plot_combo = QComboBox()
        self.compare_plot_combo.addItems(
            [
                "Overlay histogram",
                "Boxplot",
                "G-ratio vs Axon Diameter overlay",
                "G-ratio vs Myelin Thickness overlay",
            ]
        )
        self.btn_generate_compare_plot = QPushButton("Compare")

        grid.addWidget(QLabel("Compare by:"), 0, 0)
        grid.addWidget(self.compare_level_combo, 0, 1)
        grid.addWidget(QLabel("A:"), 1, 0)
        grid.addWidget(self.compare_value_a, 1, 1)
        grid.addWidget(QLabel("B:"), 2, 0)
        grid.addWidget(self.compare_value_b, 2, 1)
        grid.addWidget(QLabel("C:"), 3, 0)
        grid.addWidget(self.compare_value_c, 3, 1)
        grid.addWidget(QLabel("Metric:"), 4, 0)
        grid.addWidget(self.compare_metric_combo, 4, 1)
        grid.addWidget(QLabel("Plot:"), 5, 0)
        grid.addWidget(self.compare_plot_combo, 5, 1)
        grid.addWidget(self.btn_generate_compare_plot, 6, 0, 1, 2)
        layout.addWidget(controls)

        self.compare_figure = Figure(figsize=(5.5, 4.2), tight_layout=True)
        self.compare_canvas = FigureCanvasQTAgg(self.compare_figure)
        self.compare_canvas.setMinimumHeight(360)
        self.compare_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.compare_canvas, stretch=1)

        self.compare_level_combo.currentTextChanged.connect(
            self._refresh_compare_selectors
        )
        self.btn_generate_compare_plot.clicked.connect(self._generate_compare_plot)

        self.tabs.addTab(tab, "Compare")

    def _refresh_compare_selectors(self) -> None:
        if not hasattr(self, "compare_value_a"):
            return
        level = self.compare_level_combo.currentText()
        values = available_groups(self.combined_objects_df, level)
        values = [""] + values
        combos = (
            self.compare_value_a,
            self.compare_value_b,
            self.compare_value_c,
        )
        previous = [combo.currentText() for combo in combos]
        for combo, old_value in zip(combos, previous, strict=False):
            combo.clear()
            combo.addItems(values)
            if old_value in values:
                combo.setCurrentText(old_value)
        if len(values) > 1:
            self.compare_value_a.setCurrentIndex(1)
        if len(values) > 2:
            self.compare_value_b.setCurrentIndex(2)
        if len(values) > 3:
            self.compare_value_c.setCurrentIndex(3)

    def _comparison_values(self) -> list[str]:
        values = [
            self.compare_value_a.currentText().strip(),
            self.compare_value_b.currentText().strip(),
            self.compare_value_c.currentText().strip(),
        ]
        out: list[str] = []
        for value in values:
            if value and value not in out:
                out.append(value)
        return out

    def _generate_compare_plot(self) -> None:
        if self.combined_objects_df.empty:
            self.status_label.setText("Process or load samples before comparing.")
            return

        level = self.compare_level_combo.currentText()
        values = self._comparison_values()
        if len(values) < 2:
            self.status_label.setText("Select at least two subjects or groups to compare.")
            return

        df = self.combined_objects_df.copy()
        df = df[df["Valid Measurement"]].copy()
        df[level] = df[level].astype(str).str.strip()
        plot_name = self.compare_plot_combo.currentText()
        metric = self.compare_metric_combo.currentText()

        self.compare_figure.clear()
        ax = self.compare_figure.add_subplot(111)

        if plot_name == "Overlay histogram":
            for value in values:
                subset = df[df[level] == value]
                series = pd.to_numeric(subset[metric], errors="coerce").dropna()
                if not series.empty:
                    ax.hist(
                        series,
                        bins=24,
                        alpha=0.38,
                        label=f"{value} (n={len(series)})",
                        histtype="stepfilled",
                    )
            ax.set_xlabel(metric)
            ax.set_ylabel("Object count")
        elif plot_name == "Boxplot":
            data = []
            labels = []
            for value in values:
                subset = df[df[level] == value]
                series = pd.to_numeric(subset[metric], errors="coerce").dropna()
                if not series.empty:
                    data.append(series.to_numpy())
                    labels.append(f"{value}\nn={len(series)}")
            if data:
                ax.boxplot(data, labels=labels, showmeans=True)
            ax.set_ylabel(metric)
        elif plot_name == "G-ratio vs Axon Diameter overlay":
            for value in values:
                subset = df[df[level] == value]
                ax.scatter(
                    subset["Axon Diameter µm"],
                    subset["G-ratio"],
                    s=14,
                    alpha=0.62,
                    label=value,
                )
            ax.set_xlabel("Axon Diameter µm")
            ax.set_ylabel("G-ratio")
        else:
            for value in values:
                subset = df[df[level] == value]
                ax.scatter(
                    subset["thickness (Myelin)"],
                    subset["G-ratio"],
                    s=14,
                    alpha=0.62,
                    label=value,
                )
            ax.set_xlabel("thickness (Myelin)")
            ax.set_ylabel("G-ratio")

        ax.set_title(f"{plot_name}: {level}")
        with suppress(Exception):
            ax.legend(fontsize=8)
        self.compare_canvas.draw()
        self.status_label.setText(
            f"Compared {len(values)} {level.lower()} value(s): {', '.join(values)}"
        )

    # ------------------------------------------------------------------
    # Tab 5
    # ------------------------------------------------------------------
    def _build_pca_tab(self) -> None:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        setup_panel = QWidget()
        setup_layout = QVBoxLayout(setup_panel)
        setup_layout.setContentsMargins(0, 0, 0, 0)

        controls = QGroupBox("Analysis setup")
        form = QFormLayout(controls)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.analysis_level_combo = QComboBox()
        self.analysis_level_combo.addItems(["Sample-level only"])
        self.pca_group_color_combo = QComboBox()
        self.pca_group_color_combo.addItems(["Blind Group", "Final Group", "Sample ID"])
        self.kmeans_spin = QSpinBox()
        self.kmeans_spin.setRange(2, 10)
        self.kmeans_spin.setValue(2)
        form.addRow("Analysis level:", self.analysis_level_combo)
        form.addRow("Group color:", self.pca_group_color_combo)
        form.addRow("K-means clusters:", self.kmeans_spin)
        setup_layout.addWidget(controls)

        feature_box = QGroupBox("PCA / k-means features")
        feature_layout = QGridLayout(feature_box)
        feature_layout.setContentsMargins(10, 8, 10, 8)
        feature_layout.setHorizontalSpacing(16)
        feature_layout.setVerticalSpacing(3)
        self.feature_checks: dict[str, QCheckBox] = {}
        for index, feature in enumerate(PCA_DEFAULT_FEATURES):
            cb = QCheckBox(feature)
            cb.setChecked(True)
            cb.setToolTip(feature)
            self.feature_checks[feature] = cb
            feature_layout.addWidget(cb, index // 2, index % 2)
        setup_layout.addWidget(feature_box)

        buttons = QWidget()
        b = QHBoxLayout(buttons)
        b.setContentsMargins(0, 0, 0, 0)
        self.btn_run_pca = QPushButton("Run PCA")
        self.btn_run_kmeans = QPushButton("Run k-means")
        self.btn_run_both = QPushButton("Run PCA + k-means")
        b.addWidget(self.btn_run_pca)
        b.addWidget(self.btn_run_kmeans)
        b.addWidget(self.btn_run_both)
        setup_layout.addWidget(buttons)

        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 0, 0, 0)

        self.pca_text = QTextEdit()
        self.pca_text.setReadOnly(True)
        self.pca_text.setMaximumHeight(90)
        self.pca_text.setPlaceholderText("Run PCA or k-means to see analysis details.")
        results_layout.addWidget(self.pca_text)

        self.pca_figure = Figure(figsize=(5.5, 3.8), tight_layout=True)
        self.pca_canvas = FigureCanvasQTAgg(self.pca_figure)
        self.pca_canvas.setMinimumHeight(260)
        self.pca_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_layout.addWidget(self.pca_canvas, stretch=3)

        self.cluster_table = QTableWidget(0, 6)
        self.cluster_table.setHorizontalHeaderLabels(["Sample ID", "Blind Group", "Final Group", "Cluster", "PC1", "PC2"])
        self.cluster_table.setMinimumHeight(170)
        self.cluster_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cluster_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.cluster_table, stretch=2)

        splitter.addWidget(setup_panel)
        splitter.addWidget(results_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 520])

        self.btn_run_pca.clicked.connect(lambda: self._run_pca(run_kmeans=False))
        self.btn_run_kmeans.clicked.connect(lambda: self._run_pca(run_kmeans=True, plot_kmeans_only=True))
        self.btn_run_both.clicked.connect(lambda: self._run_pca(run_kmeans=True))

        self.tabs.addTab(tab, "PCA / Clustering")

    def _selected_features(self) -> list[str]:
        return [name for name, cb in self.feature_checks.items() if cb.isChecked()]

    def _run_pca(self, run_kmeans: bool = False, plot_kmeans_only: bool = False) -> None:
        if self.sample_summary_df.empty:
            self.status_label.setText("Process samples before running PCA/k-means.")
            return
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except Exception:
            self.status_label.setText("scikit-learn is required for PCA/k-means. Please install scikit-learn.")
            return

        features = self._selected_features()
        metadata, matrix = prepare_pca_input(self.sample_summary_df, features)
        if len(matrix) < 2:
            self.status_label.setText("At least 2 samples with complete selected features are required.")
            return
        if matrix.shape[1] < 2:
            self.status_label.setText("Select at least 2 numeric features for PCA.")
            return

        scaled = StandardScaler().fit_transform(matrix.to_numpy(dtype=float))
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(scaled)

        result = metadata.copy()
        result["PC1"] = pcs[:, 0]
        result["PC2"] = pcs[:, 1]
        result["Cluster"] = ""

        if run_kmeans:
            n_clusters = int(self.kmeans_spin.value())
            if len(result) < n_clusters:
                self.status_label.setText("K-means cluster number cannot exceed the number of valid samples.")
                return
            labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(scaled)
            result["Cluster"] = labels.astype(int)
            self.last_kmeans_n = n_clusters

        self.pca_cluster_df = result
        self.selected_pca_features = features
        self._refresh_cluster_table()
        self._plot_pca_result(result)

        ev = pca.explained_variance_ratio_ * 100.0
        self.pca_text.setPlainText(
            f"PCA input samples: {len(result)}\n"
            f"Features used: {', '.join(features)}\n"
            f"Explained variance: PC1={ev[0]:.2f}%, PC2={ev[1]:.2f}%\n"
            f"K-means: {'n=' + str(self.last_kmeans_n) if run_kmeans else 'not run'}"
        )
        if plot_kmeans_only:
            self.status_label.setText("Ran k-means using sample-level scaled features and refreshed PCA projection.")
        else:
            self.status_label.setText("Ran sample-level PCA" + (" + k-means." if run_kmeans else "."))

    def _plot_pca_result(self, result: pd.DataFrame) -> None:
        self.pca_figure.clear()
        ax = self.pca_figure.add_subplot(111)
        color_col = self.pca_group_color_combo.currentText()
        groups = available_groups(result, color_col)
        if not groups:
            groups = ["All"]
        for group in groups:
            subset = result if group == "All" else result[result[color_col].astype(str) == group]
            ax.scatter(subset["PC1"], subset["PC2"], s=42, label=group)
            for _, row in subset.iterrows():
                ax.text(row["PC1"], row["PC2"], str(row.get("Sample ID", "")), fontsize=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Sample-level PCA")
        with suppress(Exception):
            ax.legend(fontsize=8)
        self.pca_canvas.draw()

    def _refresh_cluster_table(self) -> None:
        df = self.pca_cluster_df
        cols = ["Sample ID", "Blind Group", "Final Group", "Cluster", "PC1", "PC2"]
        self.cluster_table.setRowCount(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            for j, col in enumerate(cols):
                value = row.get(col, "")
                if col in ("PC1", "PC2"):
                    value = _format_float(value)
                self.cluster_table.setItem(i, j, _table_item(value))
        self.cluster_table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Tab 5
    # ------------------------------------------------------------------
    def _build_export_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.btn_export_workbook = QPushButton("Export Study Workbook")
        self.btn_export_objects = QPushButton("Export Combined Object-Level CSV")
        self.btn_export_summary = QPushButton("Export Sample Summary CSV")
        self.btn_export_pca = QPushButton("Export PCA/Cluster Results CSV")
        layout.addWidget(self.btn_export_workbook)
        layout.addWidget(self.btn_export_objects)
        layout.addWidget(self.btn_export_summary)
        layout.addWidget(self.btn_export_pca)
        layout.addStretch(1)

        self.btn_export_workbook.clicked.connect(self._export_workbook)
        self.btn_export_objects.clicked.connect(lambda: self._export_csv(self.combined_objects_df, "combined_object_level_data.csv"))
        self.btn_export_summary.clicked.connect(lambda: self._export_csv(self.sample_summary_df, "sample_level_summary.csv"))
        self.btn_export_pca.clicked.connect(lambda: self._export_csv(self.pca_cluster_df, "pca_cluster_results.csv"))

        self.tabs.addTab(tab, "Export")

    def _export_csv(self, df: pd.DataFrame, default_name: str) -> None:
        if df is None or df.empty:
            self.status_label.setText("No data available to export.")
            return
        default_path = str(Path(self.output_dir) / default_name) if self.output_dir else default_name
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", default_path, "CSV files (*.csv)")
        if not path:
            return
        df.to_csv(path, index=False)
        self.status_label.setText(f"Exported CSV: {path}")

    def _export_workbook(self) -> None:
        if self.combined_objects_df.empty and self.sample_summary_df.empty:
            self.status_label.setText("No processed study data available to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save study workbook",
            str(Path(self.output_dir) / "myelin_study_analysis.xlsx")
            if self.output_dir
            else "myelin_study_analysis.xlsx",
            "Excel workbooks (*.xlsx)",
        )
        if not path:
            return
        metadata = {
            "number of imported files": len(self.records),
            "number of processed files": sum(1 for r in self.records if r.get("processed_data") is not None),
            "selected PCA features": ", ".join(self.selected_pca_features),
            "k-means cluster number": self.last_kmeans_n or "",
        }
        export_study_workbook(path, self.combined_objects_df, self.sample_summary_df, self.pca_cluster_df, metadata)
        self.status_label.setText(f"Exported study workbook: {path}")

    # ------------------------------------------------------------------
    # General refresh helpers
    # ------------------------------------------------------------------
    def _refresh_all_tables(self) -> None:
        self._refresh_import_table()
        self._refresh_qc_table()
        self._refresh_cluster_table()
        self._refresh_group_selectors()

    def _clear_plot_canvas(self, figure, canvas) -> None:
        figure.clear()
        canvas.draw()


def csv_study_analysis_widget(viewer=None, **kwargs) -> QWidget:
    if viewer is None:
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No active napari viewer found.")
    return CSVStudyAnalysisWidget(viewer)
