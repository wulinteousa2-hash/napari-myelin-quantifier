from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .csv_quantification import (
    calculate_myelin_features,
    load_measurement_csv,
    load_processed_excel,
    processed_output_path,
    save_processed_excel,
    summarize_myelin_features,
)

METADATA_COLUMNS = (
    "Sample ID",
    "Source file name",
    "Animal ID",
    "Image ID",
    "Blind Group",
    "Final Group",
)

PCA_DEFAULT_FEATURES = (
    "Mean G-ratio",
    "Median G-ratio",
    "Standard deviation G-ratio",
    "Mean myelin thickness",
    "Median myelin thickness",
    "Mean axon diameter",
    "Median axon diameter",
    "Thin percent",
    "Medium percent",
    "Thick percent",
    "Thin mean G-ratio",
    "Medium mean G-ratio",
    "Thick mean G-ratio",
    "Thin mean axon diameter",
    "Medium mean axon diameter",
    "Thick mean axon diameter",
)


def sample_id_for_path(
    path: str | Path, prefix_numeric: bool = False
) -> str:
    """Return a default sample ID based on the source filename."""
    stem = Path(path).stem.strip()
    if stem.lower().startswith("calculated_"):
        stem = stem[len("calculated_") :]
    if prefix_numeric and stem.isdigit():
        return f"S{stem}"
    return stem


def sample_id_for_index(index: int) -> str:
    """Return a fallback sample ID such as S001."""
    return f"S{int(index) + 1:03d}"


def coerce_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Return normalized metadata values from a study record."""
    source_path = Path(record.get("path") or record.get("source_path") or "")
    return {
        "Sample ID": str(record.get("sample_id") or "").strip(),
        "Source file name": str(record.get("source_file_name") or source_path.name),
        "Animal ID": str(record.get("animal_id") or "").strip(),
        "Image ID": str(record.get("image_id") or "").strip(),
        "Blind Group": str(record.get("blind_group") or "").strip(),
        "Final Group": str(record.get("final_group") or "").strip(),
    }


def add_sample_metadata(processed_df: pd.DataFrame, record: dict[str, Any]) -> pd.DataFrame:
    """Add study/sample metadata columns to an object-level dataframe."""
    result = processed_df.copy()
    metadata = coerce_record_metadata(record)
    for column, value in metadata.items():
        result[column] = value
    return result


def _safe_mean(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def _safe_percent(count: int, total: int) -> float:
    if total <= 0:
        return np.nan
    return float(count) / float(total) * 100.0


def summarize_sample_features(processed_df: pd.DataFrame, record: dict[str, Any]) -> pd.DataFrame:
    """Create one sample-level summary row from one processed object table."""
    if "Valid Measurement" not in processed_df.columns:
        raise ValueError("processed_df must contain 'Valid Measurement'.")

    metadata = coerce_record_metadata(record)
    valid_df = processed_df[processed_df["Valid Measurement"]].copy()
    valid_count = int(len(valid_df))

    class_counts = {
        size_class: int((valid_df["Axon Size Class"] == size_class).sum())
        for size_class in ("Thin", "Medium", "Thick")
    }

    row: dict[str, Any] = {
        "Include": bool(record.get("include", True)),
        **metadata,
        "Total object count": int(len(processed_df)),
        "Valid object count": valid_count,
        "Invalid object count": int(len(processed_df) - valid_count),
        "Mean G-ratio": _safe_mean(valid_df, "G-ratio"),
        "Median G-ratio": float(pd.to_numeric(valid_df["G-ratio"], errors="coerce").median()) if valid_count else np.nan,
        "Standard deviation G-ratio": float(pd.to_numeric(valid_df["G-ratio"], errors="coerce").std()) if valid_count else np.nan,
        "Mean myelin thickness": _safe_mean(valid_df, "thickness (Myelin)"),
        "Median myelin thickness": float(pd.to_numeric(valid_df["thickness (Myelin)"], errors="coerce").median()) if valid_count else np.nan,
        "Mean axon diameter": _safe_mean(valid_df, "Axon Diameter µm"),
        "Median axon diameter": float(pd.to_numeric(valid_df["Axon Diameter µm"], errors="coerce").median()) if valid_count else np.nan,
        "Thin count": class_counts["Thin"],
        "Medium count": class_counts["Medium"],
        "Thick count": class_counts["Thick"],
        "Thin percent": _safe_percent(class_counts["Thin"], valid_count),
        "Medium percent": _safe_percent(class_counts["Medium"], valid_count),
        "Thick percent": _safe_percent(class_counts["Thick"], valid_count),
        "Thin mean G-ratio": _safe_mean(valid_df[valid_df["Axon Size Class"] == "Thin"], "G-ratio"),
        "Medium mean G-ratio": _safe_mean(valid_df[valid_df["Axon Size Class"] == "Medium"], "G-ratio"),
        "Thick mean G-ratio": _safe_mean(valid_df[valid_df["Axon Size Class"] == "Thick"], "G-ratio"),
        "Thin mean axon diameter": _safe_mean(valid_df[valid_df["Axon Size Class"] == "Thin"], "Axon Diameter µm"),
        "Medium mean axon diameter": _safe_mean(valid_df[valid_df["Axon Size Class"] == "Medium"], "Axon Diameter µm"),
        "Thick mean axon diameter": _safe_mean(valid_df[valid_df["Axon Size Class"] == "Thick"], "Axon Diameter µm"),
        "Status": str(record.get("status") or ""),
    }
    return pd.DataFrame([row])


def process_csv_file_with_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Process or load one study record and return tables with metadata."""
    source = Path(record["path"])
    suffix = source.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        processed, basic_summary = load_processed_excel(source)
        output_path = source
    else:
        raw_df = load_measurement_csv(source)
        processed = calculate_myelin_features(raw_df)
        basic_summary = summarize_myelin_features(processed, source.name)
        output_dir = record.get("output_dir") or None
        output_path = processed_output_path(source, output_dir=output_dir)
        save_processed_excel(processed, basic_summary, output_path)

    processed = add_sample_metadata(processed, record)
    sample_summary = summarize_sample_features(processed, record)

    return {
        "source_path": source,
        "output_path": output_path,
        "processed_data": processed,
        "summary": basic_summary,
        "sample_summary": sample_summary,
    }


def _iter_included_processed(records: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for record in records:
        if not record.get("include", True):
            continue
        if record.get("processed_data") is None:
            continue
        yield record


def build_combined_object_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Concatenate included processed object-level tables."""
    frames = [record["processed_data"] for record in _iter_included_processed(records)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_sample_summary_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build one sample-level summary table from included processed records."""
    frames: list[pd.DataFrame] = []
    for record in _iter_included_processed(records):
        sample_summary = record.get("sample_summary")
        if sample_summary is None or sample_summary.empty:
            sample_summary = summarize_sample_features(record["processed_data"], record)
        frames.append(sample_summary)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def available_groups(df: pd.DataFrame, column: str) -> list[str]:
    """Return sorted non-empty group labels."""
    if df.empty or column not in df.columns:
        return []
    values = [str(v).strip() for v in df[column].dropna().unique() if str(v).strip()]
    return sorted(values)


def prepare_pca_input(
    sample_summary: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return metadata and numeric feature matrix after dropping missing values."""
    if sample_summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    features = [f for f in selected_features if f in sample_summary.columns]
    if not features:
        return pd.DataFrame(), pd.DataFrame()

    metadata_cols = [
        col
        for col in ("Sample ID", "Blind Group", "Final Group", "Source file name")
        if col in sample_summary.columns
    ]
    feature_matrix = sample_summary[features].apply(pd.to_numeric, errors="coerce")
    valid_mask = feature_matrix.notna().all(axis=1)
    return sample_summary.loc[valid_mask, metadata_cols].reset_index(drop=True), feature_matrix.loc[valid_mask].reset_index(drop=True)


def export_study_workbook(
    output_path: str | Path,
    object_level_df: pd.DataFrame,
    sample_summary_df: pd.DataFrame,
    pca_cluster_df: pd.DataFrame | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Export the full study analysis workbook."""
    output_path = Path(output_path)
    metadata = dict(metadata or {})
    metadata.setdefault("plugin name", "napari-myelin-quantifier")
    metadata.setdefault("analysis date/time", datetime.now().isoformat(timespec="seconds"))

    metadata_df = pd.DataFrame(
        [{"Key": str(key), "Value": str(value)} for key, value in metadata.items()]
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        object_level_df.to_excel(writer, sheet_name="Object_Level_Data", index=False)
        sample_summary_df.to_excel(writer, sheet_name="Sample_Level_Summary", index=False)
        if pca_cluster_df is not None and not pca_cluster_df.empty:
            pca_cluster_df.to_excel(writer, sheet_name="PCA_Cluster_Results", index=False)
        metadata_df.to_excel(writer, sheet_name="Analysis_Metadata", index=False)
    return output_path
