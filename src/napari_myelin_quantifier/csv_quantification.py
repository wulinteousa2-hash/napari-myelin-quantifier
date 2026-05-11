from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AREA_COLUMN = "2D Area (µm²)"
FILLED_AREA_COLUMN = "2D Filled Area (µm²)"
REQUIRED_COLUMNS = (AREA_COLUMN, FILLED_AREA_COLUMN)
MASK_AREA_COLUMN = "ring_area_um2"
MASK_FILLED_AREA_COLUMN = "filled_area_um2"
AREA_COLUMN_ALIASES = {
    AREA_COLUMN: (AREA_COLUMN, MASK_AREA_COLUMN),
    FILLED_AREA_COLUMN: (FILLED_AREA_COLUMN, MASK_FILLED_AREA_COLUMN),
}

CALCULATED_COLUMNS = (
    "MyelinatedArea",
    "AxonArea",
    "Rout µm",
    "Rin µm",
    "thickness (Myelin)",
    "G-ratio",
    "Axon Diameter µm",
    "Valid Measurement",
    "Validity Note",
    "Axon Size Class",
)


def load_measurement_csv(path: str | Path) -> pd.DataFrame:
    """Load a label/object measurement CSV file."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.removeprefix("\ufeff")
    return df


def validate_required_columns(df: pd.DataFrame) -> bool:
    """Return True when all columns required for CSV quantification exist."""
    return all(
        any(alias in df.columns for alias in AREA_COLUMN_ALIASES[column])
        for column in REQUIRED_COLUMNS
    )


def missing_required_columns(df: pd.DataFrame) -> list[str]:
    """Return required input columns not present in the dataframe."""
    missing = []
    for column in REQUIRED_COLUMNS:
        aliases = AREA_COLUMN_ALIASES[column]
        if not any(alias in df.columns for alias in aliases):
            missing.append(" or ".join(aliases))
    return missing


def normalize_measurement_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical measurement columns when compatible aliases are present."""
    result = df.copy()
    for column in REQUIRED_COLUMNS:
        if column in result.columns:
            continue
        for alias in AREA_COLUMN_ALIASES[column]:
            if alias in result.columns:
                result[column] = result[alias]
                break
    return result


def _validity_notes(
    filled_area: pd.Series,
    ring_area: pd.Series,
    axon_area: pd.Series,
    rout: pd.Series,
    rin: pd.Series,
    g_ratio: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    notes: list[str] = []
    valid: list[bool] = []

    for values in zip(
        filled_area,
        ring_area,
        axon_area,
        rout,
        rin,
        g_ratio,
        strict=False,
    ):
        filled, ring, axon, rout_value, rin_value, g_value = values
        row_notes: list[str] = []

        if not np.isfinite(filled) or not np.isfinite(ring):
            row_notes.append("Required area values must be numeric and finite")
        if not filled > ring:
            row_notes.append(
                f"{FILLED_AREA_COLUMN} must be greater than {AREA_COLUMN}"
            )
        if not axon > 0:
            row_notes.append("AxonArea must be greater than 0")
        if not rout_value > 0:
            row_notes.append("Rout µm must be greater than 0")
        if not rin_value >= 0:
            row_notes.append("Rin µm must be greater than or equal to 0")
        if not np.isfinite(g_value):
            row_notes.append("G-ratio must be finite")
        elif not 0 <= g_value <= 1:
            row_notes.append("G-ratio must be between 0 and 1")

        valid.append(not row_notes)
        notes.append("Valid" if not row_notes else "; ".join(row_notes))

    return pd.Series(valid, index=filled_area.index), pd.Series(
        notes, index=filled_area.index
    )


def calculate_myelin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add myelin morphometric calculations and row validity columns."""
    if not validate_required_columns(df):
        missing = ", ".join(missing_required_columns(df))
        raise ValueError(f"Missing required column(s): {missing}")

    result = normalize_measurement_columns(df)
    ring_area = pd.to_numeric(result[AREA_COLUMN], errors="coerce")
    filled_area = pd.to_numeric(result[FILLED_AREA_COLUMN], errors="coerce")

    result["MyelinatedArea"] = filled_area
    result["AxonArea"] = filled_area - ring_area

    with np.errstate(invalid="ignore", divide="ignore"):
        result["Rout µm"] = np.sqrt(filled_area / np.pi)
        result["Rin µm"] = np.sqrt(result["AxonArea"] / np.pi)
        result["thickness (Myelin)"] = result["Rout µm"] - result["Rin µm"]
        result["G-ratio"] = result["Rin µm"] / result["Rout µm"]
        result["Axon Diameter µm"] = result["Rin µm"] * 2

    valid, notes = _validity_notes(
        filled_area=filled_area,
        ring_area=ring_area,
        axon_area=result["AxonArea"],
        rout=result["Rout µm"],
        rin=result["Rin µm"],
        g_ratio=result["G-ratio"],
    )
    result["Valid Measurement"] = valid
    result["Validity Note"] = notes

    result["Axon Size Class"] = "Invalid"
    valid_rows = result["Valid Measurement"]
    axon_diameter = result["Axon Diameter µm"]
    result.loc[valid_rows & (axon_diameter < 1.0), "Axon Size Class"] = "Thin"
    result.loc[
        valid_rows & (axon_diameter >= 1.0) & (axon_diameter < 3.0),
        "Axon Size Class",
    ] = "Medium"
    result.loc[valid_rows & (axon_diameter >= 3.0), "Axon Size Class"] = "Thick"

    return result


def summarize_myelin_features(
    df: pd.DataFrame, source_file_name: str | None = None
) -> pd.DataFrame:
    """Create a one-row summary dataframe for processed measurements."""
    if "Valid Measurement" not in df.columns:
        raise ValueError("Dataframe must be processed before summarizing.")

    valid_df = df[df["Valid Measurement"]].copy()
    summary: dict[str, Any] = {
        "Source file name": source_file_name or "",
        "Total object count": int(len(df)),
        "Valid object count": int(len(valid_df)),
        "Invalid object count": int(len(df) - len(valid_df)),
        "Mean G-ratio": valid_df["G-ratio"].mean(),
        "Median G-ratio": valid_df["G-ratio"].median(),
        "Standard deviation G-ratio": valid_df["G-ratio"].std(),
        "Mean myelin thickness": valid_df["thickness (Myelin)"].mean(),
        "Median myelin thickness": valid_df["thickness (Myelin)"].median(),
        "Mean axon diameter": valid_df["Axon Diameter µm"].mean(),
        "Median axon diameter": valid_df["Axon Diameter µm"].median(),
        "Min axon diameter": valid_df["Axon Diameter µm"].min(),
        "Max axon diameter": valid_df["Axon Diameter µm"].max(),
        "Min G-ratio": valid_df["G-ratio"].min(),
        "Max G-ratio": valid_df["G-ratio"].max(),
    }
    return pd.DataFrame([summary])


def save_processed_excel(
    df: pd.DataFrame, summary: pd.DataFrame, output_path: str | Path
) -> Path:
    """Write processed data and summary sheets to an Excel workbook."""
    output_path = Path(output_path)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Processed_Data", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
    return output_path


def processed_output_path(path: str | Path) -> Path:
    """Return calculated_<stem>.xlsx next to the source CSV."""
    source = Path(path)
    return source.with_name(f"calculated_{source.stem}.xlsx")


def process_csv_file(path: str | Path) -> dict[str, Any]:
    """Process one measurement CSV and write its Excel output."""
    source = Path(path)
    df = load_measurement_csv(source)
    processed = calculate_myelin_features(df)
    summary = summarize_myelin_features(processed, source.name)
    output_path = save_processed_excel(
        processed, summary, processed_output_path(source)
    )
    return {
        "source_path": source,
        "output_path": output_path,
        "processed_data": processed,
        "summary": summary,
    }


def process_multiple_csv_files(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Process multiple measurement CSV files."""
    return [process_csv_file(path) for path in paths]


def save_combined_summary(
    summaries: list[pd.DataFrame], output_path: str | Path
) -> Path | None:
    """Save one combined summary workbook, returning None when empty."""
    if not summaries:
        return None
    output_path = Path(output_path)
    combined = pd.concat(summaries, ignore_index=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="Summary", index=False)
    return output_path
