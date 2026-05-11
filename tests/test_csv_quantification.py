import numpy as np
import pandas as pd

from napari_myelin_quantifier.csv_quantification import (
    AREA_COLUMN,
    FILLED_AREA_COLUMN,
    MASK_AREA_COLUMN,
    MASK_FILLED_AREA_COLUMN,
    calculate_myelin_features,
    load_measurement_csv,
    process_csv_file,
    save_processed_excel,
    summarize_myelin_features,
    validate_required_columns,
)


def _measurement_df():
    return pd.DataFrame(
        {
            "Time Step": [0, 0, 0],
            "Label Index": [1, 2, 3],
            "Name": ["NA", "NA", "NA"],
            AREA_COLUMN: [3.0, 10.0, 2.0],
            "2D Euler Number": [0, 0, 0],
            FILLED_AREA_COLUMN: [12.0, 8.0, 18.0],
        }
    )


def test_required_columns_are_detected():
    df = _measurement_df()
    assert validate_required_columns(df)
    assert not validate_required_columns(df.drop(columns=[AREA_COLUMN]))


def test_mask_area_columns_are_detected():
    df = pd.DataFrame(
        {
            "ring_id": [1, 2],
            MASK_AREA_COLUMN: [3.0, 4.0],
            MASK_FILLED_AREA_COLUMN: [12.0, 20.0],
        }
    )

    assert validate_required_columns(df)


def test_semicolon_csv_with_bom_loads_required_columns(tmp_path):
    csv_path = tmp_path / "semicolon.csv"
    csv_path.write_text(
        "\ufeffTime Step;Label Index;Name (NA);"
        "2D Area (µm²);2D Euler Number;2D Filled Area (µm²)\n"
        "0;1;;1.0;1.0;4.0\n",
        encoding="utf-8",
    )

    df = load_measurement_csv(csv_path)

    assert validate_required_columns(df)
    assert len(df) == 1


def test_calculated_columns_are_created():
    processed = calculate_myelin_features(_measurement_df())
    expected_columns = {
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
    }
    assert expected_columns.issubset(processed.columns)


def test_axon_area_is_filled_area_minus_area():
    processed = calculate_myelin_features(_measurement_df())
    expected = processed[FILLED_AREA_COLUMN] - processed[AREA_COLUMN]
    np.testing.assert_allclose(processed["AxonArea"], expected)


def test_mask_area_columns_are_used_for_calculations():
    df = pd.DataFrame(
        {
            "ring_id": [1, 2],
            MASK_AREA_COLUMN: [3.0, 4.0],
            MASK_FILLED_AREA_COLUMN: [12.0, 20.0],
        }
    )

    processed = calculate_myelin_features(df)

    assert AREA_COLUMN in processed.columns
    assert FILLED_AREA_COLUMN in processed.columns
    np.testing.assert_allclose(processed[AREA_COLUMN], df[MASK_AREA_COLUMN])
    np.testing.assert_allclose(
        processed[FILLED_AREA_COLUMN], df[MASK_FILLED_AREA_COLUMN]
    )
    np.testing.assert_allclose(
        processed["AxonArea"], df[MASK_FILLED_AREA_COLUMN] - df[MASK_AREA_COLUMN]
    )


def test_g_ratio_is_rin_divided_by_rout():
    processed = calculate_myelin_features(_measurement_df())
    expected = processed["Rin µm"] / processed["Rout µm"]
    np.testing.assert_allclose(processed["G-ratio"], expected)


def test_invalid_rows_are_marked_invalid():
    processed = calculate_myelin_features(_measurement_df())
    invalid = processed.loc[processed["Label Index"] == 2].iloc[0]
    assert not invalid["Valid Measurement"]
    assert "must be greater than" in invalid["Validity Note"]
    assert invalid["Axon Size Class"] == "Invalid"


def test_excel_output_is_created(tmp_path):
    processed = calculate_myelin_features(_measurement_df())
    summary = summarize_myelin_features(processed, "measurements.csv")
    output_path = tmp_path / "calculated_measurements.xlsx"

    save_processed_excel(processed, summary, output_path)

    assert output_path.exists()
    sheets = pd.read_excel(output_path, sheet_name=None)
    assert set(sheets) == {"Processed_Data", "Summary"}


def test_process_csv_file_writes_expected_output_name(tmp_path):
    csv_path = tmp_path / "117.csv"
    _measurement_df().to_csv(csv_path, index=False)

    result = process_csv_file(csv_path)

    assert result["output_path"] == tmp_path / "calculated_117.xlsx"
    assert result["output_path"].exists()
