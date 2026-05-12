import pandas as pd

from napari_myelin_quantifier.csv_quantification import (
    AREA_COLUMN,
    FILLED_AREA_COLUMN,
    calculate_myelin_features,
    save_processed_excel,
    summarize_myelin_features,
)
from napari_myelin_quantifier.csv_study_analysis import (
    process_csv_file_with_metadata,
    sample_id_for_path,
)


def _measurement_df():
    return pd.DataFrame(
        {
            "Label Index": [1, 2],
            AREA_COLUMN: [3.0, 4.0],
            FILLED_AREA_COLUMN: [12.0, 20.0],
        }
    )


def test_study_record_can_load_calculated_excel_without_reprocessing(tmp_path):
    processed = calculate_myelin_features(_measurement_df())
    summary = summarize_myelin_features(processed, "raw.csv")
    workbook = tmp_path / "calculated_raw.xlsx"
    save_processed_excel(processed, summary, workbook)

    result = process_csv_file_with_metadata(
        {
            "path": str(workbook),
            "sample_id": "S001",
            "blind_group": "A",
            "final_group": "Control",
            "include": True,
        }
    )

    assert result["output_path"] == workbook
    assert result["processed_data"]["Sample ID"].eq("S001").all()
    assert result["sample_summary"].iloc[0]["Blind Group"] == "A"


def test_study_record_raw_csv_uses_selected_output_dir(tmp_path):
    csv_path = tmp_path / "raw.csv"
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    _measurement_df().to_csv(csv_path, index=False)

    result = process_csv_file_with_metadata(
        {
            "path": str(csv_path),
            "sample_id": "S001",
            "output_dir": str(out_dir),
            "include": True,
        }
    )

    assert result["output_path"] == out_dir / "calculated_raw.xlsx"
    assert result["output_path"].exists()


def test_sample_id_defaults_to_filename_stem():
    assert sample_id_for_path("119.csv") == "119"
    assert sample_id_for_path("S119.csv") == "S119"
    assert sample_id_for_path("mouseA_03.csv") == "mouseA_03"


def test_sample_id_strips_calculated_prefix():
    assert sample_id_for_path("calculated_119.xlsx") == "119"
    assert sample_id_for_path("calculated_S119.xlsx") == "S119"


def test_sample_id_can_prefix_numeric_filename():
    assert sample_id_for_path("119.csv", prefix_numeric=True) == "S119"
    assert sample_id_for_path("S119.csv", prefix_numeric=True) == "S119"
