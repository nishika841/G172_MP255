"""Unit tests for MP255 data cleaning module."""

from src.mp255.clean import clean_dataset, clean_department


def test_clean_department_variants():
    """Verify department spelling normalization."""
    assert clean_department("Gen.") == "General"
    assert clean_department(" GENERAL ") == "General"
    assert clean_department("general") == "General"
    assert clean_department("Pae.") == "Paediatrics"
    assert clean_department(" PAEDIATRICS") == "Paediatrics"
    assert clean_department("Ort.") == "Orthopaedics"
    assert clean_department(" Orthopaedics ") == "Orthopaedics"


def test_clean_dataset_removes_duplicates(sample_raw_data):
    """Verify duplicate rows are removed."""
    cleaned = clean_dataset(sample_raw_data)
    assert len(cleaned) < len(sample_raw_data)
    assert not cleaned.duplicated().any()


def test_clean_dataset_filters_stuck_sensor(sample_raw_data):
    """Verify stuck sensor rows in May 2026 with avg_consult_min == 14.0 are filtered."""
    cleaned = clean_dataset(sample_raw_data)
    stuck = cleaned[
        (cleaned["date"] >= "2026-05-03")
        & (cleaned["date"] <= "2026-05-17")
        & (cleaned["avg_consult_min"] == 14.0)
    ]
    assert len(stuck) == 0


def test_clean_dataset_department_canonical(sample_raw_data):
    """Verify departments in cleaned output match canonical set."""
    cleaned = clean_dataset(sample_raw_data)
    allowed = {"General", "Paediatrics", "Orthopaedics"}
    assert set(cleaned["department"].unique()).issubset(allowed)
