"""Data cleaning module for MP255 OPD Waiting Time dataset."""

from pathlib import Path
from typing import Optional
import pandas as pd

from src.mp255.config import load_params


def clean_department(val: str) -> str:
    """Normalize department names into one of the three canonical categories."""
    if not isinstance(val, str):
        return val
    cleaned = val.strip().lower()
    if cleaned.startswith("gen"):
        return "General"
    elif cleaned.startswith("ort"):
        return "Orthopaedics"
    elif cleaned.startswith("pae"):
        return "Paediatrics"
    return val.strip().title()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning rules to train.csv:
    1. Drop duplicate rows.
    2. Filter out stuck sensor period for avg_consult_min (2026-05-03 to 2026-05-17).
    3. Normalize department spelling variants to 3 clean categories.
    """
    cleaned = df.copy()

    # 1. Deduplicate
    cleaned = cleaned.drop_duplicates()

    # 2. Handle stuck sensor on avg_consult_min
    if "date" in cleaned.columns and "avg_consult_min" in cleaned.columns:
        stuck_mask = (
            (cleaned["date"] >= "2026-05-03")
            & (cleaned["date"] <= "2026-05-17")
            & (cleaned["avg_consult_min"] == 14.0)
        )
        cleaned = cleaned[~stuck_mask]

    # 3. Clean department spelling variants
    if "department" in cleaned.columns:
        cleaned["department"] = cleaned["department"].apply(clean_department)

    return cleaned.reset_index(drop=True)


def run_clean_pipeline(
    raw_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Read raw training data, clean it, and save to processed directory."""
    params = load_params()
    raw_csv = raw_path or params["data"]["raw_train_path"]
    out_csv = output_path or params["data"]["cleaned_train_path"]

    df = pd.read_csv(raw_csv)
    cleaned_df = clean_dataset(df)

    out_file = Path(out_csv)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(out_file, index=False)
    print(f"Cleaned dataset saved to {out_file} ({len(cleaned_df)} rows, original {len(df)} rows)")
    return cleaned_df


if __name__ == "__main__":
    run_clean_pipeline()
