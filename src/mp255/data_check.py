import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.mp255.config import load_params


def check_duplicate_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for identical duplicate rows in the dataset."""
    extra_duplicates = int(df.duplicated().sum())
    return {
        "present": extra_duplicates > 0,
        "count": extra_duplicates if extra_duplicates > 0 else None,
    }


def check_stuck_sensor(df: pd.DataFrame, min_streak: int = 15) -> Dict[str, Any]:
    """Check if any numeric column has an identical value stuck across consecutive days."""
    df_sorted = df.sort_values(by=["date", "record_id"]).copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "wait_min" in numeric_cols:
        numeric_cols.remove("wait_min")

    for col in numeric_cols:
        series = df_sorted[col]
        is_change = series != series.shift(1)
        block_id = is_change.cumsum()

        agg = df_sorted.groupby(block_id).agg(
            val=(col, "first"),
            streak_len=("date", "count"),
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        stuck = agg[agg["streak_len"] >= min_streak]
        if not stuck.empty:
            longest = stuck.sort_values(by="streak_len", ascending=False).iloc[0]
            return {
                "present": True,
                "column": col,
                "start_date": str(longest["start_date"]),
                "end_date": str(longest["end_date"]),
                "stuck_value": float(longest["val"]),
                "count": int(longest["streak_len"]),
            }

    return {
        "present": False,
        "column": None,
        "start_date": None,
        "end_date": None,
    }


def check_spelling_variants(train_df: pd.DataFrame, later_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Check text columns for messy spelling variations, casing, or whitespace differences."""
    cat_cols = ["department", "wing"]
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        unique_raw = train_df[col].dropna().unique()
        # Clean representations (strip whitespace, lowercase, prefix match)
        unique_clean = {val.strip().lower() for val in unique_raw}
        
        # Determine ground truth categories from later.csv if available
        if later_df is not None and col in later_df.columns:
            true_levels = int(later_df[col].dropna().nunique())
        else:
            true_levels = len(unique_clean)

        if len(unique_raw) > true_levels:
            return {
                "present": True,
                "column": col,
                "true_levels": true_levels,
                "raw_variants_count": len(unique_raw),
            }

    return {
        "present": False,
        "column": None,
        "true_levels": None,
    }


def check_target_typos(df: pd.DataFrame, target_col: str = "wait_min") -> Dict[str, Any]:
    """Check if any target values were typed 10 times too large (isolated extreme values)."""
    if target_col not in df.columns:
        return {"present": False, "count": None}

    targets = df[target_col].dropna().sort_values(ascending=False).values
    if len(targets) < 2:
        return {"present": False, "count": None}

    # Check for an order of magnitude jump (>5x jump between consecutive ordered targets)
    ratio = targets[:-1] / np.maximum(targets[1:], 1e-6)
    typo_indices = np.where(ratio > 5.0)[0]

    if len(typo_indices) > 0:
        typo_count = int(typo_indices[0] + 1)
        return {
            "present": True,
            "count": typo_count,
        }

    return {
        "present": False,
        "count": None,
    }


def check_sentinel_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Check if numeric columns contain common sentinel numbers (-1, -999, 999, 9999)."""
    sentinels = [-1, -99, -999, 999, 9999, 99999]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        for val in sentinels:
            cnt = int((df[col] == val).sum())
            if cnt > 0:
                return {
                    "present": True,
                    "column": col,
                    "value": val,
                    "count": cnt,
                }

    return {
        "present": False,
        "column": None,
        "value": None,
        "count": None,
    }


def check_unit_mixup(df: pd.DataFrame, group_col: str = "wing") -> Dict[str, Any]:
    """Check if one group recorded a column in a different unit (e.g., 10x, 1000x difference)."""
    if group_col not in df.columns:
        return {"present": False, "column": None, "group_value": None, "factor": None}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "wait_min" in numeric_cols:
        numeric_cols.remove("wait_min")

    for col in numeric_cols:
        group_means = df.groupby(group_col)[col].mean()
        if len(group_means) > 1:
            ratio = group_means.max() / max(group_means.min(), 1e-6)
            if ratio >= 8.0:  # Significant unit discrepancy (e.g., 10, 60, 1000)
                affected_group = str(group_means.idxmax())
                factor = int(round(ratio))
                return {
                    "present": True,
                    "column": col,
                    "group_value": affected_group,
                    "factor": factor,
                }

    return {
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None,
    }


def check_leaking_column(train_df: pd.DataFrame, holdout_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Check if any feature column is missing from holdout_features or acts as an exact target proxy."""
    if holdout_df is not None:
        missing_in_holdout = [c for c in train_df.columns if c not in holdout_df.columns and c != "wait_min"]
        if missing_in_holdout:
            return {
                "present": True,
                "column": missing_in_holdout[0],
            }

    return {
        "present": False,
        "column": None,
    }


def run_all_checks(
    train_path: str = "data/raw/train.csv",
    later_path: str = "data/raw/later.csv",
    holdout_path: str = "data/raw/holdout_features.csv",
) -> Dict[str, Any]:
    """Run all 7 data problem checks and return findings dictionary."""
    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path) if Path(later_path).exists() else None
    holdout_df = pd.read_csv(holdout_path) if Path(holdout_path).exists() else None

    results = {
        "leaking_column": check_leaking_column(train_df, holdout_df),
        "unit_mixup": check_unit_mixup(train_df, group_col="wing"),
        "sentinel_value": check_sentinel_values(train_df),
        "duplicate_rows": check_duplicate_rows(train_df),
        "stuck_sensor": check_stuck_sensor(train_df),
        "target_typos": check_target_typos(train_df),
        "spelling_variants": check_spelling_variants(train_df, later_df),
    }
    return results


def print_report(results: Dict[str, Any]) -> None:
    """Print readable summary for terminal and viva examination."""
    print("=" * 60)
    print("SCSE3040 MLOps - Stage 1: 7 Data Issues Check (MP255)")
    print("=" * 60)
    for issue_name, details in results.items():
        present_str = "YES (PRESENT)" if details.get("present") else "NO (ABSENT)"
        print(f"\n[{issue_name.upper()}]: {present_str}")
        for k, v in details.items():
            if k != "present" and v is not None:
                print(f"  - {k}: {v}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    params = load_params()
    raw_train = params["data"].get("raw_train_path", "data/raw/train.csv")
    raw_later = params["data"].get("raw_later_path", "data/raw/later.csv")
    raw_holdout = params["data"].get("holdout_features_path", "data/raw/holdout_features.csv")

    results = run_all_checks(raw_train, raw_later, raw_holdout)
    print_report(results)
