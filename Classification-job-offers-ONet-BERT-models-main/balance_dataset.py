"""
Balance Job Descriptions Dataset

This script balances classes via undersampling (down to the minority class size) and performs a stratified train/test split by 'label'.

It saves two CSVs (train/test) and can optionally write simple reports.

Usage:
  python balance_dataset.py \
    --input dataset.csv \
    --test-size 0.1 \
    --random-state 42 \
    --train-out dataset_train.csv \
    --test-out dataset_test.csv \
    --report-dir ./reports
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def balance_undersample(df: pd.DataFrame, label_col: str, random_state: int) -> pd.DataFrame:
    """Undersample each class to the minority class size."""
    counts = df[label_col].value_counts(dropna=False)
    min_count = counts.min()
    balanced = (
        df.groupby(label_col, group_keys=False)
          .apply(lambda x: x.sample(n=min_count, random_state=random_state))
          .reset_index(drop=True)
    )
    return balanced


def process(
    input_csv: Path,
    test_size: float,
    random_state: int,
    train_out: Path,
    test_out: Path,
    report_dir: Optional[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    print(f"[load] Reading: {input_csv}")
    df = pd.read_csv(input_csv)

    # Validate required columns
    required = ["description", "Label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    # Pre-balance class counts
    pre_counts = df["Label"].value_counts().sort_index()

    # Balance by undersampling
    df_bal = balance_undersample(df, "Label", random_state)

    # Post-balance class counts
    post_counts = df_bal["Label"].value_counts().sort_index()

    # Stratified split
    train_df, test_df = train_test_split(
        df_bal,
        test_size=test_size,
        random_state=random_state,
        stratify=df_bal["Label"],
    )

    # Save
    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    # Optional reports
    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)
        pre_counts.to_csv(report_dir / "class_counts_before.csv", header=["count"])
        post_counts.to_csv(report_dir / "class_counts_after.csv", header=["count"])
        pd.DataFrame({
            "n_rows_input": [len(df)],
            "n_rows_balanced": [len(df_bal)],
            "test_size": [test_size],
            "n_classes_before": [pre_counts.shape[0]],
            "n_classes_after": [post_counts.shape[0]],
            "min_class_size_after": [post_counts.min()],
        }).to_csv(report_dir / "summary.csv", index=False)

    return train_df, test_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Undersample balance + stratified train/test split (expects 'description' and 'label')."
    )
    p.add_argument("--input", default="dataset.csv", help="Path to the input CSV (must contain 'description' and 'label').")
    p.add_argument("--test-size", type=float, default=0.1, help="Proportion for the test split (0-1).")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--train-out", default="dataset_train.csv", help="Output CSV path for the training set.")
    p.add_argument("--test-out", default="dataset_test.csv", help="Output CSV path for the test set.")
    p.add_argument("--report-dir", default="", help="Directory to save reports; leave empty to disable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input)
    train_out = Path(args.train_out)
    test_out = Path(args.test_out)
    report_dir = Path(args.report_dir) if args.report_dir else None

    print("[info] Balancing and splitting…")
    train_df, test_df = process(
        input_csv=input_csv,
        test_size=args.test_size,
        random_state=args.random_state,
        train_out=train_out,
        test_out=test_out,
        report_dir=report_dir,
    )

    print(f"[done] train: {len(train_df)} rows | test: {len(test_df)} rows")
    if report_dir:
        print(f"[info] Reports saved at: {report_dir}")
    print(f"[saved] {train_out} | {test_out}")


if __name__ == "__main__":
    main()