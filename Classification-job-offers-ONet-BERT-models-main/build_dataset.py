"""
Build Unified Dataset from Per-Category Job Detail CSVs

This script merges all per-occupation CSVs produced in **Step 2** into a single
`dataset.csv`, adds derived fields from the job URL (e.g., `cod_empleo`,
`family_empleo`), and encodes a numeric `Label`.The description field is also 
preprocessed and offers with fewer than the number of words indicated in the 
call (50 by default) are eliminated.

Usage:
    python build_dataset.py \
        --input-dir output \
        --output dataset.csv \
        --min-words 50 \
        --occupations-csv All_Occupations.csv \
        --occupation-column Occupation

Notes:
    - `--occupations-csv` is optional; if given, only files matching those
      occupations will be included (by suffix in filename).
    - The script expects each input CSV to have at least: `title`, `company`,
      `description`, `url` (as written by Step 2).

"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd


URL_CODE_RE = re.compile(r"keyword=([0-9.-]+)")


def extract_codes(url: str) -> tuple[str | None, str | None]:
    """Extract `cod_empleo` and its family from a job URL.

    - cod_empleo: full code after `keyword=` (digits, dots, hyphens)
    - family_empleo: substring before the first hyphen in cod_empleo
    """
    if not isinstance(url, str):
        return None, None
    m = URL_CODE_RE.search(url)
    if not m:
        return None, None
    cod = m.group(1)
    fam = cod.split("-")[0] if "-" in cod else cod
    return cod, fam


def read_occupations_list(csv_path: Path, column: str) -> List[str]:
    df = pd.read_csv(csv_path, sep=",|;|\t", engine="python")
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in {csv_path}. Available: {list(df.columns)}"
        )
    return [str(x).strip() for x in df[column].dropna().tolist() if str(x).strip()]


def collect_files(input_dir: Path, occupations: Optional[List[str]]) -> List[Path]:
    files = []
    if occupations:
        for occ in occupations:
            # Expect pattern job_details_<occupation>.csv
            p = input_dir / f"job_details_{occ}.csv"
            if p.exists():
                files.append(p)
            else:
                print(f"  [warn] Missing file for occupation: {p.name}")
    else:
        files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")
    return files

# --------------------------- cleaning helpers --------------------------- #

def clean_text(text: object) -> str:
    """Lowercase; strip HTML tags; normalize whitespace."""
    s = "" if pd.isna(text) else str(text)
    s = s.lower()
    s = re.sub(r"<[^>]*>", " ", s)   # remove HTML tags
    s = re.sub(r"\s+", " ", s).strip()
    return s


def word_count(text: str) -> int:
    return 0 if not text else len(text.split())

def build_dataset(input_dir: Path, output_path: Path, min_words: int, occupations_csv: Optional[Path], occupation_col: str) -> pd.DataFrame:
    occ_list: Optional[List[str]] = None
    if occupations_csv:
        if not occupations_csv.exists():
            raise FileNotFoundError(f"Occupations CSV not found: {occupations_csv}")
        occ_list = read_occupations_list(occupations_csv, occupation_col)

    files = collect_files(input_dir, occ_list)

    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  [warn] Skipping unreadable file: {f.name} -> {e}")
            continue

        # Keep only expected columns if present
        expected_cols = ["title", "company", "description", "url"]
        present = [c for c in expected_cols if c in df.columns]
        if not present:
            print(f"  [warn] '{f.name}' has none of the expected columns; skipping.")
            continue
        df = df[present].copy()

        # Derive codes from URL
        if "url" in df.columns:
            codes = df["url"].apply(lambda u: pd.Series(extract_codes(u), index=["cod_empleo", "family_empleo"]))
            df = pd.concat([df, codes], axis=1)
        else:
            df["cod_empleo"] = None
            df["family_empleo"] = None

        frames.append(df)
        print(f"  [ok] Loaded {f.name} -> {df.shape[0]} rows")

    if not frames:
        raise RuntimeError("No valid rows found across input files.")

    total = pd.concat(frames, ignore_index=True, sort=False)
    print(f"[Before Clean] Wrote {len(total)} rows to: {output_path}")
    # Drop rows missing essential fields
    essential = [c for c in ("description", "family_empleo") if c in total.columns]
    if essential:
        total = total.dropna(subset=essential)

    # Create numeric Label from family_empleo
    if "family_empleo" in total.columns:
        total["Label"] = total["family_empleo"].astype("category").cat.codes
    
    # Clean text
    total["description"] = total["description"].apply(clean_text)
    total["word_count"] = total["description"].apply(word_count)

    # Filter by min words
    df_filt = total[total["word_count"] >= min_words].copy()
    if df_filt.empty:
        raise ValueError("All rows were filtered out by min-words threshold. Lower --min-words or check input.")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filt.to_csv(output_path, index=False)
    print(f"[done] Wrote {len(df_filt)} rows to: {output_path}")

    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge per-occupation job detail CSVs into a single labeled dataset.")
    p.add_argument("--input-dir", default="output", help="Directory containing job_details_<occupation>.csv files.")
    p.add_argument("--min-words", type=int, default=50, help="Minimum number of words to keep a row.")
    p.add_argument("--output", default="dataset.csv", help="Path to write the merged dataset.")
    p.add_argument("--occupations-csv", default="", help="Optional CSV with occupations to include.")
    p.add_argument("--occupation-column", default="Occupation", help="Column name in the occupations CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    occupations_csv = Path(args.occupations_csv) if args.occupations_csv else None

    build_dataset(
        input_dir=input_dir,
        output_path=output_path,
        min_words=args.min_words,
        occupations_csv=occupations_csv,
        occupation_col=args.occupation_column,
    )


if __name__ == "__main__":
    main()
