"""
mk_1/shared/preprocessing.py

Load, clean, and split the LING/INFO 539 competition data.

Path resolution
---------------
This file lives at:  <repo_root>/model_tests/mk_1/shared/preprocessing.py
The Kaggle CSVs live at:  <repo_root>/data/

So DATA_DIR is computed three parents up:
    HERE        = model_tests/mk_1/shared
    HERE.parent = model_tests/mk_1
    REPO_ROOT   = HERE.parent.parent.parent
    DATA_DIR    = REPO_ROOT / "data"

If you ever rearrange the tree, this is the one place to fix it.

Usage
-----
    from shared.preprocessing import load_train, load_test, train_val_split
    df_tr = load_train()
    X_tr, X_va, y_tr, y_va = train_val_split(df_tr, val_frac=0.15, seed=42)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Paths --------------------------------------------------------------
HERE       = Path(__file__).resolve().parent      # model_tests/mk_1/shared
MK_DIR     = HERE.parent                          # model_tests/mk_1
REPO_ROOT  = MK_DIR.parent.parent  # repo root (one extra .parent because model_tests/ now in chain)
DATA_DIR   = REPO_ROOT / "data"
TRAIN_CSV  = DATA_DIR / "train.csv"
TEST_CSV   = DATA_DIR / "test.csv"
SAMPLE_CSV = DATA_DIR / "sample_submission.csv"


# Cleaning -----------------------------------------------------------
_BR_RE      = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE     = re.compile(r"<[^>]+>")
_WS_RE      = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """Conservative cleaning: HTML tag removal + whitespace normalize."""
    if not isinstance(s, str):
        return ""
    s = _BR_RE.sub(" ", s)
    s = _TAG_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# Loaders ------------------------------------------------------------
def load_train(path: Path | str = TRAIN_CSV, drop_null=True, clean=True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if drop_null:
        df = df.dropna(subset=["TEXT"]).reset_index(drop=True)
    if clean:
        df["TEXT"] = df["TEXT"].map(clean_text)
        df = df[df["TEXT"].str.len() > 0].reset_index(drop=True)
    return df


def load_test(path: Path | str = TEST_CSV, clean=True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if clean:
        df["TEXT"] = df["TEXT"].map(clean_text)
        # Don't drop test rows — we need a prediction for every ID.
        df["TEXT"] = df["TEXT"].where(df["TEXT"].str.len() > 0, " ")
    return df


# Splitting ----------------------------------------------------------
def train_val_split(df: pd.DataFrame, val_frac=0.15, seed=42):
    """Stratified train/val split. Returns (X_tr, X_va, y_tr, y_va) as arrays."""
    X = df["TEXT"].values
    y = df["LABEL"].values.astype(int)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=val_frac, stratify=y, random_state=seed
    )
    return X_tr, X_va, y_tr, y_va


# Quick survey ------------------------------------------------------
def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "count": df["LABEL"].value_counts().sort_index(),
        "pct":   (df["LABEL"].value_counts(normalize=True).sort_index() * 100).round(2),
    })
    out.index.name = "LABEL"
    return out


if __name__ == "__main__":
    print(f"DATA_DIR = {DATA_DIR}")
    df = load_train()
    print(f"train: {len(df):,} rows after cleaning")
    print(class_distribution(df))
    X_tr, X_va, y_tr, y_va = train_val_split(df)
    print(f"split: train={len(X_tr):,}  val={len(X_va):,}")
