"""
mk_5/shared/preprocessing.py

Load training and test data, do minimal cleaning, produce train/val split.

The cleaning is deliberately minimal because:
- The course teaches that classifier choice + features matter more than
  preprocessing for short documents like reviews.
- Aggressive normalization (lowercasing, stemming, etc.) destroys signal that
  the n-gram + TF-IDF feature extractor would otherwise capture.

Data path: <repo_root>/data/  (one level up from mk_5/).
"""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split

HERE      = Path(__file__).resolve().parent       # mk_5/shared
MK5_DIR   = HERE.parent                           # mk_5
REPO_ROOT = MK5_DIR.parent                        # repo root
DATA_DIR  = REPO_ROOT / "data"


# ---------------------------------------------------------------- HTML strip
_HTML_BR = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)


def _clean_text(s: str) -> str:
    """Strip HTML <br> tags and collapse whitespace. Nothing else."""
    if not isinstance(s, str):
        return ""
    s = _HTML_BR.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------- loaders
def load_train() -> pd.DataFrame:
    """Load and clean train.csv. Returns DataFrame with columns: ID, TEXT, LABEL."""
    path = DATA_DIR / "train.csv"
    df = pd.read_csv(path)
    df["TEXT"] = df["TEXT"].apply(_clean_text)
    # Drop empty text rows after cleaning
    df = df[df["TEXT"].str.len() > 0].reset_index(drop=True)
    return df


def load_test() -> pd.DataFrame:
    """Load and clean test.csv. Returns DataFrame with columns: ID, TEXT."""
    path = DATA_DIR / "test.csv"
    df = pd.read_csv(path)
    df["TEXT"] = df["TEXT"].apply(_clean_text)
    return df


def load_sample_submission() -> pd.DataFrame:
    """Load sample_submission.csv to get the canonical ID order for submissions."""
    path = DATA_DIR / "sample_submission.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------- splitting
def train_val_split(df: pd.DataFrame, val_frac: float = 0.15, seed: int = 42):
    """Stratified train/val split. Returns (X_train, X_val, y_train, y_val)."""
    X_tr, X_va, y_tr, y_va = train_test_split(
        df["TEXT"].values,
        df["LABEL"].values,
        test_size=val_frac,
        stratify=df["LABEL"].values,
        random_state=seed,
    )
    return X_tr, X_va, y_tr, y_va
