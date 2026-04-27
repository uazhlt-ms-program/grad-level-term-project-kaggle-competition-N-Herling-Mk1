"""
shared/submit.py
Write a Kaggle-format submission CSV.

The required format from sample_submission.csv:
    ID,LABEL
    1087873697464833975,1
    5853461517618045821,1
    ...

IDs must match the test set ordering (we re-key by ID, not by row index).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def write_submission(
    test_ids: np.ndarray,
    predictions: np.ndarray,
    out_path: Path | str,
    sample_path: Path | str = None,
) -> Path:
    """
    Build a submission CSV. If sample_path is given, the output is reordered
    to match the sample_submission.csv ID order (defensive — Kaggle is OK
    with any order so long as every ID appears exactly once).

    Returns the output path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"ID": test_ids, "LABEL": predictions.astype(int)})

    if sample_path is not None:
        sample = pd.read_csv(sample_path)
        df = sample[["ID"]].merge(df, on="ID", how="left")
        n_missing = df["LABEL"].isna().sum()
        if n_missing > 0:
            raise ValueError(
                f"{n_missing} test IDs in sample_submission.csv are missing predictions."
            )
        df["LABEL"] = df["LABEL"].astype(int)

    df.to_csv(out_path, index=False)
    return out_path
