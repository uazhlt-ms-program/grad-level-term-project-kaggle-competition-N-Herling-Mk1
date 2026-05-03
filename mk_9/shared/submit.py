"""
mk_2/shared/submit.py

Build a Kaggle submission CSV with predictions in the canonical ID order
from sample_submission.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import load_test, load_sample_submission


def write_submission(y_pred: np.ndarray, out_path: Path) -> Path:
    """
    Build a submission CSV with predictions y_pred aligned to the canonical
    ID order from sample_submission.csv.

    Assumes y_pred is in the same row order as test.csv as loaded by
    preprocessing.load_test().
    """
    test_df = load_test()
    sample  = load_sample_submission()

    # Build prediction frame in test.csv's order
    pred_df = pd.DataFrame({"ID": test_df["ID"].values, "LABEL": y_pred.astype(int)})

    # Reorder to match sample_submission's ID order
    pred_df = pred_df.set_index("ID").loc[sample["ID"].values].reset_index()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    return out_path
