"""Unsupervised anomaly detection on 10-minute sensor windows.

Rules catch what we already know to look for. An Isolation Forest trained on
raw-signal features is a second opinion that can surface patterns no rule was
written for. Because the data is synthetic, we can score both against the
injected-fault log, which a real deployment would replace with QC analyst labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

FEATURES = ["n_readings", "max_gap_s", "depth_std", "depth_max_step",
            "temp_std", "roll_abs_max", "roll_std", "rpm_std"]
LABEL_FAULT_TYPES = ["spike", "range", "flatline", "gap"]  # drift is a daily-level SPC check


def label_windows(feats: pd.DataFrame, fault_log: pd.DataFrame, minutes: int = 10) -> pd.DataFrame:
    """Mark a window as faulty if any injected (non-drift) fault overlaps it."""
    f = fault_log[fault_log.fault_type.isin(LABEL_FAULT_TYPES)].copy()
    f["start"] = pd.to_datetime(f.start_ts, utc=True).dt.tz_localize(None)
    f["end"] = pd.to_datetime(f.end_ts, utc=True).dt.tz_localize(None)
    out = feats.copy()
    out["window_start"] = pd.to_datetime(out.window_start)
    out["window_end"] = out.window_start + pd.Timedelta(minutes=minutes)
    out["is_fault"] = False
    for v, grp in f.groupby("vessel_id"):
        m = out.vessel_id == v
        ws, we = out.loc[m, "window_start"].values, out.loc[m, "window_end"].values
        hit = np.zeros(m.sum(), dtype=bool)
        for s, e in zip(grp.start.values, grp.end.values):
            hit |= (ws <= e) & (we > s)
        out.loc[m, "is_fault"] = hit
    return out


def train_and_score(feats: pd.DataFrame, train_until: str, contamination: float = 0.04, seed: int = 7):
    """Time-based split: train on early days, score later days (no look-ahead)."""
    feats = feats.sort_values(["vessel_id", "window_start"]).reset_index(drop=True)
    cut = pd.Timestamp(train_until)
    train = feats[feats.window_start < cut]
    test = feats[feats.window_start >= cut].copy()
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
    model.fit(train[FEATURES])
    test["anomaly_score"] = -model.score_samples(test[FEATURES])
    test["is_anomaly"] = model.predict(test[FEATURES]) == -1
    metrics = {
        "train_windows": int(len(train)), "test_windows": int(len(test)),
        "test_fault_windows": int(test.is_fault.sum()),
        "precision": float(precision_score(test.is_fault, test.is_anomaly, zero_division=0)),
        "recall": float(recall_score(test.is_fault, test.is_anomaly, zero_division=0)),
    }
    return model, test, metrics
