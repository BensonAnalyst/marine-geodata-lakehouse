"""Data-quality gates and QC scoring against the known fault log.

`run_dq_checks` returns one row per check. The job fails if any CRITICAL check
fails, so bad data never reaches Gold consumers unnoticed.
"""
from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .config import DQ_THRESHOLDS

# Which Silver flag should catch each injected fault type.
FAULT_TO_FLAG = {
    ("water_depth_m", "spike"): "water_depth_m_spike",
    ("roll_deg", "range"): "roll_deg_range",
    ("sea_temp_c", "flatline"): "sea_temp_c_flatline",
    ("all", "gap"): "gap",
}


def qc_recall(silver: DataFrame, fault_log: pd.DataFrame) -> pd.DataFrame:
    """For each injected row-level fault, did Silver raise the matching flag?

    Gap faults are caught on the first reading AFTER the gap, so their search
    window extends 60 s past the fault end.
    """
    flagged = (silver.select("vessel_id", "ts", F.explode("qc_flags").alias("flag"))
                     .toPandas())
    flagged["ts"] = pd.to_datetime(flagged.ts)
    rows = []
    for _, f in fault_log.iterrows():
        flag = FAULT_TO_FLAG.get((f.sensor, f.fault_type))
        if flag is None:
            continue  # drift is scored at daily level, see drift_detection
        s = pd.to_datetime(f.start_ts, utc=True).tz_localize(None)
        e = pd.to_datetime(f.end_ts, utc=True).tz_localize(None) + pd.Timedelta(seconds=60 if flag == "gap" else 0)
        hit = flagged[(flagged.vessel_id == f.vessel_id) & (flagged.flag == flag)
                      & (flagged.ts >= s) & (flagged.ts <= e)]
        rows.append({"fault_type": f.fault_type, "vessel_id": f.vessel_id, "detected": len(hit) > 0})
    df = pd.DataFrame(rows)
    return (df.groupby("fault_type")
              .agg(injected=("detected", "size"), detected=("detected", "sum"))
              .assign(recall_pct=lambda d: (d.detected / d.injected * 100).round(1))
              .reset_index())


def drift_detection(daily_health: DataFrame, fault_log: pd.DataFrame) -> pd.DataFrame:
    """Compare drift-flagged vessel-days with the injected calibration drift."""
    flagged = daily_health.select("vessel_id", "obs_date", "temp_drift_flag").toPandas()
    d = fault_log[fault_log.fault_type == "drift"].copy()
    d["obs_date"] = pd.to_datetime(d.start_ts).dt.date
    truth = set(zip(d.vessel_id, d.obs_date))
    flagged["truth"] = [(v, o) in truth for v, o in zip(flagged.vessel_id, flagged.obs_date)]
    tp = int((flagged.truth & flagged.temp_drift_flag).sum())
    fp = int((~flagged.truth & flagged.temp_drift_flag).sum())
    fn = int((flagged.truth & ~flagged.temp_drift_flag).sum())
    return pd.DataFrame([{"drift_days_injected": len(truth), "detected": tp, "false_alarms": fp, "missed": fn}])


def run_dq_checks(silver: DataFrame, daily_health: DataFrame, recall: pd.DataFrame) -> pd.DataFrame:
    t = DQ_THRESHOLDS
    dup = silver.groupBy("vessel_id", "ts").count().filter("count > 1").count()
    fleet_comp = daily_health.agg(F.avg("completeness_pct")).first()[0]
    min_recall = float(recall.recall_pct.min()) if len(recall) else 0.0
    checks = [
        ("No duplicate vessel/timestamp keys in Silver", "CRITICAL", dup, f"<= {t['max_duplicate_keys_silver']}",
         dup <= t["max_duplicate_keys_silver"]),
        ("Fleet average completeness", "WARN", round(fleet_comp, 2), f">= {t['min_fleet_completeness_pct']}%",
         fleet_comp >= t["min_fleet_completeness_pct"]),
        ("Lowest QC recall across fault types", "CRITICAL", min_recall, f">= {t['min_qc_recall_pct']}%",
         min_recall >= t["min_qc_recall_pct"]),
    ]
    return pd.DataFrame(checks, columns=["check", "severity", "observed", "threshold", "passed"])


def enforce(results: pd.DataFrame) -> None:
    failed = results[(results.severity == "CRITICAL") & (~results.passed)]
    if len(failed):
        raise AssertionError("Critical data-quality checks failed:\n" + failed.to_string(index=False))
