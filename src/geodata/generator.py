"""Synthetic marine survey data with realistic, documented faults.

Writes raw files the way field systems deliver them: one CSV per vessel per
day, vessel configuration snapshots, and a survey-line plan. Every injected
fault is recorded in _truth/fault_log.csv so the pipeline's QC can be scored
against known answers. The truth file is for evaluation only and is never
ingested as data.

All vessel names, projects and values are fictional.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EXPECTED_INTERVAL_S

START_DATE = "2026-09-01"
N_DAYS = 7
VESSELS = [
    # vessel_id, type, speed unit reported by its logger
    ("USV-01", "USV", "m/s"),
    ("USV-02", "USV", "kn"),
    ("SV-AURORA", "Survey vessel", "m/s"),
]


@dataclass
class Fault:
    vessel_id: str
    sensor: str
    fault_type: str
    start_ts: str
    end_ts: str
    note: str


def _base_day(rng, vessel_id, day, speed_unit):
    """One day of clean 10-second telemetry for one vessel."""
    t0 = pd.Timestamp(START_DATE) + pd.Timedelta(days=day)
    n = 24 * 3600 // EXPECTED_INTERVAL_S
    ts = t0 + pd.to_timedelta(np.arange(n) * EXPECTED_INTERVAL_S, unit="s")
    hours = (ts - t0).total_seconds() / 3600.0

    # Survey pattern: back-and-forth lines, slow transit at night.
    on_survey = (hours >= 6) & (hours < 18)
    speed_ms = np.where(on_survey, 2.0, 0.8) + rng.normal(0, 0.05, n)
    heading = np.where((hours.astype(int) % 2) == 0, 90.0, 270.0) + rng.normal(0, 1.5, n)
    offset = {"USV-01": 0.00, "USV-02": 0.02, "SV-AURORA": 0.04}[vessel_id]
    lat = 1.20 + offset + 0.0005 * np.sin(hours / 3) + rng.normal(0, 1e-6, n)
    lon = 103.90 + np.cumsum(np.cos(np.radians(heading)) * speed_ms * EXPECTED_INTERVAL_S) / 111_000

    # Seabed varies smoothly along track; echo sounder noise ~0.15 m.
    # Use hours since campaign start so the seabed is continuous across midnight
    # (a per-day reset would create a fake depth "jump" at 00:00).
    t_campaign = hours + 24 * day
    seabed = 35 + 12 * np.sin(t_campaign / 1.7) + 5 * np.sin(t_campaign / 5.3)
    depth = seabed + rng.normal(0, 0.15, n)

    # Same water body for all vessels: shared diurnal cycle + small noise.
    sea_temp = 29.6 + 0.4 * np.sin((hours - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.03, n)

    pitch = 1.5 * np.sin(np.arange(n) / 7.0) + rng.normal(0, 0.4, n)
    roll = 3.0 * np.sin(np.arange(n) / 5.0) + rng.normal(0, 0.6, n)
    rpm = np.where(on_survey, 1250, 700) + rng.normal(0, 20, n)

    speed_out = speed_ms / 0.514444 if speed_unit == "kn" else speed_ms

    return pd.DataFrame({
        "vessel_id": vessel_id,
        "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lat": lat.round(6), "lon": lon.round(6),
        "heading_deg": (heading % 360).round(1),
        "speed": np.round(speed_out, 3), "speed_unit": speed_unit,
        "pitch_deg": pitch.round(2), "roll_deg": roll.round(2),
        "water_depth_m": depth.round(2), "sea_temp_c": sea_temp.round(3),
        "engine_rpm": rpm.round(0),
    })


def _span(df, i0, i1):
    return df["ts"].iloc[i0], df["ts"].iloc[i1 - 1]


def generate(out_dir: str, seed: int = 42) -> dict:
    """Generate the raw landing zone under out_dir. Returns a summary dict."""
    rng = np.random.default_rng(seed)
    tele_dir = os.path.join(out_dir, "telemetry")
    cfg_dir = os.path.join(out_dir, "vessel_config")
    line_dir = os.path.join(out_dir, "survey_lines")
    truth_dir = os.path.join(out_dir, "_truth")
    for d in (tele_dir, cfg_dir, line_dir, truth_dir):
        os.makedirs(d, exist_ok=True)

    faults: list[Fault] = []
    files_written = 0

    for day in range(N_DAYS):
        date = (pd.Timestamp(START_DATE) + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
        for vessel_id, _vtype, unit in VESSELS:
            df = _base_day(rng, vessel_id, day, unit)

            # 1) Depth spikes (multipath / bubble sweep-down): 6 per vessel-day.
            for i in rng.choice(np.arange(100, len(df) - 100), size=6, replace=False):
                df.loc[i, "water_depth_m"] = round(float(df.loc[i, "water_depth_m"]) + rng.choice([-1, 1]) * rng.uniform(15, 60), 2)
                s, e = _span(df, i, i + 1)
                faults.append(Fault(vessel_id, "water_depth_m", "spike", s, e, "single-point echo sounder spike"))

            # 2) Roll sentinel value 999 from a motion-sensor reset: USV-01 daily.
            if vessel_id == "USV-01":
                for i in rng.choice(np.arange(100, len(df) - 100), size=3, replace=False):
                    df.loc[i, "roll_deg"] = 999.0
                    s, e = _span(df, i, i + 1)
                    faults.append(Fault(vessel_id, "roll_deg", "range", s, e, "motion sensor sentinel 999"))

            # 3) Stuck temperature sensor for 2 hours: USV-01 on day 3.
            if vessel_id == "USV-01" and day == 2:
                i0, i1 = 4000, 4000 + 720
                df.loc[i0:i1 - 1, "sea_temp_c"] = df.loc[i0, "sea_temp_c"]
                s, e = _span(df, i0, i1)
                faults.append(Fault(vessel_id, "sea_temp_c", "flatline", s, e, "temperature sensor stuck for 2 h"))

            # 4) Calibration drift: SV-AURORA sea temp +1.5 C on days 6-7.
            if vessel_id == "SV-AURORA" and day >= 5:
                df["sea_temp_c"] = (df["sea_temp_c"] + 1.5).round(3)
                s, e = _span(df, 0, len(df))
                faults.append(Fault(vessel_id, "sea_temp_c", "drift", s, e, "temperature calibration offset +1.5 C"))

            # 5) Comms dropout: USV-02 loses 45 minutes on day 4.
            if vessel_id == "USV-02" and day == 3:
                i0, i1 = 5000, 5000 + 270
                s, e = _span(df, i0, i1)
                df = df.drop(index=range(i0, i1)).reset_index(drop=True)
                faults.append(Fault(vessel_id, "all", "gap", s, e, "45 min telemetry dropout"))

            # 6) Malformed rows from a logger glitch: 2 per vessel-day.
            for i in rng.choice(np.arange(10, len(df) - 10), size=2, replace=False):
                df.loc[i, "ts"] = "NaT-ERR"

            # 7) Schema drift: USV firmware adds battery_pct from day 5.
            if vessel_id.startswith("USV") and day >= 4:
                df["battery_pct"] = np.clip(95 - np.arange(len(df)) * 0.008, 5, 100).round(1)

            path = os.path.join(tele_dir, f"{vessel_id}_{date}.csv")
            df.to_csv(path, index=False)
            files_written += 1

            # 8) Re-sent file (duplicates): SV-AURORA day 2 delivered twice.
            if vessel_id == "SV-AURORA" and day == 1:
                df.to_csv(os.path.join(tele_dir, f"{vessel_id}_{date}_resend.csv"), index=False)
                files_written += 1

    # Vessel configuration snapshots: USV-02 echo sounder replaced on day 5.
    cfg_rows = [
        ("USV-01", "USV", "EchoSound-A", "fw-2.1", "MotionRef-3"),
        ("USV-02", "USV", "EchoSound-A", "fw-2.1", "MotionRef-3"),
        ("SV-AURORA", "Survey vessel", "MultiBeam-X", "fw-5.0", "MotionRef-5"),
    ]
    cols = ["vessel_id", "vessel_type", "echo_sounder", "firmware", "motion_sensor"]
    for day in range(N_DAYS):
        date = (pd.Timestamp(START_DATE) + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
        snap = pd.DataFrame(cfg_rows, columns=cols)
        if day >= 4:
            snap.loc[snap.vessel_id == "USV-02", ["echo_sounder", "firmware"]] = ["EchoSound-B", "fw-3.0"]
        snap["snapshot_date"] = date
        snap.to_csv(os.path.join(cfg_dir, f"config_{date}.csv"), index=False)

    # Survey-line plan: two 4-hour lines per vessel per day.
    lines = []
    for day in range(N_DAYS):
        d0 = pd.Timestamp(START_DATE) + pd.Timedelta(days=day)
        for vessel_id, _t, _u in VESSELS:
            for k, (h0, h1) in enumerate([(6, 10), (13, 17)]):
                lines.append({
                    "line_id": f"{vessel_id}-{d0:%m%d}-L{k + 1}",
                    "vessel_id": vessel_id,
                    "project_code": "PRJ-NORTH" if vessel_id != "SV-AURORA" else "PRJ-SOUTH",
                    "planned_start": (d0 + pd.Timedelta(hours=h0)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "planned_end": (d0 + pd.Timedelta(hours=h1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                })
    pd.DataFrame(lines).to_csv(os.path.join(line_dir, "survey_lines.csv"), index=False)

    pd.DataFrame([f.__dict__ for f in faults]).to_csv(os.path.join(truth_dir, "fault_log.csv"), index=False)

    return {"telemetry_files": files_written, "faults_injected": len(faults),
            "survey_lines": len(lines), "days": N_DAYS, "vessels": len(VESSELS)}


if __name__ == "__main__":
    import sys
    print(generate(sys.argv[1] if len(sys.argv) > 1 else "./local_lake/raw"))
