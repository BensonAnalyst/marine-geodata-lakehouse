"""Central configuration: sensor rules, QC thresholds and data-product rules.

Keeping thresholds in one place means a change is a reviewed code change
(git + CI), not an edit buried in a notebook.
"""

# Telemetry is logged every 10 seconds per vessel.
EXPECTED_INTERVAL_S = 10
READINGS_PER_DAY = 24 * 60 * 60 // EXPECTED_INTERVAL_S  # 8,640

# Columns every telemetry file must contain. Anything else is "schema drift"
# and lands in _rescued_data in Bronze instead of being silently dropped.
TELEMETRY_BASE_COLUMNS = [
    "vessel_id", "ts", "lat", "lon", "heading_deg", "speed", "speed_unit",
    "pitch_deg", "roll_deg", "water_depth_m", "sea_temp_c", "engine_rpm",
]

KNOTS_TO_MS = 0.514444

# Physical range checks (inclusive). Values outside are impossible or sentinels.
RANGE_RULES = {
    "water_depth_m": (0.5, 500.0),
    "sea_temp_c": (-2.0, 40.0),
    "pitch_deg": (-45.0, 45.0),
    "roll_deg": (-45.0, 45.0),
    "engine_rpm": (0.0, 3000.0),
}

# Spike detection: robust deviation from the preceding window.
SPIKE_RULES = {
    "water_depth_m": {"window_rows": 30, "k_sigma": 6.0},
}

# Flatline (stuck sensor): identical consecutive values for >= min_rows.
FLATLINE_RULES = {
    "sea_temp_c": {"min_rows": 60},      # 10 minutes unchanged
    "water_depth_m": {"min_rows": 60},
}

# A gap is any interval between consecutive readings longer than this.
GAP_THRESHOLD_S = 60

# Which QC flags make a reading unusable for each sensor.
SENSOR_FLAG_MAP = {
    "water_depth_m": ["water_depth_m_range", "water_depth_m_spike", "water_depth_m_flatline"],
    "sea_temp_c": ["sea_temp_c_range", "sea_temp_c_flatline"],
    "roll_deg": ["roll_deg_range"],
    "pitch_deg": ["pitch_deg_range"],
    "engine_rpm": ["engine_rpm_range"],
}

# Drift (calibration offset) on daily means, SPC X-bar style, fleet-relative.
DRIFT_BASELINE_DAYS = 3
DRIFT_K_SIGMA = 3.0
DRIFT_MIN_TOLERANCE_C = 0.3

# Survey-line delivery verdict thresholds (percent).
LINE_RULES = {
    "deliverable": {"completeness": 98.0, "depth_valid": 98.0},
    "review": {"completeness": 90.0, "depth_valid": 95.0},
}

# Data-quality gates the job enforces (critical = job fails).
DQ_THRESHOLDS = {
    "max_duplicate_keys_silver": 0,
    "min_fleet_completeness_pct": 90.0,
    "min_qc_recall_pct": 90.0,
}
