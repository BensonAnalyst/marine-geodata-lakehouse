"""Silver: standardise, de-duplicate, quarantine and quality-flag telemetry.

Readings are never deleted for being suspicious. Each row carries a list of
QC flags, so Gold products decide per sensor what is usable and analysts can
always see why a value was excluded.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .config import (FLATLINE_RULES, GAP_THRESHOLD_S, KNOTS_TO_MS,
                     RANGE_RULES, SENSOR_FLAG_MAP, SPIKE_RULES)

NUMERIC_COLS = ["lat", "lon", "heading_deg", "speed", "pitch_deg", "roll_deg",
                "water_depth_m", "sea_temp_c", "engine_rpm"]


# ---------------------------------------------------------------- standardise
def standardise(bronze: DataFrame) -> DataFrame:
    """Cast types, parse UTC timestamps, convert units, surface rescued fields."""
    df = bronze
    for c in NUMERIC_COLS:
        df = df.withColumn(c, F.col(c).try_cast("double"))
    df = (df
          # try_ variants return NULL instead of failing the job (ANSI mode is on by default).
          .withColumn("ts_raw", F.col("ts"))
          .withColumn("ts", F.try_to_timestamp("ts", F.lit("yyyy-MM-dd'T'HH:mm:ss'Z'")))
          .withColumn("speed_ms", F.when(F.col("speed_unit") == "kn", F.col("speed") * KNOTS_TO_MS)
                                   .otherwise(F.col("speed")))
          # Field added by new USV firmware arrives via _rescued_data.
          .withColumn("battery_pct", F.get_json_object("_rescued_data", "$.battery_pct").try_cast("double"))
          .withColumn("obs_date", F.to_date("ts"))
          .drop("speed", "speed_unit"))
    return df


def split_quarantine(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Rows that cannot be placed in time or attributed to a vessel are quarantined."""
    bad = F.col("ts").isNull() | F.col("vessel_id").isNull()
    quarantine = df.filter(bad).withColumn(
        "quarantine_reason",
        F.when(F.col("vessel_id").isNull(), F.lit("missing vessel_id")).otherwise(F.lit("unparseable timestamp")))
    return df.filter(~bad), quarantine


def deduplicate(df: DataFrame) -> DataFrame:
    """One reading per vessel per timestamp; the most recently ingested file wins."""
    w = Window.partitionBy("vessel_id", "ts").orderBy(F.col("_ingest_time").desc(), F.col("_source_file").desc())
    return df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")


# ------------------------------------------------------------------- QC rules
def _ordered(rows_before: int = 0, rows_after: int = 0):
    return (Window.partitionBy("vessel_id").orderBy("ts").rowsBetween(rows_before, rows_after))


def flag_range(df: DataFrame, rules: dict = RANGE_RULES) -> DataFrame:
    for col, (lo, hi) in rules.items():
        if col not in df.columns:
            continue
        df = df.withColumn(f"_f_{col}_range", (F.col(col) < lo) | (F.col(col) > hi))
    return df


def flag_spikes(df: DataFrame, rules: dict = SPIKE_RULES) -> DataFrame:
    """Compare each value with the mean and spread of the readings just before it.

    The current value is excluded from its own baseline, so a spike cannot hide
    by inflating the standard deviation it is judged against.
    """
    w_prev = Window.partitionBy("vessel_id").orderBy("ts")
    secs_since_prev = F.unix_timestamp("ts") - F.unix_timestamp(F.lag("ts").over(w_prev))
    for col, r in rules.items():
        n, k = r["window_rows"], r["k_sigma"]
        w = _ordered(-n, -1)
        # Exclude out-of-range values from the baseline.
        clean = F.when(~F.col(f"_f_{col}_range"), F.col(col)) if f"_f_{col}_range" in df.columns else F.col(col)
        # Only judge a reading when the baseline is trustworthy: enough history,
        # and no telemetry gap just before it (the seabed may have changed).
        judged = (F.col("_n") >= r.get("min_baseline_rows", 10)) & (F.col("_dt") <= GAP_THRESHOLD_S)
        df = (df.withColumn("_mu", F.avg(clean).over(w))
                .withColumn("_sd", F.stddev(clean).over(w))
                .withColumn("_n", F.count(clean).over(w))
                .withColumn("_dt", secs_since_prev)
                .withColumn(f"_f_{col}_spike", F.coalesce(
                    judged & (F.abs(F.col(col) - F.col("_mu")) > k * F.greatest(F.col("_sd"), F.lit(0.05))),
                    F.lit(False)))
                .drop("_mu", "_sd", "_n", "_dt"))
    return df


def flag_flatline(df: DataFrame, rules: dict = FLATLINE_RULES) -> DataFrame:
    """Flag every reading inside a run of identical values of at least min_rows."""
    w = Window.partitionBy("vessel_id").orderBy("ts")
    for col, r in rules.items():
        changed = (F.col(col) != F.lag(col).over(w)) | F.lag(col).over(w).isNull()
        df = df.withColumn("_chg", changed.cast("int"))
        df = df.withColumn("_run", F.sum("_chg").over(w.rowsBetween(Window.unboundedPreceding, 0)))
        run_len = F.count(F.lit(1)).over(Window.partitionBy("vessel_id", "_run"))
        df = df.withColumn(f"_f_{col}_flatline", run_len >= r["min_rows"]).drop("_chg", "_run")
    return df


def flag_gaps(df: DataFrame, threshold_s: int = GAP_THRESHOLD_S) -> DataFrame:
    """Seconds since the previous reading; a gap is longer than threshold_s."""
    w = Window.partitionBy("vessel_id").orderBy("ts")
    prev = F.lag("ts").over(w)
    df = df.withColumn("gap_before_s", F.unix_timestamp("ts") - F.unix_timestamp(prev))
    return df.withColumn("_f_gap", F.coalesce(F.col("gap_before_s") > threshold_s, F.lit(False)))


def collect_flags(df: DataFrame) -> DataFrame:
    """Turn the _f_* booleans into one readable array column plus per-sensor validity."""
    flag_cols = [c for c in df.columns if c.startswith("_f_")]
    names = F.array(*[F.when(F.col(c), F.lit(c[3:])) for c in flag_cols])
    df = df.withColumn("qc_flags", F.filter(names, lambda x: x.isNotNull()))
    for sensor, flags in SENSOR_FLAG_MAP.items():
        bad = F.lit(False)
        for f in flags:
            if f"_f_{f}" in df.columns:
                bad = bad | F.col(f"_f_{f}")
        df = df.withColumn(f"{sensor}_valid", ~bad)
    df = df.withColumn("qc_ok", F.size("qc_flags") == 0)
    return df.drop(*flag_cols)


def apply_qc(df: DataFrame) -> DataFrame:
    df = flag_range(df)
    df = flag_spikes(df)
    df = flag_flatline(df)
    df = flag_gaps(df)
    return collect_flags(df)


def build_silver(bronze: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Full Bronze -> Silver step. Returns (silver, quarantine)."""
    std = standardise(bronze)
    good, quarantine = split_quarantine(std)
    silver = apply_qc(deduplicate(good))
    cols = ["vessel_id", "ts", "obs_date", "lat", "lon", "heading_deg", "speed_ms",
            "pitch_deg", "roll_deg", "water_depth_m", "sea_temp_c", "engine_rpm",
            "battery_pct", "gap_before_s", "qc_flags", "qc_ok",
            *[f"{s}_valid" for s in SENSOR_FLAG_MAP], "_source_file", "_ingest_time"]
    return silver.select(*cols), quarantine


# ------------------------------------------------------------ SCD2 dimension
def build_scd2(snapshots: DataFrame, key: str, attrs: list[str], snapshot_col: str = "snapshot_date") -> DataFrame:
    """Turn daily configuration snapshots into an SCD Type 2 history.

    A new version starts only when an attribute actually changes, so a vessel's
    readings can always be joined to the configuration in force at that time.
    """
    w = Window.partitionBy(key).orderBy(snapshot_col)
    h = F.sha2(F.concat_ws("||", *[F.coalesce(F.col(a).cast("string"), F.lit("")) for a in attrs]), 256)
    df = (snapshots.withColumn(snapshot_col, F.to_date(snapshot_col))
          .withColumn("_hash", h)
          .withColumn("_prev", F.lag("_hash").over(w))
          .filter(F.col("_prev").isNull() | (F.col("_prev") != F.col("_hash"))))
    w2 = Window.partitionBy(key).orderBy(snapshot_col)
    df = (df.withColumn("valid_from", F.col(snapshot_col))
            .withColumn("valid_to", F.date_sub(F.lead(snapshot_col).over(w2), 1))
            .withColumn("is_current", F.lead(snapshot_col).over(w2).isNull())
            .withColumn("version", F.row_number().over(w2)))
    return df.select(key, *attrs, "version", "valid_from", "valid_to", "is_current")
