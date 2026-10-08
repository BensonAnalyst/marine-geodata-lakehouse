"""Gold: data products people and models actually use.

* vessel_daily_health   - one row per vessel per day: completeness, QC, drift
* survey_line_quality   - one row per planned survey line with a delivery verdict
* telemetry_minute      - star-schema fact at 1-minute grain, valid values only
* ml_features           - 10-minute windows of raw-signal features for anomaly ML
"""
from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .config import (DRIFT_BASELINE_DAYS, DRIFT_K_SIGMA, DRIFT_MIN_TOLERANCE_C,
                     GAP_THRESHOLD_S, LINE_RULES, READINGS_PER_DAY)


def _valid(col: str) -> F.Column:
    return F.when(F.col(f"{col}_valid"), F.col(col))


def vessel_daily_health(silver: DataFrame) -> DataFrame:
    daily = (silver.groupBy("vessel_id", "obs_date").agg(
        F.count("*").alias("readings"),
        F.sum(F.col("qc_ok").cast("int")).alias("readings_qc_ok"),
        F.sum(F.array_contains("qc_flags", "water_depth_m_spike").cast("int")).alias("depth_spikes"),
        F.sum(F.array_contains("qc_flags", "roll_deg_range").cast("int")).alias("roll_out_of_range"),
        (F.sum(F.array_contains("qc_flags", "sea_temp_c_flatline").cast("int")) * 10 / 60).alias("temp_flatline_min"),
        (F.sum(F.when(F.col("gap_before_s") > GAP_THRESHOLD_S, F.col("gap_before_s"))) / 60).alias("gap_min"),
        F.avg(_valid("sea_temp_c")).alias("mean_sea_temp_c"),
        F.avg(_valid("water_depth_m")).alias("mean_depth_m"),
        F.min("battery_pct").alias("min_battery_pct"),
    ))
    daily = (daily
             .withColumn("completeness_pct", F.round(F.col("readings") / READINGS_PER_DAY * 100, 2))
             .withColumn("qc_ok_pct", F.round(F.col("readings_qc_ok") / F.col("readings") * 100, 2))
             .withColumn("gap_min", F.round(F.coalesce(F.col("gap_min"), F.lit(0.0)), 1))
             .withColumn("temp_flatline_min", F.round("temp_flatline_min", 1)))
    return add_temp_drift_flag(daily)


def add_temp_drift_flag(daily: DataFrame) -> DataFrame:
    """SPC X-bar check on each vessel's daily mean temperature versus the fleet.

    Vessels working the same water should read nearly the same temperature. We
    learn each vessel's normal offset from the fleet median over the first
    baseline days, then flag days that move beyond k-sigma of that offset (with
    a minimum tolerance so a very quiet baseline doesn't cause false alarms).
    """
    fleet = daily.groupBy("obs_date").agg(F.percentile_approx("mean_sea_temp_c", 0.5).alias("fleet_median_temp_c"))
    d = (daily.join(fleet, "obs_date")
              .withColumn("temp_offset_c", F.col("mean_sea_temp_c") - F.col("fleet_median_temp_c")))
    first = d.agg(F.min("obs_date")).first()[0]
    base = (d.filter(F.col("obs_date") < F.date_add(F.lit(first), DRIFT_BASELINE_DAYS))
             .groupBy("vessel_id").agg(F.avg("temp_offset_c").alias("_cl"),
                                       F.stddev("temp_offset_c").alias("_sd")))
    tol = F.greatest(F.lit(DRIFT_K_SIGMA) * F.coalesce(F.col("_sd"), F.lit(0.0)), F.lit(DRIFT_MIN_TOLERANCE_C))
    return (d.join(base, "vessel_id", "left")
             .withColumn("temp_drift_flag", F.abs(F.col("temp_offset_c") - F.col("_cl")) > tol)
             .withColumn("temp_offset_c", F.round("temp_offset_c", 3))
             .drop("_cl", "_sd"))


def survey_line_quality(silver: DataFrame, lines: DataFrame) -> DataFrame:
    """Is each survey line good enough to deliver, or must it be re-run?"""
    ln = (lines.withColumn("planned_start", F.try_to_timestamp("planned_start", F.lit("yyyy-MM-dd'T'HH:mm:ss'Z'")))
               .withColumn("planned_end", F.try_to_timestamp("planned_end", F.lit("yyyy-MM-dd'T'HH:mm:ss'Z'")))
               .withColumn("expected_readings",
                           ((F.unix_timestamp("planned_end") - F.unix_timestamp("planned_start")) / 10).cast("int")))
    s = silver.select("vessel_id", "ts", "water_depth_m_valid", "qc_flags").alias("s")
    j = ln.alias("l").join(
        s, (F.col("s.vessel_id") == F.col("l.vessel_id"))
           & (F.col("s.ts") >= F.col("l.planned_start")) & (F.col("s.ts") < F.col("l.planned_end")), "left")
    agg = j.groupBy("l.line_id", "l.vessel_id", "l.project_code", "l.planned_start", "l.planned_end",
                    "l.expected_readings").agg(
        F.count("s.ts").alias("readings"),
        F.sum(F.col("s.water_depth_m_valid").cast("int")).alias("depth_valid_readings"))
    r = LINE_RULES
    out = (agg.withColumn("completeness_pct", F.round(F.col("readings") / F.col("expected_readings") * 100, 2))
              .withColumn("depth_valid_pct", F.round(F.col("depth_valid_readings") / F.col("readings") * 100, 2))
              .fillna({"depth_valid_pct": 0.0}))
    verdict = (F.when((F.col("completeness_pct") >= r["deliverable"]["completeness"])
                      & (F.col("depth_valid_pct") >= r["deliverable"]["depth_valid"]), F.lit("Deliverable"))
                .when((F.col("completeness_pct") >= r["review"]["completeness"])
                      & (F.col("depth_valid_pct") >= r["review"]["depth_valid"]), F.lit("Review"))
                .otherwise(F.lit("Re-run")))
    return out.withColumn("verdict", verdict)


def telemetry_minute(silver: DataFrame) -> DataFrame:
    """Fact table at 1-minute grain; averages use only readings that passed QC."""
    return (silver.withColumn("minute_ts", F.date_trunc("minute", "ts"))
            .groupBy("vessel_id", "obs_date", "minute_ts").agg(
                F.count("*").alias("readings"),
                F.avg("lat").alias("lat"), F.avg("lon").alias("lon"),
                F.avg("speed_ms").alias("speed_ms"),
                F.avg(_valid("water_depth_m")).alias("water_depth_m"),
                F.avg(_valid("sea_temp_c")).alias("sea_temp_c"),
                F.avg(_valid("roll_deg")).alias("roll_deg"),
                F.avg(_valid("pitch_deg")).alias("pitch_deg"),
                F.avg(_valid("engine_rpm")).alias("engine_rpm"),
                F.sum((~F.col("qc_ok")).cast("int")).alias("flagged_readings")))


def ml_features(silver: DataFrame) -> DataFrame:
    """10-minute windows of RAW signal statistics (QC flags deliberately not used),
    so the anomaly model is an independent second opinion to the rules."""
    w = silver.withColumn("window_start", F.window("ts", "10 minutes").start)
    # Window functions can't sit inside an aggregate, so compute the step first.
    step = F.abs(F.col("water_depth_m") - F.lag("water_depth_m").over(Window.partitionBy("vessel_id").orderBy("ts")))
    w = w.withColumn("depth_step", step)
    feats = w.groupBy("vessel_id", "obs_date", "window_start").agg(
        F.count("*").alias("n_readings"),
        F.coalesce(F.max("gap_before_s"), F.lit(10)).alias("max_gap_s"),
        F.stddev("water_depth_m").alias("depth_std"),
        F.max("depth_step").alias("depth_max_step"),
        F.stddev("sea_temp_c").alias("temp_std"),
        F.avg("sea_temp_c").alias("temp_mean"),
        F.max(F.abs("roll_deg")).alias("roll_abs_max"),
        F.stddev("roll_deg").alias("roll_std"),
        F.stddev("engine_rpm").alias("rpm_std"),
    )
    return feats.fillna(0.0)
