"""Run the whole pipeline locally with plain Spark (Parquet stands in for Delta).

    python scripts/run_local.py

This proves the transformation logic end to end without a workspace. On
Databricks, the notebooks in /notebooks call the same functions and write
Delta tables in Unity Catalog instead.
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from sklearn.metrics import precision_score, recall_score  # noqa: E402

from geodata import bronze, generator, gold, ml, quality, silver  # noqa: E402

LAKE = os.path.join(ROOT, "local_lake")
RAW = os.path.join(LAKE, "raw")


def main():
    shutil.rmtree(LAKE, ignore_errors=True)
    print("1/6 generate raw files:", generator.generate(RAW))

    spark = (SparkSession.builder.master("local[4]").appName("geodata-local")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.driver.host", "127.0.0.1")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    def save(df, name):
        path = os.path.join(LAKE, name)
        df.write.mode("overwrite").parquet(path)
        return spark.read.parquet(path)

    b = save(bronze.read_raw_csv_local(spark, os.path.join(RAW, "telemetry")), "bronze_telemetry")
    print(f"2/6 bronze rows: {b.count():,}  files: {b.select('_source_file').distinct().count()}  "
          f"rows with rescued fields: {b.filter('_rescued_data IS NOT NULL').count():,}")

    s, q = silver.build_silver(b)
    s, q = save(s, "silver_telemetry"), save(q, "quarantine_telemetry")
    cfg = spark.read.option("header", True).csv(os.path.join(RAW, "vessel_config"))
    dim = save(silver.build_scd2(cfg, "vessel_id", ["vessel_type", "echo_sounder", "firmware", "motion_sensor"]),
               "dim_vessel")
    print(f"3/6 silver rows: {s.count():,}  quarantined: {q.count():,}  "
          f"duplicates removed: {b.count() - q.count() - s.count():,}")
    dim.orderBy("vessel_id", "version").show(truncate=False)

    lines = spark.read.option("header", True).csv(os.path.join(RAW, "survey_lines"))
    health = save(gold.vessel_daily_health(s), "gold_vessel_daily_health")
    lq = save(gold.survey_line_quality(s, lines), "gold_survey_line_quality")
    save(gold.telemetry_minute(s), "gold_telemetry_minute")
    feats = save(gold.ml_features(s), "gold_ml_features")
    print("4/6 gold vessel_daily_health:")
    health.select("vessel_id", "obs_date", "completeness_pct", "qc_ok_pct", "depth_spikes", "roll_out_of_range",
                  "temp_flatline_min", "gap_min", "temp_offset_c", "temp_drift_flag") \
          .orderBy("vessel_id", "obs_date").show(30)
    print("   survey line verdicts:")
    lq.groupBy("verdict").count().show()
    lq.filter("verdict != 'Deliverable'").select("line_id", "completeness_pct", "depth_valid_pct", "verdict").show()

    fault_log = pd.read_csv(os.path.join(RAW, "_truth", "fault_log.csv"))
    recall = quality.qc_recall(s, fault_log)
    drift = quality.drift_detection(health, fault_log)
    print("5/6 QC recall vs injected faults:\n", recall.to_string(index=False))
    print("   drift detection:\n", drift.to_string(index=False))

    fp = feats.toPandas()
    labelled = ml.label_windows(fp, fault_log)
    _, scored, metrics = ml.train_and_score(labelled, train_until="2026-09-05")
    # Rules baseline at the same window grain: window flagged if any reading has a flag.
    rules_win = (s.withColumn("window_start", F.window("ts", "10 minutes").start)
                   .groupBy("vessel_id", "window_start").agg(F.min("qc_ok").alias("all_ok")).toPandas())
    rules_win["rules_flag"] = ~rules_win["all_ok"]
    m = scored.merge(rules_win[["vessel_id", "window_start", "rules_flag"]], on=["vessel_id", "window_start"], how="left")
    metrics["rules_precision"] = float(precision_score(m.is_fault, m.rules_flag.fillna(False)))
    metrics["rules_recall"] = float(recall_score(m.is_fault, m.rules_flag.fillna(False)))
    metrics["model_only_catches"] = int((m.is_fault & m.is_anomaly & ~m.rules_flag.fillna(False)).sum())
    print("6/6 ML anomaly model (test days):", json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                                           for k, v in metrics.items()}))

    dq = quality.run_dq_checks(s, health, recall)
    print("DQ gates:\n", dq.to_string(index=False))
    quality.enforce(dq)
    print("\nAll critical data-quality gates passed.")
    spark.stop()


if __name__ == "__main__":
    main()
