# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Gold: data products
# MAGIC | Table | Grain | Who uses it |
# MAGIC | --- | --- | --- |
# MAGIC | `gold_vessel_daily_health` | vessel × day | Operations: completeness, QC rate, sensor drift (SPC) |
# MAGIC | `gold_survey_line_quality` | survey line | Survey QC: Deliverable / Review / Re-run verdict |
# MAGIC | `gold_fact_telemetry_minute` | vessel × minute | BI and the digital twin: QC-passed values only |
# MAGIC | `gold_ml_features` | vessel × 10 min | Anomaly model input (raw-signal features) |

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))
from geodata import gold  # noqa: E402

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "marine_geodata")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
tbl = lambda name: f"{catalog}.{schema}.{name}"  # noqa: E731
spark.conf.set("spark.sql.session.timeZone", "UTC")

silver_df = spark.table(tbl("silver_telemetry"))
lines = spark.table(tbl("bronze_survey_lines"))

# COMMAND ----------

products = {
    "gold_vessel_daily_health": (gold.vessel_daily_health(silver_df),
                                 "Daily vessel data health: completeness, QC pass rate, fault counts, SPC temperature drift flag."),
    "gold_survey_line_quality": (gold.survey_line_quality(silver_df, lines),
                                 "Per survey line: completeness and valid depth share, with a delivery verdict."),
    "gold_fact_telemetry_minute": (gold.telemetry_minute(silver_df),
                                   "Fact table at 1-minute grain; sensor averages use QC-passed readings only."),
    "gold_ml_features": (gold.ml_features(silver_df),
                         "10-minute raw-signal features for the anomaly model."),
}
for name, (df, comment) in products.items():
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tbl(name))
    spark.sql(f"COMMENT ON TABLE {tbl(name)} IS '{comment}'")

# Liquid clustering on the largest table for fast vessel/date filtering.
spark.sql(f"ALTER TABLE {tbl('gold_fact_telemetry_minute')} CLUSTER BY (vessel_id, obs_date)")
spark.sql(f"OPTIMIZE {tbl('gold_fact_telemetry_minute')}")

# COMMAND ----------

display(spark.table(tbl("gold_vessel_daily_health"))
        .select("vessel_id", "obs_date", "completeness_pct", "qc_ok_pct", "depth_spikes",
                "temp_flatline_min", "gap_min", "temp_offset_c", "temp_drift_flag")
        .orderBy("vessel_id", "obs_date"))

# COMMAND ----------

display(spark.table(tbl("gold_survey_line_quality")).filter("verdict != 'Deliverable'"))
