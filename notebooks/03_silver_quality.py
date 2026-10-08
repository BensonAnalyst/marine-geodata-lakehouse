# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Silver: standardise, de-duplicate, quarantine, quality-flag
# MAGIC * Types cast, timestamps in UTC, speed converted to m/s whatever unit the logger used.
# MAGIC * Rows that can't be placed in time go to **quarantine**, never silently dropped.
# MAGIC * Duplicate readings from re-sent files are removed (latest ingest wins).
# MAGIC * QC rules (from `src/geodata/config.py`) flag **range violations, spikes, flatlines, gaps**.
# MAGIC   Readings are kept with a `qc_flags` array, so every exclusion downstream is explainable.
# MAGIC * Written with **MERGE** on (vessel_id, ts), so re-running is idempotent.
# MAGIC * Vessel configuration snapshots become an **SCD Type 2** dimension.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))
from delta.tables import DeltaTable  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from geodata import silver  # noqa: E402

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "marine_geodata")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
tbl = lambda name: f"{catalog}.{schema}.{name}"  # noqa: E731
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

bronze = spark.table(tbl("bronze_telemetry"))
silver_df, quarantine_df = silver.build_silver(bronze)

# COMMAND ----------

# Upsert Silver. First run creates the table; later runs MERGE on the business key.
target = tbl("silver_telemetry")
if not spark.catalog.tableExists(target):
    silver_df.write.saveAsTable(target)
    spark.sql(f"ALTER TABLE {target} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
else:
    (DeltaTable.forName(spark, target).alias("t")
        .merge(silver_df.alias("s"), "t.vessel_id = s.vessel_id AND t.ts = s.ts")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute())

quarantine_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tbl("silver_quarantine"))

# COMMAND ----------

# Governance: describe the table and its key columns in Unity Catalog.
spark.sql(f"COMMENT ON TABLE {target} IS 'Quality-flagged vessel telemetry, one row per vessel per timestamp (UTC).'")
for col, text in {
    "qc_flags": "QC rules this reading failed: *_range, *_spike, *_flatline, gap. Empty = clean.",
    "qc_ok": "True when the reading failed no QC rule.",
    "water_depth_m_valid": "Depth usable for products (no range, spike or flatline flag).",
    "speed_ms": "Speed over ground in m/s, converted from knots where the logger reported kn.",
    "battery_pct": "USV battery level; new firmware field recovered from Bronze _rescued_data.",
}.items():
    spark.sql(f"ALTER TABLE {target} ALTER COLUMN {col} COMMENT '{text}'")

# COMMAND ----------

# SCD Type 2 vessel dimension from daily configuration snapshots.
cfg = spark.table(tbl("bronze_vessel_config"))
dim = silver.build_scd2(cfg, "vessel_id", ["vessel_type", "echo_sounder", "firmware", "motion_sensor"])
dim.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tbl("dim_vessel"))
display(spark.table(tbl("dim_vessel")).orderBy("vessel_id", "version"))

# COMMAND ----------

display(spark.sql(f"""
  SELECT flag, count(*) AS readings
  FROM {target} LATERAL VIEW explode(qc_flags) f AS flag
  GROUP BY flag ORDER BY readings DESC"""))
