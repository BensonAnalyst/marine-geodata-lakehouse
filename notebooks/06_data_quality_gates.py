# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Data-quality gates
# MAGIC Scores the Silver QC against the known fault log and enforces gates.
# MAGIC Results are appended to `dq_results` (a run history), and the **job fails** if a
# MAGIC CRITICAL gate fails, so consumers never get silently bad data.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))
import pandas as pd  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from geodata import quality  # noqa: E402

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "marine_geodata")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
tbl = lambda name: f"{catalog}.{schema}.{name}"  # noqa: E731
spark.conf.set("spark.sql.session.timeZone", "UTC")

silver_df = spark.table(tbl("silver_telemetry"))
health = spark.table(tbl("gold_vessel_daily_health"))
fault_log = pd.read_csv(f"/Volumes/{catalog}/{schema}/landing/raw/_truth/fault_log.csv")

# COMMAND ----------

recall = quality.qc_recall(silver_df, fault_log)
drift = quality.drift_detection(health, fault_log)
display(recall)
display(drift)

# COMMAND ----------

results = quality.run_dq_checks(silver_df, health, recall)
results["observed"] = results["observed"].astype(float)
(spark.createDataFrame(results)
      .withColumn("run_time", F.current_timestamp())
      .write.mode("append").option("mergeSchema", "true").saveAsTable(tbl("dq_results")))
display(results)

quality.enforce(results)  # raises -> task fails -> job alerts
