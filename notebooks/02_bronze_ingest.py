# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Bronze: ingest raw files with Auto Loader
# MAGIC * **Incremental:** Auto Loader tracks which files it has seen; re-running only picks up new files.
# MAGIC * **Schema drift is rescued, not lost:** the agreed columns are declared; anything new (e.g. a
# MAGIC   firmware update adding `battery_pct`) lands in `_rescued_data` as JSON.
# MAGIC * **Lineage:** every row keeps its source file path and ingest time.

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import StringType, StructField, StructType  # noqa: E402

from geodata.config import TELEMETRY_BASE_COLUMNS  # noqa: E402

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "marine_geodata")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
landing = f"/Volumes/{catalog}/{schema}/landing"
tbl = lambda name: f"{catalog}.{schema}.{name}"  # noqa: E731

# COMMAND ----------

# Raw is raw: read every agreed column as text; typing happens in Silver.
raw_schema = StructType([StructField(c, StringType()) for c in TELEMETRY_BASE_COLUMNS])

query = (spark.readStream.format("cloudFiles")
         .option("cloudFiles.format", "csv")
         .option("header", "true")
         .option("rescuedDataColumn", "_rescued_data")
         .option("cloudFiles.schemaLocation", f"{landing}/_checkpoints/bronze_telemetry/schema")
         .schema(raw_schema)
         .load(f"{landing}/raw/telemetry")
         .withColumn("_source_file", F.col("_metadata.file_path"))
         .withColumn("_ingest_time", F.current_timestamp())
         .writeStream
         .option("checkpointLocation", f"{landing}/_checkpoints/bronze_telemetry")
         .trigger(availableNow=True)
         .toTable(tbl("bronze_telemetry")))
query.awaitTermination()

# COMMAND ----------

# Small reference files: simple batch loads.
for name in ("vessel_config", "survey_lines"):
    (spark.read.option("header", True).csv(f"{landing}/raw/{name}")
         .withColumn("_source_file", F.col("_metadata.file_path"))
         .withColumn("_ingest_time", F.current_timestamp())
         .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tbl(f"bronze_{name}")))

spark.sql(f"COMMENT ON TABLE {tbl('bronze_telemetry')} IS "
          "'Raw vessel telemetry exactly as delivered. Unexpected columns in _rescued_data.'")

# COMMAND ----------

display(spark.sql(f"""
  SELECT count(*) AS rows,
         count(DISTINCT _source_file) AS files,
         count_if(_rescued_data IS NOT NULL) AS rows_with_rescued_fields
  FROM {tbl('bronze_telemetry')}"""))
