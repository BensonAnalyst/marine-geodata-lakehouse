# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Creates the Unity Catalog schema and the Volume that acts as the raw landing zone.
# MAGIC Every later notebook reads the same `catalog` and `schema` job parameters.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "marine_geodata")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema} "
          "COMMENT 'Marine geodata lakehouse demo: synthetic vessel telemetry, medallion pipeline'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.landing "
          "COMMENT 'Raw files as delivered by vessels (CSV), plus Auto Loader checkpoints'")

print(f"Ready: {catalog}.{schema}  ·  landing zone /Volumes/{catalog}/{schema}/landing")
