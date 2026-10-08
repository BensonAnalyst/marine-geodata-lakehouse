# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Generate synthetic raw files
# MAGIC Simulates 7 days of 10-second telemetry from 3 fictional vessels, delivered as one CSV per
# MAGIC vessel per day, with **documented faults** injected: echo-sounder spikes, a stuck temperature
# MAGIC sensor, a motion-sensor sentinel value, a comms dropout, a re-sent file, malformed rows,
# MAGIC a calibration drift and new firmware adding a column (schema drift).
# MAGIC
# MAGIC Every fault is written to `_truth/fault_log.csv`, used only to score the QC later.

# COMMAND ----------

import os
import shutil
import sys

sys.path.append(os.path.abspath("../src"))
from geodata import generator  # noqa: E402

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "marine_geodata")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
landing = f"/Volumes/{catalog}/{schema}/landing"

# COMMAND ----------

# Fresh demo run: clear previous raw files and stream checkpoints.
for sub in ("raw", "_checkpoints"):
    shutil.rmtree(os.path.join(landing, sub), ignore_errors=True)

summary = generator.generate(os.path.join(landing, "raw"))
summary

# COMMAND ----------

display(dbutils.fs.ls(f"{landing}/raw/telemetry"))
