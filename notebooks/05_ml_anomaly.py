# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Anomaly model (MLflow)
# MAGIC An Isolation Forest on 10-minute raw-signal features, as a second opinion to the QC rules.
# MAGIC * **Time-based split** (train on days 1–4, test on days 5–7) to avoid look-ahead leakage.
# MAGIC * Scored against the injected-fault log; in production that would be QC-analyst labels.
# MAGIC * Parameters, metrics and the model are tracked in MLflow.

# COMMAND ----------

# MAGIC %pip install -q scikit-learn mlflow
# MAGIC %restart_python

# COMMAND ----------

import os
import sys

sys.path.append(os.path.abspath("../src"))
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from geodata import ml  # noqa: E402

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "marine_geodata")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
tbl = lambda name: f"{catalog}.{schema}.{name}"  # noqa: E731
spark.conf.set("spark.sql.session.timeZone", "UTC")

feats = spark.table(tbl("gold_ml_features")).toPandas()
fault_log = pd.read_csv(f"/Volumes/{catalog}/{schema}/landing/raw/_truth/fault_log.csv")
labelled = ml.label_windows(feats, fault_log)

# COMMAND ----------

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/marine-geodata-anomaly")

with mlflow.start_run(run_name="isolation_forest") as run:
    params = {"contamination": 0.04, "n_estimators": 300, "train_until": "2026-09-05",
              "features": ",".join(ml.FEATURES)}
    model, scored, metrics = ml.train_and_score(labelled, train_until=params["train_until"],
                                                contamination=params["contamination"])
    mlflow.log_params(params)
    mlflow.log_metrics({k: v for k, v in metrics.items()})
    mlflow.sklearn.log_model(model, name="model", input_example=scored[ml.FEATURES].head(5))
    print(run.info.run_id, metrics)

# COMMAND ----------

(spark.createDataFrame(scored[["vessel_id", "window_start", "anomaly_score", "is_anomaly", "is_fault"]])
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tbl("gold_anomaly_scores")))
display(spark.table(tbl("gold_anomaly_scores")).orderBy("anomaly_score", ascending=False).limit(20))
