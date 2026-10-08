"""Bronze: land raw files as they arrive, with lineage columns.

On Databricks the notebook uses Auto Loader (cloudFiles) with
schemaEvolutionMode = "rescue": unexpected columns go into _rescued_data as
JSON instead of breaking the stream or being silently dropped.

`read_raw_csv_local` reproduces that behaviour with plain Spark so the same
downstream code can be tested without a Databricks workspace.
"""
from __future__ import annotations

import glob
import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import TELEMETRY_BASE_COLUMNS


def add_lineage(df: DataFrame, source_col: F.Column) -> DataFrame:
    """Every Bronze row records which file it came from and when it landed."""
    return (df.withColumn("_source_file", source_col)
              .withColumn("_ingest_time", F.current_timestamp()))


def rescue_extra_columns(df: DataFrame, base_cols: list[str]) -> DataFrame:
    """Pack columns outside the agreed schema into a _rescued_data JSON string."""
    extra = [c for c in df.columns if c not in base_cols and not c.startswith("_")]
    if extra:
        packed = F.when(
            F.coalesce(*[F.col(c).isNotNull() for c in extra]),
            F.to_json(F.struct(*[F.col(c) for c in extra])),
        )
        df = df.withColumn("_rescued_data", packed).drop(*extra)
    else:
        df = df.withColumn("_rescued_data", F.lit(None).cast("string"))
    return df


def read_raw_csv_local(spark: SparkSession, folder: str, base_cols: list[str] | None = None) -> DataFrame:
    """Local stand-in for Auto Loader: read every CSV as strings, keep lineage."""
    base_cols = base_cols or TELEMETRY_BASE_COLUMNS
    frames = []
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        df = spark.read.option("header", True).csv(path)  # all strings, like raw text
        df = df.withColumn("_source_file", F.lit(os.path.basename(path)))
        frames.append(df)
    out = frames[0]
    for df in frames[1:]:
        out = out.unionByName(df, allowMissingColumns=True)
    out = rescue_extra_columns(out, base_cols + ["_source_file"])
    return out.withColumn("_ingest_time", F.current_timestamp())
