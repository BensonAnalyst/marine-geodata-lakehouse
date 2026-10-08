"""Marine geodata lakehouse: reusable pipeline logic.

All functions take and return Spark DataFrames (or pandas for the data
generator) so the same code runs in Databricks notebooks and in local tests.
"""

__version__ = "1.0.0"
