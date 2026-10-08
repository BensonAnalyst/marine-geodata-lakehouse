"""Unit tests for the transformation logic. Run with: pytest -q"""
import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from geodata import bronze, silver  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    s = (SparkSession.builder.master("local[2]").appName("tests")
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.driver.host", "127.0.0.1")
         .config("spark.sql.shuffle.partitions", "2").getOrCreate())
    yield s
    s.stop()


def _series(spark, values, col="water_depth_m", vessel="V1", start=datetime(2026, 9, 1), step_s=10):
    rows = [(vessel, start + timedelta(seconds=i * step_s), float(v)) for i, v in enumerate(values)]
    return spark.createDataFrame(rows, ["vessel_id", "ts", col])


def _flag(df, name):
    return [r[0] for r in df.orderBy("ts").select(name).collect()]


def test_spike_is_flagged_and_neighbours_are_not(spark):
    vals = [30.0 + 0.1 * (i % 3) for i in range(60)]
    vals[40] = 75.0  # one spike
    df = silver.flag_spikes(silver.flag_range(_series(spark, vals)))
    flags = _flag(df, "_f_water_depth_m_spike")
    assert flags[40] is True
    assert sum(flags) == 1


def test_no_spike_judged_without_enough_history(spark):
    vals = [30.0, 30.1, 45.0] + [30.0] * 20
    df = silver.flag_spikes(silver.flag_range(_series(spark, vals)))
    assert _flag(df, "_f_water_depth_m_spike")[2] is False


def test_flatline_flags_the_whole_run_only(spark):
    vals = [29.0 + 0.01 * i for i in range(20)] + [29.5] * 70 + [29.6 + 0.01 * i for i in range(20)]
    df = silver.flag_flatline(_series(spark, vals, col="sea_temp_c"), {"sea_temp_c": {"min_rows": 60}})
    flags = _flag(df, "_f_sea_temp_c_flatline")
    assert sum(flags) == 70
    assert flags[19] is False and flags[20] is True and flags[90] is False


def test_gap_detected_on_first_reading_after_dropout(spark):
    base = datetime(2026, 9, 1)
    times = [base + timedelta(seconds=10 * i) for i in range(5)] + [base + timedelta(seconds=40 + 600)]
    df = spark.createDataFrame([("V1", t, 1.0) for t in times], ["vessel_id", "ts", "x"])
    flags = _flag(silver.flag_gaps(df), "_f_gap")
    assert flags == [False, False, False, False, False, True]


def test_range_rule_catches_sentinel(spark):
    df = silver.flag_range(_series(spark, [1.0, 999.0, -2.0], col="roll_deg"), {"roll_deg": (-45, 45)})
    assert _flag(df, "_f_roll_deg_range") == [False, True, False]


def test_deduplicate_keeps_latest_ingest(spark):
    t = datetime(2026, 9, 1)
    rows = [("V1", t, 1.0, "a.csv", datetime(2026, 9, 2)), ("V1", t, 2.0, "b.csv", datetime(2026, 9, 3))]
    df = spark.createDataFrame(rows, ["vessel_id", "ts", "x", "_source_file", "_ingest_time"])
    out = silver.deduplicate(df).collect()
    assert len(out) == 1 and out[0].x == 2.0


def test_standardise_converts_knots_and_quarantines_bad_timestamps(spark):
    cols = bronze.TELEMETRY_BASE_COLUMNS
    good = ["V1", "2026-09-01T00:00:10Z", "1.2", "103.9", "90", "10", "kn", "0", "0", "30", "29", "1200"]
    bad = ["V1", "NaT-ERR", "1.2", "103.9", "90", "2", "m/s", "0", "0", "30", "29", "1200"]
    df = spark.createDataFrame([good, bad], cols).withColumn("_rescued_data", F.lit('{"battery_pct":"80.5"}'))
    std = silver.standardise(df)
    ok, quarantine = silver.split_quarantine(std)
    row = ok.collect()[0]
    assert abs(row.speed_ms - 5.14444) < 1e-4
    assert row.battery_pct == 80.5
    assert quarantine.count() == 1


def test_rescue_packs_unexpected_columns(spark):
    df = spark.createDataFrame([("V1", "x", "77")], ["vessel_id", "ts", "battery_pct"])
    out = bronze.rescue_extra_columns(df, ["vessel_id", "ts"])
    assert "battery_pct" not in out.columns
    assert out.collect()[0]._rescued_data == '{"battery_pct":"77"}'


def test_scd2_creates_new_version_only_on_change(spark):
    rows = [("V1", "A", "2026-09-01"), ("V1", "A", "2026-09-02"), ("V1", "B", "2026-09-03"), ("V1", "B", "2026-09-04")]
    snaps = spark.createDataFrame(rows, ["vessel_id", "sounder", "snapshot_date"])
    out = silver.build_scd2(snaps, "vessel_id", ["sounder"]).orderBy("version").collect()
    assert [r.version for r in out] == [1, 2]
    assert str(out[0].valid_to) == "2026-09-02" and out[0].is_current is False
    assert out[1].valid_to is None and out[1].is_current is True
