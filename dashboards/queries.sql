-- Queries for a Databricks AI/BI dashboard or Power BI (via the Databricks SQL warehouse connector).
-- Replace workspace.marine_geodata with your catalog.schema.

-- 1. Fleet data health today vs. last 7 days (KPI tiles)
SELECT obs_date,
       round(avg(completeness_pct), 2) AS fleet_completeness_pct,
       round(avg(qc_ok_pct), 2)        AS fleet_qc_ok_pct,
       sum(depth_spikes)               AS depth_spikes,
       count_if(temp_drift_flag)       AS vessels_with_temp_drift
FROM workspace.marine_geodata.gold_vessel_daily_health
GROUP BY obs_date
ORDER BY obs_date;

-- 2. Exceptions an operations lead must act on today
SELECT vessel_id, obs_date, completeness_pct, gap_min, temp_flatline_min, temp_offset_c,
       CASE WHEN temp_drift_flag THEN 'Recalibrate temperature sensor'
            WHEN temp_flatline_min > 0 THEN 'Check temperature sensor (stuck)'
            WHEN gap_min > 15 THEN 'Investigate comms dropout'
       END AS action
FROM workspace.marine_geodata.gold_vessel_daily_health
WHERE temp_drift_flag OR temp_flatline_min > 0 OR gap_min > 15
ORDER BY obs_date DESC, vessel_id;

-- 3. Survey lines that are not deliverable (re-run while the vessel is still on site)
SELECT line_id, vessel_id, project_code, planned_start, completeness_pct, depth_valid_pct, verdict
FROM workspace.marine_geodata.gold_survey_line_quality
WHERE verdict <> 'Deliverable'
ORDER BY planned_start;

-- 4. Vessel track and depth at 1-minute grain (map / line chart), configuration in force at the time
SELECT f.vessel_id, f.minute_ts, f.lat, f.lon, f.water_depth_m, d.echo_sounder, d.firmware
FROM workspace.marine_geodata.gold_fact_telemetry_minute f
JOIN workspace.marine_geodata.dim_vessel d
  ON f.vessel_id = d.vessel_id
 AND f.obs_date BETWEEN d.valid_from AND coalesce(d.valid_to, DATE'9999-12-31')
WHERE f.vessel_id = 'USV-02';

-- 5. Data-quality gate history (did last night's run pass?)
SELECT run_time, check, severity, observed, threshold, passed
FROM workspace.marine_geodata.dq_results
ORDER BY run_time DESC;
