-- ClickHouse discovery and extraction-reference queries for the offline research
-- project. These are read-only and used for one-time raw Parquet refreshes.

-- List available databases.
SELECT name
FROM system.databases
ORDER BY name;

-- List non-system tables.
SELECT
    database,
    name,
    engine
FROM system.tables
WHERE database NOT IN ('system', 'information_schema', 'INFORMATION_SCHEMA')
ORDER BY database, name;

-- Inspect schema contract.
DESCRIBE TABLE firstrate.stocks;

-- Check symbol coverage and history depth.
SELECT
    symbol,
    count() AS row_count,
    min(ts) AS min_ts,
    max(ts) AS max_ts
FROM firstrate.stocks
GROUP BY symbol
ORDER BY row_count DESC
LIMIT 25;

-- Check regular-hours row counts by day for one symbol.
SELECT
    toDate(ts) AS trade_date,
    count() AS rows
FROM firstrate.stocks
WHERE
    symbol = 'AAPL'
    AND toTime(ts) >= toTime('14:30:00')
    AND toTime(ts) <= toTime('20:00:00')
GROUP BY trade_date
ORDER BY trade_date DESC
LIMIT 30;

-- Check bar spacing inside a regular-hours filtered day.
WITH ordered AS (
    SELECT
        ts,
        lagInFrame(ts) OVER (ORDER BY ts) AS prev_ts
    FROM firstrate.stocks
    WHERE
        symbol = 'AAPL'
        AND toDate(ts) = toDate('2026-01-05')
        AND toTime(ts) >= toTime('14:30:00')
        AND toTime(ts) <= toTime('20:00:00')
)
SELECT
    dateDiff('minute', prev_ts, ts) AS gap_minutes,
    count() AS rows
FROM ordered
WHERE prev_ts IS NOT NULL
GROUP BY gap_minutes
ORDER BY gap_minutes;

-- Extraction template for one symbol with explicit regular-hours session filter.
SELECT
    symbol,
    ts,
    open,
    high,
    low,
    close,
    volume
FROM firstrate.stocks
WHERE
    symbol = 'AAPL'
    AND toDate(ts) BETWEEN toDate('2025-11-03') AND toDate('2026-01-16')
    AND toTime(ts) >= toTime('14:30:00')
    AND toTime(ts) <= toTime('20:00:00')
ORDER BY ts;

-- Session-boundary sanity check for extracted slice.
SELECT
    min(toTime(ts)) AS min_time,
    max(toTime(ts)) AS max_time,
    uniqExact(toDate(ts)) AS trade_days,
    min(ts) AS first_ts,
    max(ts) AS last_ts
FROM firstrate.stocks
WHERE
    symbol = 'AAPL'
    AND toDate(ts) BETWEEN toDate('2025-11-03') AND toDate('2026-01-16')
    AND toTime(ts) >= toTime('14:30:00')
    AND toTime(ts) <= toTime('20:00:00');
