-- For each index, check that the number of daily records matches the expected number of days.
SELECT 
    index_id, 
    MIN(date) AS start_date, 
    MAX(date) AS end_date,
    COUNT(*) AS total_records,
    (DATE_PART('day', MAX(date) - MIN(date)) + 1) AS expected_records
FROM raw_price_data
GROUP BY index_id;


-- For each (country, metric), check the number of records matches the expected number of months.
SELECT 
    country, 
    metric, 
    MIN(date) AS start_date, 
    MAX(date) AS end_date,
    COUNT(*) AS total_records,
    ((DATE_PART('year', MAX(date)) - DATE_PART('year', MIN(date))) * 12 
      + (DATE_PART('month', MAX(date)) - DATE_PART('month', MIN(date))) + 1) AS expected_records
FROM macro_indicators
GROUP BY country, metric;



SELECT 
    metric, 
    MIN(value) AS min_value, 
    MAX(value) AS max_value, 
    AVG(value) AS avg_value
FROM macro_indicators
GROUP BY metric;
