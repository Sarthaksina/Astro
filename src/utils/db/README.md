# TimescaleDB Connector for Cosmic Market Oracle

This module provides a comprehensive connector for TimescaleDB, optimized for time-series data management in the Cosmic Market Oracle project. It enables efficient storage, querying, and analysis of market data alongside astrological information.

## Features

### Core Functionality
- Database connection management with both SQLAlchemy and psycopg2
- Hypertable creation and management
- Efficient data insertion with conflict handling
- Optimized time-series queries with bucketing and aggregation

### Advanced TimescaleDB Features
- **Continuous Aggregates**: Pre-computed aggregations for faster queries
- **Native Compression**: Automatic data compression for efficient storage
- **Retention Policies**: Automated data lifecycle management
- **Chunk Interval Optimization**: Performance tuning for different data patterns

### Specialized for Cosmic Market Oracle
- Integrated market and astrological data management
- Synchronized views for cross-domain analysis
- Correlation analysis between market movements and planetary positions
- Complete environment setup with a single method call

## Usage Examples

### Basic Setup

```python
from src.utils.db.timescale_connector import get_timescale_connector

# Get connector with default settings
connector = get_timescale_connector()
connector.connect()

# Set up complete environment with all tables and optimizations
connector.setup_complete_timescale_environment()

# Always disconnect when done
connector.disconnect()
```

### Data Insertion

```python
import pandas as pd

# Insert market data
market_data = pd.DataFrame({
    "timestamp": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
    "symbol": ["DJI", "DJI"],
    "open": [33000.0, 33100.0],
    "high": [33200.0, 33300.0],
    "low": [32900.0, 33000.0],
    "close": [33100.0, 33200.0],
    "volume": [1000000, 1100000],
    "source": ["example", "example"]
})

connector.insert_market_data(market_data)
```

### Advanced Features

```python
# Enable compression for efficient storage
connector.enable_compression("market_data", compress_after="7 days")

# Set retention policy to automatically drop old data
connector.set_retention_policy("market_data", drop_after="2 years")

# Create synchronized view for integrated analysis
connector.create_synchronized_view("market_planetary_daily", interval="1 day")

# Perform correlation analysis
connector.create_correlation_analysis(
    output_table="market_planetary_correlations",
    market_symbol="DJI",
    planet="Jupiter",
    window="90 days"
)
```

### Querying Data

```python
from datetime import datetime, timedelta

# Query market data
start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now()
market_data = connector.query_market_data("DJI", start_date, end_date)

# Query planetary positions
planetary_data = connector.query_planetary_positions(
    ["Jupiter", "Saturn", "Mars"], 
    start_date, 
    end_date
)

# Query features
feature_data = connector.query_features(
    ["jupiter_saturn_angle", "moon_phase"], 
    start_date, 
    end_date, 
    category="planetary_aspect"
)
```

## Continuous Aggregates

Continuous aggregates are a powerful TimescaleDB feature that pre-computes and stores aggregated data for faster query performance. The connector provides methods to create and manage these aggregates.

### Creating Custom Continuous Aggregates

```python
# Create a custom continuous aggregate for market volatility
connector.create_continuous_aggregate(
    view_name="market_volatility_weekly",
    source_table="market_data",
    time_bucket="1 week",
    group_columns=["symbol"],
    aggregates={
        "high": "MAX",
        "low": "MIN",
        "close": "LAST"
    },
    # Automatically refresh data older than 1 day and newer than 1 hour
    start_offset="1 day",
    end_offset="1 hour"
)

# Query the continuous aggregate
query = """
    SELECT 
        time, symbol, 
        (high_max - low_min) / close_last AS volatility_ratio
    FROM market_volatility_weekly
    WHERE symbol = 'DJI'
    ORDER BY time DESC
    LIMIT 10
"""
volatility_data = pd.read_sql_query(query, connector.engine)
```

### Default Continuous Aggregates

The connector automatically creates several useful continuous aggregates when you call `setup_complete_timescale_environment()`:

- `market_data_daily`: Daily OHLCV aggregation by symbol
- `planetary_positions_daily`: Daily planetary position snapshots
- `features_daily`: Daily average of calculated features

## Compression Strategies

TimescaleDB's native compression can significantly reduce storage requirements for historical data. The connector provides methods to configure compression based on your specific needs.

### Compression Configuration

```python
# Basic compression setup
connector.enable_compression("market_data", compress_after="7 days")

# More aggressive compression for rarely accessed data
connector.enable_compression("planetary_positions", compress_after="3 days")

# Custom compression with specific columns
with connector.conn.cursor() as cur:
    cur.execute("""
        ALTER TABLE features SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'feature_name,feature_category'
        )
    """)
    connector.conn.commit()
```

### Compression Monitoring

```python
# Check compression statistics
query = """
    SELECT 
        hypertable_name, 
        compression_status,
        pg_size_pretty(before_compression_total_bytes) as before_size,
        pg_size_pretty(after_compression_total_bytes) as after_size,
        round(100 * (before_compression_total_bytes - after_compression_total_bytes) / 
              nullif(before_compression_total_bytes, 0), 2) as compression_ratio
    FROM timescaledb_information.compression_stats
"""
compression_stats = pd.read_sql_query(query, connector.engine)
```

## Chunk Management and Optimization

TimescaleDB stores data in chunks based on time intervals. Optimizing chunk size is crucial for performance.

### Chunk Interval Optimization

```python
# Set chunk interval based on data patterns
# Smaller intervals for high-frequency data
connector.optimize_chunk_time_interval("market_data", new_interval="1 day")

# Larger intervals for less frequent data
connector.optimize_chunk_time_interval("planetary_positions", new_interval="7 days")
```

### Chunk Management Guidelines

- **High-frequency data**: Use smaller chunks (hours to days)
- **Daily data**: Use 7-30 day chunks
- **Monthly data**: Use 3-6 month chunks
- **Target chunk size**: Aim for chunks between 100MB-1GB for optimal performance
- **Query patterns**: Align chunk intervals with common query time ranges

## Integrated Analysis with Synchronized Views

The connector provides specialized views that synchronize market and astrological data for integrated analysis.

### Creating Synchronized Views

```python
# Create a daily synchronized view
connector.create_synchronized_view("market_planetary_daily", interval="1 day")

# Create a comprehensive view with market data, planetary positions, and features
connector.create_market_planetary_features_view("cosmic_market_integrated", interval="1 day")
```

### Querying Synchronized Views

```python
# Find correlations between planetary positions and market movements
query = """
    SELECT 
        time, symbol, 
        planet, longitude, is_retrograde,
        close, 
        (close - LAG(close) OVER (PARTITION BY symbol ORDER BY time)) / LAG(close) OVER (PARTITION BY symbol ORDER BY time) * 100 AS daily_return
    FROM market_planetary_daily
    WHERE 
        symbol = 'DJI' AND 
        planet IN ('Jupiter', 'Saturn') AND
        time >= '2023-01-01' AND 
        time <= '2023-03-31'
    ORDER BY time, planet
"""
integrated_data = pd.read_sql_query(query, connector.engine)
```

## Performance Tuning Recommendations

### Query Optimization

- **Use time bucketing**: Always use `time_bucket()` for time-series aggregation
- **Limit time ranges**: Always include time constraints in your queries
- **Use continuous aggregates**: Query pre-aggregated data when possible
- **Leverage materialized views**: For complex, frequently-used queries

### Index Optimization

```python
# Create additional indexes for common query patterns
with connector.conn.cursor() as cur:
    # Index for querying specific planets
    cur.execute("CREATE INDEX IF NOT EXISTS idx_planetary_positions_planet ON planetary_positions(planet)")
    
    # Index for feature categories
    cur.execute("CREATE INDEX IF NOT EXISTS idx_features_category ON features(feature_category)")
    
    connector.conn.commit()
```

### Resource Management

- **Connection pooling**: Use SQLAlchemy's connection pooling for multi-user environments
- **Statement timeout**: Set timeouts to prevent long-running queries from consuming resources
- **Memory settings**: Adjust `work_mem` and `maintenance_work_mem` based on available system memory

## Configuration

The connector can be configured using environment variables or a configuration file:

```python
# From environment variables (default)
connector = get_timescale_connector(env_prefix="TIMESCALE")

# From configuration file
connector = get_timescale_connector(config_file="/path/to/config.yaml")
```

Environment variables should be prefixed with the specified prefix (default: `TIMESCALE`):
- `TIMESCALE_HOST`: Database host (default: "localhost")
- `TIMESCALE_PORT`: Database port (default: 5432)
- `TIMESCALE_DATABASE`: Database name (default: "cosmic_market_oracle")
- `TIMESCALE_USER`: Database user (default: "cosmic")
- `TIMESCALE_PASSWORD`: Database password (default: "cosmic_password")
- `TIMESCALE_SCHEMA`: Database schema (default: "public")

## Best Practices for Astronomical-Financial Data

### Time Alignment Strategies

- **Consistent timestamps**: Ensure market and planetary data use consistent timezone handling
- **Bucketing alignment**: Use the same time buckets when joining different data types
- **Ephemeris precision**: For high-precision astronomical calculations, store intermediate positions at higher frequencies

### Data Lifecycle Management

- **Hot data**: Recent data (last 30-90 days) - no compression, optimized for fast queries
- **Warm data**: Medium-term data (90 days - 2 years) - compressed but retained in main tables
- **Cold data**: Historical data (> 2 years) - heavily compressed or moved to separate tables

### Correlation Analysis Optimization

```python
# Create a specialized table for correlation analysis
connector.create_correlation_analysis(
    output_table="jupiter_market_correlations",
    market_symbol="DJI",
    planet="Jupiter",
    window="90 days",
    correlation_method="pearson"  # or "spearman"
)

# Query correlation results
query = "SELECT * FROM jupiter_market_correlations ORDER BY time DESC LIMIT 10"
correlation_data = pd.read_sql_query(query, connector.engine)
```

## Troubleshooting

### Common Issues

- **Connection failures**: Check network settings, credentials, and database existence
- **Slow queries**: Examine query plans with `EXPLAIN ANALYZE`, check for missing indexes
- **High disk usage**: Review compression settings and retention policies
- **Memory errors**: Adjust TimescaleDB memory parameters or reduce chunk size

### Logging and Monitoring

```python
# Enable detailed logging
import logging
logging.getLogger("timescale_connector").setLevel(logging.DEBUG)

# Monitor database size
query = """
    SELECT 
        hypertable_name, 
        pg_size_pretty(hypertable_size) as size
    FROM timescaledb_information.hypertable_size
    ORDER BY hypertable_size DESC
"""
size_data = pd.read_sql_query(query, connector.engine)
```