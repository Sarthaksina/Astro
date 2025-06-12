from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.models import Variable
from src.astro_engine.planetary_positions import PlanetaryCalculator  # Import from src.astro_engine.planetary_positions
# Import project modules
import sys
import os
sys.path.append('/app')

# Default arguments for DAG
default_args = {
    'owner': 'cosmic_market_oracle',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'historical_data_pipeline',
    default_args=default_args,
    description='Pipeline for processing historical market and astrological data',
    schedule_interval='@weekly',  # Run weekly
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['market_data', 'astrology'],
)

# Define functions for tasks
def fetch_market_data(**kwargs):
    """
    Fetch historical market data from sources and store in TimescaleDB.
    """
    from src.data_acquisition.financial_data import YahooFinanceDataSource # Added
    logger = kwargs['ti'].log # Added Airflow task logger
    
    # Get parameters
    start_date = kwargs.get('start_date', '1800-01-01')
    end_date = kwargs.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    source = kwargs.get('source', 'yahoo')
    
    symbol_to_fetch = kwargs.get('dag_run').conf.get('symbol', '^DJI') # Get symbol from DAG run config or default
    logger.info(f"Attempting to fetch market data for symbol: {symbol_to_fetch} from source: {source} for dates: {start_date} to {end_date}")

    if source.lower() == 'yahoo':
        data_source = YahooFinanceDataSource()
        data = data_source.fetch_data(
            symbol=symbol_to_fetch,
            start_date=start_date,
            end_date=end_date
            # Assuming default interval '1d' is acceptable
        )
        logger.info(f"Successfully fetched {len(data)} records for {symbol_to_fetch} from Yahoo Finance.")
    else:
        logger.error(f"Unsupported data source specified: {source}")
        raise ValueError(f"Data source '{source}' is not supported by this DAG's fetch_market_data task.")
    
    # Return data for next task
    return data
def calculate_planetary_positions(**kwargs):
    """
    Calculate planetary positions for the given date range.
    Uses the consolidated implementation from data_integration.airflow_dags.
    """
    # Import the consolidated function to avoid code duplication
    from src.data_integration.airflow_dags import calculate_planetary_positions as calc_positions
    
    # Get parameters
    start_date = kwargs.get('start_date', '1800-01-01')
    end_date = kwargs.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    # Call the consolidated function
    return calc_positions(start_date=start_date, end_date=end_date, **kwargs)


def integrate_data(**kwargs):
    """
    Integrate market data with planetary positions.
    """
    from src.data_processing.integrator import integrate_market_astro_data
    
    # Get task instances
    ti = kwargs['ti']
    
    # Pull data from previous tasks
    market_data = ti.xcom_pull(task_ids='fetch_market_data')
    planetary_data = ti.xcom_pull(task_ids='calculate_planetary_positions')
    
    # Integrate data
    integrated_data = integrate_market_astro_data(market_data, planetary_data)
    
    # Return integrated data
    return integrated_data

def generate_features(**kwargs):
    """
    Generate features from integrated data.
    """
    from src.feature_engineering.feature_generator import generate_features
    
    # Get task instance
    ti = kwargs['ti']
    
    # Pull data from previous task
    integrated_data = ti.xcom_pull(task_ids='integrate_data')
    
    # Generate features
    features = generate_features(integrated_data)
    
    # Return features
    return features

def store_features(**kwargs):
    """
    Store generated features in TimescaleDB.
    """
    from src.data_processing.db_writer import write_features_to_db
    
    # Get task instance
    ti = kwargs['ti']
    
    # Pull features from previous task
    features = ti.xcom_pull(task_ids='generate_features')
    
    # Store features
    write_features_to_db(features)
    
    return "Features stored successfully"

# Create tasks
fetch_market_data_task = PythonOperator(
    task_id='fetch_market_data',
    python_callable=fetch_market_data,
    provide_context=True,
    op_kwargs={
        'start_date': '{{ var.value.start_date }}',
        'end_date': '{{ var.value.end_date }}',
        'source': '{{ var.value.data_source }}',
    },
    dag=dag,
)

calculate_planetary_positions_task = PythonOperator(
    task_id='calculate_planetary_positions',
    python_callable=calculate_planetary_positions,
    provide_context=True,
    op_kwargs={
        'start_date': '{{ var.value.start_date }}',
        'end_date': '{{ var.value.end_date }}',
    },
    dag=dag,
)

integrate_data_task = PythonOperator(
    task_id='integrate_data',
    python_callable=integrate_data,
    provide_context=True,
    dag=dag,
)

generate_features_task = PythonOperator(
    task_id='generate_features',
    python_callable=generate_features,
    provide_context=True,
    dag=dag,
)

store_features_task = PythonOperator(
    task_id='store_features',
    python_callable=store_features,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
fetch_market_data_task >> integrate_data_task
calculate_planetary_positions_task >> integrate_data_task
integrate_data_task >> generate_features_task >> store_features_task