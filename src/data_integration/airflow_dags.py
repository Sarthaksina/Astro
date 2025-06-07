"""
Apache Airflow DAGs for the Cosmic Market Oracle.

This module defines Airflow DAGs (Directed Acyclic Graphs) for orchestrating
data pipelines that integrate financial and astrological data.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple

# Airflow imports
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago

# Project imports
from src.data_acquisition.financial_data import YahooFinanceDataSource, BloombergDataSource
from src.astro_engine.planetary_positions import PlanetaryCalculator
from src.astro_engine.vedic_analysis import VedicAnalyzer
from src.data_processing.market_regime import MarketRegimeLabeler
from src.utils.db.timescale_connector import TimescaleConnector
from src.utils.logging_config import setup_logging
from config.db_config import get_db_params

# Set up logging
logger = setup_logging(__name__)

# Get database parameters
DB_PARAMS = get_db_params()

# Define default arguments for DAGs
default_args = {
    'owner': 'cosmic_oracle',
    'depends_on_past': False,
    'email': ['admin@cosmicmarketoracle.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# Database connection parameters
DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': 'postgres',
    'database': 'cosmic_oracle',
    'schema': 'cosmic_data'
}


def fetch_financial_data(symbol: str, start_date: str, end_date: str, **kwargs) -> None:
    """
    Fetch financial data from Yahoo Finance and store in TimescaleDB.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    logger.info(f"Fetching financial data for {symbol} from {start_date} to {end_date}")
    
    # Create Yahoo Finance connector
    yahoo_source = YahooFinanceDataSource()
    
    # Fetch data
    data = yahoo_source.fetch_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Connect to database
    db_connector = TimescaleConnector(**DB_PARAMS)
    db_connector.connect()
    
    # Create session
    session = db_connector.get_session()
    
    try:
        # Insert data into database
        from src.data_integration.timescale_schema import FinancialData
        
        # Delete existing data for the same period to avoid duplicates
        session.query(FinancialData).filter(
            FinancialData.symbol == symbol,
            FinancialData.timestamp >= datetime.fromisoformat(start_date),
            FinancialData.timestamp <= datetime.fromisoformat(end_date)
        ).delete()
        
        # Insert new data
        for index, row in data.iterrows():
            financial_data = FinancialData(
                timestamp=index,
                symbol=symbol,
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row['Volume'],
                adjusted_close=row['Adj Close'],
                source='yahoo_finance',
                metadata={'data_acquisition_time': datetime.now().isoformat()}
            )
            session.add(financial_data)
        
        # Commit changes
        session.commit()
        logger.info(f"Successfully inserted {len(data)} records for {symbol}")
        
        # Log sync
        from src.data_integration.timescale_schema import DataSyncLog
        
        sync_log = DataSyncLog(
            timestamp=datetime.now(),
            data_type='financial_data',
            source='yahoo_finance',
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            records_processed=len(data),
            success=True,
            metadata={'symbol': symbol}
        )
        session.add(sync_log)
        session.commit()
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error inserting financial data: {str(e)}")
        
        # Log sync error
        from src.data_integration.timescale_schema import DataSyncLog
        
        sync_log = DataSyncLog(
            timestamp=datetime.now(),
            data_type='financial_data',
            source='yahoo_finance',
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            records_processed=0,
            success=False,
            error_message=str(e),
            metadata={'symbol': symbol}
        )
        session.add(sync_log)
        session.commit()
        
        raise
    finally:
        session.close()


def calculate_planetary_positions(start_date: str, end_date: str, interval_days: int = 1, **kwargs) -> None:
    """
    Calculate planetary positions and store in TimescaleDB.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval_days: Interval between calculations in days
    """
    logger.info(f"Calculating planetary positions from {start_date} to {end_date}")
    
    # Create planetary calculator
    calculator = PlanetaryCalculator()
    
    # Generate date range
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    dates = []
    
    current_date = start
    while current_date <= end:
        dates.append(current_date)
        current_date += timedelta(days=interval_days)
    
    # Connect to database
    db_connector = TimescaleConnector(**DB_PARAMS)
    db_connector.connect()
    
    # Create session
    session = db_connector.get_session()
    
    try:
        # Insert data into database
        from src.data_integration.timescale_schema import PlanetaryPosition
        from src.astro_engine.planetary_positions import (
            SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU
        )
        
        # List of planets to calculate
        planets = [SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU]
        
        # Delete existing data for the same period to avoid duplicates
        session.query(PlanetaryPosition).filter(
            PlanetaryPosition.timestamp >= start,
            PlanetaryPosition.timestamp <= end
        ).delete()
        
        # Calculate and insert positions for each date and planet
        records_processed = 0
        
        for date in dates:
            for planet_id in planets:
                # Calculate position
                position = calculator.get_planet_position(planet_id, date)
                
                # Get nakshatra
                nakshatra_details = calculator.get_nakshatra_details(position['longitude'])
                
                # Create database record
                planetary_position = PlanetaryPosition(
                    timestamp=date,
                    planet_id=planet_id,
                    longitude=position['longitude'],
                    latitude=position['latitude'],
                    distance=position['distance'],
                    speed=position['longitude_speed'],
                    is_retrograde=position['is_retrograde'],
                    sign=int(position['longitude'] / 30) + 1,  # 1-based sign
                    nakshatra=nakshatra_details['nakshatra'],
                    dignity=None,  # Will be calculated in a separate step
                    aspects=None   # Will be calculated in a separate step
                )
                
                session.add(planetary_position)
                records_processed += 1
                
                # Commit in batches to avoid memory issues
                if records_processed % 100 == 0:
                    session.commit()
        
        # Final commit
        session.commit()
        logger.info(f"Successfully inserted {records_processed} planetary positions")
        
        # Log sync
        from src.data_integration.timescale_schema import DataSyncLog
        
        sync_log = DataSyncLog(
            timestamp=datetime.now(),
            data_type='planetary_positions',
            source='swiss_ephemeris',
            start_date=start,
            end_date=end,
            records_processed=records_processed,
            success=True,
            metadata={'interval_days': interval_days}
        )
        session.add(sync_log)
        session.commit()
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error calculating planetary positions: {str(e)}")
        
        # Log sync error
        from src.data_integration.timescale_schema import DataSyncLog
        
        sync_log = DataSyncLog(
            timestamp=datetime.now(),
            data_type='planetary_positions',
            source='swiss_ephemeris',
            start_date=start,
            end_date=end,
            records_processed=0,
            success=False,
            error_message=str(e),
            metadata={'interval_days': interval_days}
        )
        session.add(sync_log)
        session.commit()
        
        raise
    finally:
        session.close()


def calculate_astrological_events(start_date: str, end_date: str, **kwargs) -> None:
    """
    Calculate significant astrological events and store in TimescaleDB.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    logger.info(f"Calculating astrological events from {start_date} to {end_date}")
    
    # Create Vedic market analyzer
    # Using the new consolidated VedicAnalyzer class instead of the deprecated VedicMarketAnalyzer
    analyzer = VedicAnalyzer()
    
    # Connect to database
    db_manager = DatabaseManager(**DB_PARAMS)
    db_manager.connect()
    
    # Create session
    session = db_manager.Session()
    
    try:
        # Generate date range
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        # Analyze each date
        date_range = analyzer.analyze_date_range(start, end)
        
        # Insert data into database
        from src.data_integration.timescale_schema import AstrologicalEvent
        
        # Delete existing data for the same period to avoid duplicates
        session.query(AstrologicalEvent).filter(
            AstrologicalEvent.timestamp >= start,
            AstrologicalEvent.timestamp <= end
        ).delete()
        
        # Process each date's analysis
        records_processed = 0
        
        for _, row in date_range.iterrows():
            date = row['date']
            
            # Get full analysis for the date
            analysis = analyzer.analyze_date(date)
            
            # Extract yogas (planetary combinations)
            if 'yogas' in analysis and 'dhana_yogas' in analysis['yogas']:
                for yoga in analysis['yogas']['dhana_yogas']:
                    event = AstrologicalEvent(
                        timestamp=date,
                        event_type='dhana_yoga',
                        planets_involved=yoga['planets_involved'],
                        description=yoga['description'],
                        strength=yoga['strength'],
                        market_impact=yoga['market_impact'],
                        metadata={'yoga_name': yoga['name']}
                    )
                    session.add(event)
                    records_processed += 1
            
            if 'yogas' in analysis and 'raja_yogas' in analysis['yogas']:
                for yoga in analysis['yogas']['raja_yogas']:
                    event = AstrologicalEvent(
                        timestamp=date,
                        event_type='raja_yoga',
                        planets_involved=yoga['planets_involved'],
                        description=yoga['description'],
                        strength=yoga['strength'],
                        market_impact=yoga['market_impact'],
                        metadata={'yoga_name': yoga['name']}
                    )
                    session.add(event)
                    records_processed += 1
            
            # Extract timing signals
            if 'timing_signals' in analysis:
                signals = analysis['timing_signals']
                
                for signal_type in ['entry_signals', 'exit_signals', 'caution_signals']:
                    if signal_type in signals:
                        for signal in signals[signal_type]:
                            event = AstrologicalEvent(
                                timestamp=date,
                                event_type=f"timing_{signal['type']}",
                                planets_involved=[signal.get('planet')] if 'planet' in signal else [],
                                description=signal['description'],
                                strength=0.8,  # Default strength for timing signals
                                market_impact='positive' if signal_type == 'entry_signals' else 
                                             ('negative' if signal_type == 'exit_signals' else 'neutral'),
                                metadata={'signal_type': signal_type}
                            )
                            session.add(event)
                            records_processed += 1
            
            # Commit in batches to avoid memory issues
            if records_processed % 50 == 0:
                session.commit()
        
        # Final commit
        session.commit()
        logger.info(f"Successfully inserted {records_processed} astrological events")
        
        # Log sync
        from src.data_integration.timescale_schema import DataSyncLog
        
        sync_log = DataSyncLog(
            timestamp=datetime.now(),
            data_type='astrological_events',
            source='vedic_analysis',
            start_date=start,
            end_date=end,
            records_processed=records_processed,
            success=True,
            metadata={}
        )
        session.add(sync_log)
        session.commit()
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error calculating astrological events: {str(e)}")
        
        # Log sync error
        from src.data_integration.timescale_schema import DataSyncLog
        
        sync_log = DataSyncLog(
            timestamp=datetime.now(),
            data_type='astrological_events',
            source='vedic_analysis',
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            records_processed=0,
            success=False,
            error_message=str(e),
            metadata={}
        )
        session.add(sync_log)
        session.commit()
        
        raise
    finally:
        session.close()


def calculate_market_regimes(symbol: str, start_date: str, end_date: str, **kwargs) -> None:
    """
    Calculate market regimes and store in TimescaleDB.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    logger.info(f"Calculating market regimes for {symbol} from {start_date} to {end_date}")
    
    # Connect to database
    db_manager = DatabaseManager(**DB_PARAMS)
    db_manager.connect()
    
    # Create session
    session = db_manager.Session()
    
    try:
        # Fetch financial data from database
        from src.data_integration.timescale_schema import FinancialData
        
        query = session.query(FinancialData).filter(
            FinancialData.symbol == symbol,
            FinancialData.timestamp >= datetime.fromisoformat(start_date),
            FinancialData.timestamp <= datetime.fromisoformat(end_date)
        ).order_by(FinancialData.timestamp)
        
        # Convert to DataFrame
        import pandas as pd
        
        data = pd.DataFrame([
            {
                'Date': row.timestamp,
                'Open': row.open,
                'High': row.high,
                'Low': row.low,
                'Close': row.close,
                'Volume': row.volume,
                'Adj Close': row.adjusted_close
            }
            for row in query
        ])
        
        if len(data) == 0:
            logger.warning(f"No financial data found for {symbol} from {start_date} to {end_date}")
            return
        
        # Set index
        data.set_index('Date', inplace=True)
        
        # Create market regime labeler
        labeler = MarketRegimeLabeler()
        
        # Calculate regimes
        hmm_regimes = labeler.label_regimes_hmm(data)
        kmeans_regimes = labeler.label_regimes_kmeans(data)
        rule_based_regimes = labeler.label_regimes_rule_based(data)
        
        # Calculate consensus
        consensus_regimes = labeler.create_consensus_labels(
            [hmm_regimes, kmeans_regimes, rule_based_regimes]
        )
        
        # Insert data into database
        from src.data_integration.timescale_schema import MarketRegime
        
        # Delete existing data for the same period to avoid duplicates
        session.query(MarketRegime).filter(
            MarketRegime.symbol == symbol,
            MarketRegime.timestamp >= datetime.fromisoformat(start_date),
            MarketRegime.timestamp <= datetime.fromisoformat(end_date)
        ).delete()
        
        # Insert new data
        records_processed = 0
        
        for date, row in hmm_regimes.iterrows():
            # Calculate volatility and trend
            if date in data.index:
                price_data = data.loc[date]
                volatility = (price_data['High'] - price_data['Low']) / price_data['Close']
                
                # Simple trend calculation (20-day moving average)
                if len(data[:date]) >= 20:
                    ma20 = data['Close'][:date].tail(20).mean()
                    trend = (price_data['Close'] - ma20) / ma20
                else:
                    trend = 0
            else:
                volatility = 0
                trend = 0
            
            # Create database record
            market_regime = MarketRegime(
                timestamp=date,
                symbol=symbol,
                regime_hmm=int(hmm_regimes.loc[date, 'regime']),
                regime_kmeans=int(kmeans_regimes.loc[date, 'regime']),
                regime_rule_based=int(rule_based_regimes.loc[date, 'regime']),
                regime_consensus=int(consensus_regimes.loc[date, 'regime']),
                volatility=float(volatility),
                trend=float(trend),
                metadata={
                    'hmm_probability': float(hmm_regimes.loc[date, 'probability']),
                    'kmeans_distance': float(kmeans_regimes.loc[date, 'distance']),
                    'rule_based_strength': float(rule_based_regimes.loc[date, 'strength'])
                }
            )
            
            session.add(market_regime)
            records_processed += 1
            
            # Commit in batches to avoid memory issues
            if records_processed % 100 == 0:
                session.commit()
        
        # Final commit
        session.commit()
        logger.info(f"Successfully inserted {records_processed} market regimes for {symbol}")
        
        # Log sync
        from src.data_integration.timescale_schema import DataSyncLog
        
        sync_log = DataSyncLog(
            timestamp=datetime.now(),
            data_type='market_regimes',
            source='market_regime_labeler',
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            records_processed=records_processed,
            success=True,
            metadata={'symbol': symbol}
        )
        session.add(sync_log)
        session.commit()
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error calculating market regimes: {str(e)}")
        
        # Log sync error
        from src.data_integration.timescale_schema import DataSyncLog
        
        sync_log = DataSyncLog(
            timestamp=datetime.now(),
            data_type='market_regimes',
            source='market_regime_labeler',
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            records_processed=0,
            success=False,
            error_message=str(e),
            metadata={'symbol': symbol}
        )
        session.add(sync_log)
        session.commit()
        
        raise
    finally:
        session.close()


# Define DAG for daily financial data updates
daily_financial_data_dag = DAG(
    'daily_financial_data_update',
    default_args=default_args,
    description='Daily update of financial data',
    schedule_interval='0 0 * * *',  # Run at midnight every day
    start_date=days_ago(1),
    catchup=False,
    tags=['financial', 'daily'],
)

# Define tasks for daily financial data DAG
for symbol in ['SPY', 'QQQ', 'DIA', 'IWM']:
    fetch_task = PythonOperator(
        task_id=f'fetch_{symbol}_data',
        python_callable=fetch_financial_data,
        op_kwargs={
            'symbol': symbol,
            'start_date': '{{ macros.ds_add(ds, -5) }}',  # 5 days ago
            'end_date': '{{ ds }}'  # Execution date
        },
        dag=daily_financial_data_dag,
    )
    
    regime_task = PythonOperator(
        task_id=f'calculate_{symbol}_regimes',
        python_callable=calculate_market_regimes,
        op_kwargs={
            'symbol': symbol,
            'start_date': '{{ macros.ds_add(ds, -30) }}',  # 30 days ago (need history for calculations)
            'end_date': '{{ ds }}'  # Execution date
        },
        dag=daily_financial_data_dag,
    )
    
    fetch_task >> regime_task


# Define DAG for daily astrological data updates
daily_astrological_data_dag = DAG(
    'daily_astrological_data_update',
    default_args=default_args,
    description='Daily update of astrological data',
    schedule_interval='0 1 * * *',  # Run at 1 AM every day
    start_date=days_ago(1),
    catchup=False,
    tags=['astrological', 'daily'],
)

# Define tasks for daily astrological data DAG
planetary_positions_task = PythonOperator(
    task_id='calculate_planetary_positions',
    python_callable=calculate_planetary_positions,
    op_kwargs={
        'start_date': '{{ ds }}',  # Execution date
        'end_date': '{{ macros.ds_add(ds, 7) }}',  # 7 days ahead
        'interval_days': 1
    },
    dag=daily_astrological_data_dag,
)

astrological_events_task = PythonOperator(
    task_id='calculate_astrological_events',
    python_callable=calculate_astrological_events,
    op_kwargs={
        'start_date': '{{ ds }}',  # Execution date
        'end_date': '{{ macros.ds_add(ds, 7) }}',  # 7 days ahead
    },
    dag=daily_astrological_data_dag,
)

planetary_positions_task >> astrological_events_task


# Define DAG for historical data backfill
historical_backfill_dag = DAG(
    'historical_data_backfill',
    default_args=default_args,
    description='Backfill historical financial and astrological data',
    schedule_interval=None,  # Manual trigger only
    start_date=days_ago(1),
    catchup=False,
    tags=['backfill', 'historical'],
)

# Define tasks for historical backfill DAG
for symbol in ['SPY', 'QQQ', 'DIA', 'IWM']:
    historical_fetch_task = PythonOperator(
        task_id=f'fetch_historical_{symbol}_data',
        python_callable=fetch_financial_data,
        op_kwargs={
            'symbol': symbol,
            'start_date': '{{ params.start_date }}',
            'end_date': '{{ params.end_date }}'
        },
        dag=historical_backfill_dag,
    )
    
    historical_regime_task = PythonOperator(
        task_id=f'calculate_historical_{symbol}_regimes',
        python_callable=calculate_market_regimes,
        op_kwargs={
            'symbol': symbol,
            'start_date': '{{ params.start_date }}',
            'end_date': '{{ params.end_date }}'
        },
        dag=historical_backfill_dag,
    )
    
    historical_fetch_task >> historical_regime_task

historical_planetary_positions_task = PythonOperator(
    task_id='calculate_historical_planetary_positions',
    python_callable=calculate_planetary_positions,
    op_kwargs={
        'start_date': '{{ params.start_date }}',
        'end_date': '{{ params.end_date }}',
        'interval_days': 1
    },
    dag=historical_backfill_dag,
)

historical_astrological_events_task = PythonOperator(
    task_id='calculate_historical_astrological_events',
    python_callable=calculate_astrological_events,
    op_kwargs={
        'start_date': '{{ params.start_date }}',
        'end_date': '{{ params.end_date }}'
    },
    dag=historical_backfill_dag,
)

historical_planetary_positions_task >> historical_astrological_events_task
