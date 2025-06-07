"""
Data Quality Monitoring System for the Cosmic Market Oracle.

This module implements an automated data quality monitoring system that checks
for data issues, missing values, outliers, and inconsistencies in the integrated
datasets, with alerting capabilities.
"""

import os
import logging
import json
import smtplib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sqlalchemy import func, and_, or_, not_
from sqlalchemy.sql import text

from src.data_integration.timescale_schema import (
    PlanetaryPosition, FinancialData, MarketRegime,
    AstrologicalEvent, TechnicalIndicator, Prediction, DataSyncLog
)
from src.utils.db.timescale_connector import TimescaleConnector
from src.utils.logging_config import setup_logging
from config.db_config import get_db_params

# Set up logging
logger = setup_logging(__name__)

# Get database parameters
DB_PARAMS = get_db_params()

# Email configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': 'alerts@cosmicmarketoracle.com',
    'password': os.environ.get('EMAIL_PASSWORD', ''),
    'from_email': 'alerts@cosmicmarketoracle.com',
    'to_emails': ['admin@cosmicmarketoracle.com']
}


class DataQualityCheck:
    """Base class for data quality checks."""
    
    def __init__(self, name: str, description: str, severity: str = 'medium'):
        """
        Initialize a data quality check.
        
        Args:
            name: Check name
            description: Check description
            severity: Check severity (low, medium, high, critical)
        """
        self.name = name
        self.description = description
        self.severity = severity
        self.passed = None
        self.details = {}
        
    def run(self, session) -> bool:
        """
        Run the check.
        
        Args:
            session: Database session
            
        Returns:
            True if check passed, False otherwise
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def get_result(self) -> Dict:
        """Get check result."""
        return {
            'name': self.name,
            'description': self.description,
            'severity': self.severity,
            'passed': self.passed,
            'details': self.details,
            'timestamp': datetime.now().isoformat()
        }


class MissingDataCheck(DataQualityCheck):
    """Check for missing data in a specified time range."""
    
    def __init__(self, table_class, time_column: str, expected_interval: timedelta,
                 start_date: datetime, end_date: datetime, filters: Optional[List] = None,
                 name: Optional[str] = None, description: Optional[str] = None,
                 severity: str = 'medium', threshold: float = 0.05):
        """
        Initialize a missing data check.
        
        Args:
            table_class: SQLAlchemy model class
            time_column: Name of the timestamp column
            expected_interval: Expected interval between records
            start_date: Start date for check
            end_date: End date for check
            filters: Additional filters to apply
            name: Check name
            description: Check description
            severity: Check severity
            threshold: Maximum allowed missing data ratio
        """
        name = name or f"Missing {table_class.__name__} Check"
        description = description or f"Check for missing {table_class.__name__} data"
        
        super().__init__(name, description, severity)
        
        self.table_class = table_class
        self.time_column = time_column
        self.expected_interval = expected_interval
        self.start_date = start_date
        self.end_date = end_date
        self.filters = filters or []
        self.threshold = threshold
        
    def run(self, session) -> bool:
        """Run the check."""
        # Get actual timestamps in the range
        query = session.query(getattr(self.table_class, self.time_column)).filter(
            getattr(self.table_class, self.time_column) >= self.start_date,
            getattr(self.table_class, self.time_column) <= self.end_date
        )
        
        # Apply additional filters
        for filter_condition in self.filters:
            query = query.filter(filter_condition)
            
        actual_timestamps = [row[0] for row in query.all()]
        
        # Generate expected timestamps
        expected_timestamps = []
        current_time = self.start_date
        
        while current_time <= self.end_date:
            expected_timestamps.append(current_time)
            current_time += self.expected_interval
            
        # Find missing timestamps
        missing_timestamps = [ts for ts in expected_timestamps if ts not in actual_timestamps]
        
        # Calculate missing ratio
        total_expected = len(expected_timestamps)
        total_missing = len(missing_timestamps)
        missing_ratio = total_missing / total_expected if total_expected > 0 else 0
        
        # Check if missing ratio exceeds threshold
        self.passed = missing_ratio <= self.threshold
        
        # Store details
        self.details = {
            'total_expected': total_expected,
            'total_actual': len(actual_timestamps),
            'total_missing': total_missing,
            'missing_ratio': missing_ratio,
            'threshold': self.threshold,
            'missing_timestamps': [ts.isoformat() for ts in missing_timestamps[:10]]  # Show first 10
        }
        
        return self.passed


class OutlierCheck(DataQualityCheck):
    """Check for outliers in numerical columns."""
    
    def __init__(self, table_class, column: str, start_date: datetime, end_date: datetime,
                 time_column: str = 'timestamp', method: str = 'zscore',
                 threshold: float = 3.0, filters: Optional[List] = None,
                 name: Optional[str] = None, description: Optional[str] = None,
                 severity: str = 'medium', max_outliers: int = 10):
        """
        Initialize an outlier check.
        
        Args:
            table_class: SQLAlchemy model class
            column: Column to check for outliers
            start_date: Start date for check
            end_date: End date for check
            time_column: Name of the timestamp column
            method: Outlier detection method ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            filters: Additional filters to apply
            name: Check name
            description: Check description
            severity: Check severity
            max_outliers: Maximum allowed number of outliers
        """
        name = name or f"Outlier {table_class.__name__}.{column} Check"
        description = description or f"Check for outliers in {table_class.__name__}.{column}"
        
        super().__init__(name, description, severity)
        
        self.table_class = table_class
        self.column = column
        self.time_column = time_column
        self.start_date = start_date
        self.end_date = end_date
        self.method = method
        self.threshold = threshold
        self.filters = filters or []
        self.max_outliers = max_outliers
        
    def run(self, session) -> bool:
        """Run the check."""
        # Get data
        query = session.query(
            getattr(self.table_class, self.time_column),
            getattr(self.table_class, self.column)
        ).filter(
            getattr(self.table_class, self.time_column) >= self.start_date,
            getattr(self.table_class, self.time_column) <= self.end_date
        )
        
        # Apply additional filters
        for filter_condition in self.filters:
            query = query.filter(filter_condition)
            
        # Convert to DataFrame
        data = pd.DataFrame(query.all(), columns=[self.time_column, self.column])
        
        if len(data) == 0:
            self.passed = True
            self.details = {
                'message': 'No data found for the specified period',
                'total_records': 0,
                'outliers': []
            }
            return self.passed
        
        # Detect outliers
        if self.method == 'zscore':
            # Z-score method
            mean = data[self.column].mean()
            std = data[self.column].std()
            
            if std == 0:
                self.passed = True
                self.details = {
                    'message': 'Standard deviation is zero, no outliers detected',
                    'total_records': len(data),
                    'outliers': []
                }
                return self.passed
                
            data['zscore'] = (data[self.column] - mean) / std
            outliers = data[abs(data['zscore']) > self.threshold]
            
        elif self.method == 'iqr':
            # IQR method
            q1 = data[self.column].quantile(0.25)
            q3 = data[self.column].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr
            
            outliers = data[(data[self.column] < lower_bound) | (data[self.column] > upper_bound)]
            
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")
        
        # Check if number of outliers exceeds threshold
        self.passed = len(outliers) <= self.max_outliers
        
        # Store details
        self.details = {
            'total_records': len(data),
            'total_outliers': len(outliers),
            'method': self.method,
            'threshold': self.threshold,
            'max_outliers': self.max_outliers,
            'outliers': outliers.to_dict(orient='records')[:10]  # Show first 10
        }
        
        return self.passed


class ConsistencyCheck(DataQualityCheck):
    """Check for data consistency between related tables."""
    
    def __init__(self, primary_table, primary_column: str, related_table, related_column: str,
                 start_date: datetime, end_date: datetime, time_column: str = 'timestamp',
                 filters: Optional[List] = None, name: Optional[str] = None,
                 description: Optional[str] = None, severity: str = 'high',
                 threshold: float = 0.01):
        """
        Initialize a consistency check.
        
        Args:
            primary_table: Primary SQLAlchemy model class
            primary_column: Primary column name
            related_table: Related SQLAlchemy model class
            related_column: Related column name
            start_date: Start date for check
            end_date: End date for check
            time_column: Name of the timestamp column
            filters: Additional filters to apply
            name: Check name
            description: Check description
            severity: Check severity
            threshold: Maximum allowed inconsistency ratio
        """
        name = name or f"Consistency {primary_table.__name__} vs {related_table.__name__} Check"
        description = description or f"Check for consistency between {primary_table.__name__} and {related_table.__name__}"
        
        super().__init__(name, description, severity)
        
        self.primary_table = primary_table
        self.primary_column = primary_column
        self.related_table = related_table
        self.related_column = related_column
        self.time_column = time_column
        self.start_date = start_date
        self.end_date = end_date
        self.filters = filters or []
        self.threshold = threshold
        
    def run(self, session) -> bool:
        """Run the check."""
        # Get primary values
        primary_query = session.query(
            getattr(self.primary_table, self.time_column),
            getattr(self.primary_table, self.primary_column)
        ).filter(
            getattr(self.primary_table, self.time_column) >= self.start_date,
            getattr(self.primary_table, self.time_column) <= self.end_date
        )
        
        # Apply additional filters to primary query
        for filter_condition in self.filters:
            if hasattr(filter_condition, '__self__') and isinstance(filter_condition.__self__, self.primary_table):
                primary_query = primary_query.filter(filter_condition)
                
        primary_data = {row[0]: row[1] for row in primary_query.all()}
        
        # Get related values
        related_query = session.query(
            getattr(self.related_table, self.time_column),
            getattr(self.related_table, self.related_column)
        ).filter(
            getattr(self.related_table, self.time_column) >= self.start_date,
            getattr(self.related_table, self.time_column) <= self.end_date
        )
        
        # Apply additional filters to related query
        for filter_condition in self.filters:
            if hasattr(filter_condition, '__self__') and isinstance(filter_condition.__self__, self.related_table):
                related_query = related_query.filter(filter_condition)
                
        related_data = {row[0]: row[1] for row in related_query.all()}
        
        # Find common timestamps
        common_timestamps = set(primary_data.keys()) & set(related_data.keys())
        
        if len(common_timestamps) == 0:
            self.passed = False
            self.details = {
                'message': 'No common timestamps found between tables',
                'primary_count': len(primary_data),
                'related_count': len(related_data),
                'common_count': 0,
                'inconsistencies': []
            }
            return self.passed
        
        # Check for inconsistencies
        inconsistencies = []
        
        for ts in common_timestamps:
            primary_value = primary_data[ts]
            related_value = related_data[ts]
            
            if primary_value != related_value:
                inconsistencies.append({
                    'timestamp': ts.isoformat(),
                    'primary_value': primary_value,
                    'related_value': related_value
                })
        
        # Calculate inconsistency ratio
        inconsistency_ratio = len(inconsistencies) / len(common_timestamps)
        
        # Check if inconsistency ratio exceeds threshold
        self.passed = inconsistency_ratio <= self.threshold
        
        # Store details
        self.details = {
            'primary_count': len(primary_data),
            'related_count': len(related_data),
            'common_count': len(common_timestamps),
            'inconsistency_count': len(inconsistencies),
            'inconsistency_ratio': inconsistency_ratio,
            'threshold': self.threshold,
            'inconsistencies': inconsistencies[:10]  # Show first 10
        }
        
        return self.passed


class CustomSQLCheck(DataQualityCheck):
    """Run a custom SQL query and check the results."""
    
    def __init__(self, sql_query: str, validation_func: Callable[[Any], bool],
                 name: str, description: str, severity: str = 'medium'):
        """
        Initialize a custom SQL check.
        
        Args:
            sql_query: SQL query to execute
            validation_func: Function to validate query results
            name: Check name
            description: Check description
            severity: Check severity
        """
        super().__init__(name, description, severity)
        
        self.sql_query = sql_query
        self.validation_func = validation_func
        
    def run(self, session) -> bool:
        """Run the check."""
        # Execute SQL query
        result = session.execute(text(self.sql_query))
        
        # Convert to list of dictionaries
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
        
        # Validate results
        self.passed = self.validation_func(rows)
        
        # Store details
        self.details = {
            'sql_query': self.sql_query,
            'row_count': len(rows),
            'sample_rows': rows[:5]  # Show first 5 rows
        }
        
        return self.passed


class DataQualityMonitor:
    """Data quality monitoring system."""
    
    def __init__(self, db_params: Dict = None, email_config: Dict = None):
        """
        Initialize the data quality monitor.
        
        Args:
            db_params: Database connection parameters
            email_config: Email configuration
        """
        self.db_params = db_params or DB_PARAMS
        self.email_config = email_config or EMAIL_CONFIG
        self.checks = []
        self.results = []
        
       # Create TimescaleDB connector
        self.db_connector = TimescaleConnector(**self.db_params)
        self.db_connector.connect()
        
    def add_check(self, check: DataQualityCheck) -> None:
        """
        Add a data quality check.
        
        Args:
            check: Data quality check to add
        """
        self.checks.append(check)
        
    def run_checks(self) -> List[Dict]:
        """
        Run all data quality checks.
        
        Returns:
            List of check results
        """
        self.results = []
        
        # Create session
        session = self.db_connector.get_session()
        
        try:
            for check in self.checks:
                logger.info(f"Running check: {check.name}")
                
                try:
                    check.run(session)
                    result = check.get_result()
                    self.results.append(result)
                    
                    if not result['passed']:
                        logger.warning(f"Check failed: {check.name}")
                        logger.warning(f"Details: {result['details']}")
                    else:
                        logger.info(f"Check passed: {check.name}")
                        
                except Exception as e:
                    logger.error(f"Error running check {check.name}: {str(e)}")
                    
                    # Create error result
                    result = {
                        'name': check.name,
                        'description': check.description,
                        'severity': check.severity,
                        'passed': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.results.append(result)
                    
        finally:
            session.close()
            
        return self.results
    
    def send_alert_email(self, failed_checks: List[Dict]) -> None:
        """
        Send an alert email for failed checks.
        
        Args:
            failed_checks: List of failed check results
        """
        if not self.email_config['password']:
            logger.warning("Email password not set, skipping alert email")
            return
            
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = self.email_config['from_email']
        msg['To'] = ', '.join(self.email_config['to_emails'])
        msg['Subject'] = f"Cosmic Market Oracle: Data Quality Alert - {len(failed_checks)} Failed Checks"
        
        # Create email body
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .critical {{ color: #d9534f; font-weight: bold; }}
                .high {{ color: #f0ad4e; font-weight: bold; }}
                .medium {{ color: #5bc0de; }}
                .low {{ color: #5cb85c; }}
            </style>
        </head>
        <body>
            <h2>Cosmic Market Oracle: Data Quality Alert</h2>
            <p>The following data quality checks have failed:</p>
            <table>
                <tr>
                    <th>Check Name</th>
                    <th>Description</th>
                    <th>Severity</th>
                    <th>Timestamp</th>
                </tr>
        """
        
        for check in failed_checks:
            severity_class = check['severity'].lower()
            body += f"""
                <tr>
                    <td>{check['name']}</td>
                    <td>{check['description']}</td>
                    <td class="{severity_class}">{check['severity'].upper()}</td>
                    <td>{check['timestamp']}</td>
                </tr>
            """
            
        body += """
            </table>
            <p>Please check the data quality monitoring system for details.</p>
        </body>
        </html>
        """
        
        # Attach body to email
        msg.attach(MIMEText(body, 'html'))
        
        # Send email
        try:
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Alert email sent to {', '.join(self.email_config['to_emails'])}")
            
        except Exception as e:
            logger.error(f"Error sending alert email: {str(e)}")
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict:
        """
        Generate a data quality report.
        
        Args:
            output_file: Optional file to write report to
            
        Returns:
            Report as dictionary
        """
        # Count results by status and severity
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.get('passed', False))
        failed_checks = total_checks - passed_checks
        
        severity_counts = {
            'critical': sum(1 for r in self.results if r.get('severity') == 'critical'),
            'high': sum(1 for r in self.results if r.get('severity') == 'high'),
            'medium': sum(1 for r in self.results if r.get('severity') == 'medium'),
            'low': sum(1 for r in self.results if r.get('severity') == 'low')
        }
        
        failed_by_severity = {
            'critical': sum(1 for r in self.results if not r.get('passed', False) and r.get('severity') == 'critical'),
            'high': sum(1 for r in self.results if not r.get('passed', False) and r.get('severity') == 'high'),
            'medium': sum(1 for r in self.results if not r.get('passed', False) and r.get('severity') == 'medium'),
            'low': sum(1 for r in self.results if not r.get('passed', False) and r.get('severity') == 'low')
        }
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'failed_checks': failed_checks,
                'pass_rate': passed_checks / total_checks if total_checks > 0 else 0,
                'severity_counts': severity_counts,
                'failed_by_severity': failed_by_severity
            },
            'results': self.results
        }
        
        # Write report to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Report written to {output_file}")
            
        return report
    
    def run_and_alert(self, output_file: Optional[str] = None) -> Dict:
        """
        Run all checks, send alerts for failures, and generate a report.
        
        Args:
            output_file: Optional file to write report to
            
        Returns:
            Report as dictionary
        """
        # Run checks
        self.run_checks()
        
        # Filter failed checks
        failed_checks = [r for r in self.results if not r.get('passed', False)]
        
        # Send alert if there are failed checks
        if failed_checks:
            self.send_alert_email(failed_checks)
            
        # Generate report
        return self.generate_report(output_file)


def create_standard_checks(start_date: datetime, end_date: datetime) -> List[DataQualityCheck]:
    """
    Create a standard set of data quality checks.
    
    Args:
        start_date: Start date for checks
        end_date: End date for checks
        
    Returns:
        List of data quality checks
    """
    checks = []
    
    # Financial data checks
    for symbol in ['SPY', 'QQQ', 'DIA', 'IWM']:
        # Missing data check
        checks.append(MissingDataCheck(
            FinancialData, 'timestamp', timedelta(days=1),
            start_date, end_date,
            filters=[FinancialData.symbol == symbol],
            name=f"Missing {symbol} Data Check",
            description=f"Check for missing {symbol} price data",
            severity='high'
        ))
        
        # Outlier checks
        checks.append(OutlierCheck(
            FinancialData, 'close', start_date, end_date,
            filters=[FinancialData.symbol == symbol],
            name=f"{symbol} Close Price Outlier Check",
            description=f"Check for outliers in {symbol} close prices",
            severity='medium'
        ))
        
        checks.append(OutlierCheck(
            FinancialData, 'volume', start_date, end_date,
            filters=[FinancialData.symbol == symbol],
            name=f"{symbol} Volume Outlier Check",
            description=f"Check for outliers in {symbol} volume",
            severity='medium'
        ))
        
        # Consistency check
        checks.append(ConsistencyCheck(
            FinancialData, 'close', MarketRegime, 'regime_consensus',
            start_date, end_date,
            name=f"{symbol} Price-Regime Consistency Check",
            description=f"Check for consistency between {symbol} prices and market regimes",
            severity='medium'
        ))
    
    # Planetary position checks
    from src.astro_engine.planetary_positions import SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN
    
    for planet_id in [SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN]:
        # Missing data check
        checks.append(MissingDataCheck(
            PlanetaryPosition, 'timestamp', timedelta(days=1),
            start_date, end_date,
            filters=[PlanetaryPosition.planet_id == planet_id],
            name=f"Missing Planet {planet_id} Data Check",
            description=f"Check for missing planetary position data for planet {planet_id}",
            severity='high'
        ))
        
        # Outlier checks
        checks.append(OutlierCheck(
            PlanetaryPosition, 'longitude', start_date, end_date,
            filters=[PlanetaryPosition.planet_id == planet_id],
            name=f"Planet {planet_id} Longitude Outlier Check",
            description=f"Check for outliers in planet {planet_id} longitude",
            severity='medium'
        ))
    
    # Custom SQL checks
    
    # Check for duplicate financial data
    checks.append(CustomSQLCheck(
        f"""
        SELECT symbol, timestamp, COUNT(*) as count
        FROM cosmic_data.financial_data
        WHERE timestamp BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}'
        GROUP BY symbol, timestamp
        HAVING COUNT(*) > 1
        """,
        lambda rows: len(rows) == 0,
        "Duplicate Financial Data Check",
        "Check for duplicate financial data entries",
        "high"
    ))
    
    # Check for duplicate planetary positions
    checks.append(CustomSQLCheck(
        f"""
        SELECT planet_id, timestamp, COUNT(*) as count
        FROM cosmic_data.planetary_positions
        WHERE timestamp BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}'
        GROUP BY planet_id, timestamp
        HAVING COUNT(*) > 1
        """,
        lambda rows: len(rows) == 0,
        "Duplicate Planetary Position Check",
        "Check for duplicate planetary position entries",
        "high"
    ))
    
    # Check for data sync failures
    checks.append(CustomSQLCheck(
        f"""
        SELECT data_type, source, COUNT(*) as count
        FROM cosmic_data.data_sync_log
        WHERE timestamp BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}'
        AND success = false
        GROUP BY data_type, source
        """,
        lambda rows: len(rows) == 0,
        "Data Sync Failure Check",
        "Check for data synchronization failures",
        "critical"
    ))
    
    return checks


# Example usage
if __name__ == "__main__":
    # Create data quality monitor
    monitor = DataQualityMonitor()
    
    # Define date range for checks
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Create standard checks
    checks = create_standard_checks(start_date, end_date)
    
    # Add checks to monitor
    for check in checks:
        monitor.add_check(check)
    
    # Run checks and generate report
    report = monitor.run_and_alert("data_quality_report.json")
    
    # Print summary
    print("Data Quality Report Summary:")
    print(f"Total checks: {report['summary']['total_checks']}")
    print(f"Passed checks: {report['summary']['passed_checks']}")
    print(f"Failed checks: {report['summary']['failed_checks']}")
    print(f"Pass rate: {report['summary']['pass_rate']:.2%}")
    print("Failed checks by severity:")
    for severity, count in report['summary']['failed_by_severity'].items():
        print(f"  {severity}: {count}")
