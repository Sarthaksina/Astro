"""
GraphQL API Gateway for the Cosmic Market Oracle.

This module implements a GraphQL API for flexible data access to the TimescaleDB database,
allowing complex queries across financial and astrological datasets.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta

# FastAPI imports
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# GraphQL imports
import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info
import sqlalchemy
from sqlalchemy.orm import Session

# Project imports
from src.data_integration.timescale_schema import (
    DatabaseManager, PlanetaryPosition, FinancialData, MarketRegime,
    AstrologicalEvent, TechnicalIndicator, Prediction, DataSyncLog
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection parameters
DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': 'postgres',
    'database': 'cosmic_oracle',
    'schema': 'cosmic_data'
}

# Create database manager
db_manager = DatabaseManager(**DB_PARAMS)
db_manager.connect()

# Create session factory
Session = db_manager.Session


# Define context dependency
async def get_context():
    """Create GraphQL context with database session."""
    session = Session()
    try:
        yield {"session": session}
    finally:
        session.close()


# Define GraphQL types
@strawberry.type
class PlanetaryPositionType:
    """GraphQL type for planetary positions."""
    
    id: int
    timestamp: datetime
    planet_id: int
    longitude: float
    latitude: float
    distance: float
    speed: float
    is_retrograde: bool
    sign: int
    nakshatra: int
    house: Optional[int] = None
    dignity: Optional[str] = None
    aspects: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_db_model(cls, model: PlanetaryPosition) -> 'PlanetaryPositionType':
        """Convert database model to GraphQL type."""
        return cls(
            id=model.id,
            timestamp=model.timestamp,
            planet_id=model.planet_id,
            longitude=model.longitude,
            latitude=model.latitude,
            distance=model.distance,
            speed=model.speed,
            is_retrograde=model.is_retrograde,
            sign=model.sign,
            nakshatra=model.nakshatra,
            house=model.house,
            dignity=model.dignity,
            aspects=model.aspects
        )


@strawberry.type
class FinancialDataType:
    """GraphQL type for financial data."""
    
    id: int
    timestamp: datetime
    symbol: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: float
    volume: Optional[float] = None
    adjusted_close: Optional[float] = None
    source: str
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_db_model(cls, model: FinancialData) -> 'FinancialDataType':
        """Convert database model to GraphQL type."""
        return cls(
            id=model.id,
            timestamp=model.timestamp,
            symbol=model.symbol,
            open=model.open,
            high=model.high,
            low=model.low,
            close=model.close,
            volume=model.volume,
            adjusted_close=model.adjusted_close,
            source=model.source,
            metadata=model.metadata
        )


@strawberry.type
class MarketRegimeType:
    """GraphQL type for market regimes."""
    
    id: int
    timestamp: datetime
    symbol: str
    regime_hmm: Optional[int] = None
    regime_kmeans: Optional[int] = None
    regime_rule_based: Optional[int] = None
    regime_consensus: Optional[int] = None
    volatility: Optional[float] = None
    trend: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_db_model(cls, model: MarketRegime) -> 'MarketRegimeType':
        """Convert database model to GraphQL type."""
        return cls(
            id=model.id,
            timestamp=model.timestamp,
            symbol=model.symbol,
            regime_hmm=model.regime_hmm,
            regime_kmeans=model.regime_kmeans,
            regime_rule_based=model.regime_rule_based,
            regime_consensus=model.regime_consensus,
            volatility=model.volatility,
            trend=model.trend,
            metadata=model.metadata
        )


@strawberry.type
class AstrologicalEventType:
    """GraphQL type for astrological events."""
    
    id: int
    timestamp: datetime
    event_type: str
    planets_involved: List[int]
    description: str
    strength: Optional[float] = None
    market_impact: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_db_model(cls, model: AstrologicalEvent) -> 'AstrologicalEventType':
        """Convert database model to GraphQL type."""
        return cls(
            id=model.id,
            timestamp=model.timestamp,
            event_type=model.event_type,
            planets_involved=model.planets_involved,
            description=model.description,
            strength=model.strength,
            market_impact=model.market_impact,
            metadata=model.metadata
        )


@strawberry.type
class TechnicalIndicatorType:
    """GraphQL type for technical indicators."""
    
    id: int
    timestamp: datetime
    symbol: str
    indicator_type: str
    value: float
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_db_model(cls, model: TechnicalIndicator) -> 'TechnicalIndicatorType':
        """Convert database model to GraphQL type."""
        return cls(
            id=model.id,
            timestamp=model.timestamp,
            symbol=model.symbol,
            indicator_type=model.indicator_type,
            value=model.value,
            parameters=model.parameters,
            metadata=model.metadata
        )


@strawberry.type
class PredictionType:
    """GraphQL type for model predictions."""
    
    id: int
    timestamp: datetime
    symbol: str
    model_id: str
    prediction_type: str
    target_timestamp: datetime
    value: float
    confidence: Optional[float] = None
    features_used: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_db_model(cls, model: Prediction) -> 'PredictionType':
        """Convert database model to GraphQL type."""
        return cls(
            id=model.id,
            timestamp=model.timestamp,
            symbol=model.symbol,
            model_id=model.model_id,
            prediction_type=model.prediction_type,
            target_timestamp=model.target_timestamp,
            value=model.value,
            confidence=model.confidence,
            features_used=model.features_used,
            metadata=model.metadata
        )


@strawberry.type
class DataSyncLogType:
    """GraphQL type for data sync logs."""
    
    id: int
    timestamp: datetime
    data_type: str
    source: str
    start_date: datetime
    end_date: datetime
    records_processed: int
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_db_model(cls, model: DataSyncLog) -> 'DataSyncLogType':
        """Convert database model to GraphQL type."""
        return cls(
            id=model.id,
            timestamp=model.timestamp,
            data_type=model.data_type,
            source=model.source,
            start_date=model.start_date,
            end_date=model.end_date,
            records_processed=model.records_processed,
            success=model.success,
            error_message=model.error_message,
            metadata=model.metadata
        )


@strawberry.type
class IntegratedDataPoint:
    """GraphQL type for integrated data points combining financial and astrological data."""
    
    timestamp: datetime
    financial_data: Optional[FinancialDataType] = None
    market_regime: Optional[MarketRegimeType] = None
    planetary_positions: List[PlanetaryPositionType] = strawberry.field(default_factory=list)
    astrological_events: List[AstrologicalEventType] = strawberry.field(default_factory=list)
    technical_indicators: List[TechnicalIndicatorType] = strawberry.field(default_factory=list)
    predictions: List[PredictionType] = strawberry.field(default_factory=list)


# Define GraphQL queries
@strawberry.type
class Query:
    """GraphQL query root type."""
    
    @strawberry.field
    def planetary_positions(
        self, 
        info: Info, 
        start_date: datetime, 
        end_date: datetime,
        planet_id: Optional[int] = None,
        limit: int = 100
    ) -> List[PlanetaryPositionType]:
        """Query planetary positions."""
        session = info.context["session"]
        
        query = session.query(PlanetaryPosition).filter(
            PlanetaryPosition.timestamp >= start_date,
            PlanetaryPosition.timestamp <= end_date
        )
        
        if planet_id is not None:
            query = query.filter(PlanetaryPosition.planet_id == planet_id)
            
        query = query.order_by(PlanetaryPosition.timestamp).limit(limit)
        
        return [PlanetaryPositionType.from_db_model(model) for model in query]
    
    @strawberry.field
    def financial_data(
        self, 
        info: Info, 
        symbol: str,
        start_date: datetime, 
        end_date: datetime,
        limit: int = 100
    ) -> List[FinancialDataType]:
        """Query financial data."""
        session = info.context["session"]
        
        query = session.query(FinancialData).filter(
            FinancialData.symbol == symbol,
            FinancialData.timestamp >= start_date,
            FinancialData.timestamp <= end_date
        ).order_by(FinancialData.timestamp).limit(limit)
        
        return [FinancialDataType.from_db_model(model) for model in query]
    
    @strawberry.field
    def market_regimes(
        self, 
        info: Info, 
        symbol: str,
        start_date: datetime, 
        end_date: datetime,
        limit: int = 100
    ) -> List[MarketRegimeType]:
        """Query market regimes."""
        session = info.context["session"]
        
        query = session.query(MarketRegime).filter(
            MarketRegime.symbol == symbol,
            MarketRegime.timestamp >= start_date,
            MarketRegime.timestamp <= end_date
        ).order_by(MarketRegime.timestamp).limit(limit)
        
        return [MarketRegimeType.from_db_model(model) for model in query]
    
    @strawberry.field
    def astrological_events(
        self, 
        info: Info, 
        start_date: datetime, 
        end_date: datetime,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[AstrologicalEventType]:
        """Query astrological events."""
        session = info.context["session"]
        
        query = session.query(AstrologicalEvent).filter(
            AstrologicalEvent.timestamp >= start_date,
            AstrologicalEvent.timestamp <= end_date
        )
        
        if event_type is not None:
            query = query.filter(AstrologicalEvent.event_type == event_type)
            
        query = query.order_by(AstrologicalEvent.timestamp).limit(limit)
        
        return [AstrologicalEventType.from_db_model(model) for model in query]
    
    @strawberry.field
    def technical_indicators(
        self, 
        info: Info, 
        symbol: str,
        indicator_type: str,
        start_date: datetime, 
        end_date: datetime,
        limit: int = 100
    ) -> List[TechnicalIndicatorType]:
        """Query technical indicators."""
        session = info.context["session"]
        
        query = session.query(TechnicalIndicator).filter(
            TechnicalIndicator.symbol == symbol,
            TechnicalIndicator.indicator_type == indicator_type,
            TechnicalIndicator.timestamp >= start_date,
            TechnicalIndicator.timestamp <= end_date
        ).order_by(TechnicalIndicator.timestamp).limit(limit)
        
        return [TechnicalIndicatorType.from_db_model(model) for model in query]
    
    @strawberry.field
    def predictions(
        self, 
        info: Info, 
        symbol: str,
        model_id: Optional[str] = None,
        prediction_type: Optional[str] = None,
        start_date: datetime = None, 
        end_date: datetime = None,
        limit: int = 100
    ) -> List[PredictionType]:
        """Query model predictions."""
        session = info.context["session"]
        
        query = session.query(Prediction).filter(Prediction.symbol == symbol)
        
        if model_id is not None:
            query = query.filter(Prediction.model_id == model_id)
            
        if prediction_type is not None:
            query = query.filter(Prediction.prediction_type == prediction_type)
            
        if start_date is not None:
            query = query.filter(Prediction.timestamp >= start_date)
            
        if end_date is not None:
            query = query.filter(Prediction.timestamp <= end_date)
            
        query = query.order_by(Prediction.timestamp).limit(limit)
        
        return [PredictionType.from_db_model(model) for model in query]
    
    @strawberry.field
    def data_sync_logs(
        self, 
        info: Info, 
        data_type: Optional[str] = None,
        source: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 100
    ) -> List[DataSyncLogType]:
        """Query data sync logs."""
        session = info.context["session"]
        
        query = session.query(DataSyncLog)
        
        if data_type is not None:
            query = query.filter(DataSyncLog.data_type == data_type)
            
        if source is not None:
            query = query.filter(DataSyncLog.source == source)
            
        if success is not None:
            query = query.filter(DataSyncLog.success == success)
            
        query = query.order_by(DataSyncLog.timestamp.desc()).limit(limit)
        
        return [DataSyncLogType.from_db_model(model) for model in query]
    
    @strawberry.field
    def integrated_data(
        self, 
        info: Info, 
        symbol: str,
        start_date: datetime, 
        end_date: datetime,
        include_planetary: bool = True,
        include_events: bool = True,
        include_indicators: bool = False,
        include_predictions: bool = False,
        limit: int = 100
    ) -> List[IntegratedDataPoint]:
        """Query integrated data combining financial and astrological data."""
        session = info.context["session"]
        
        # Get financial data
        financial_data_query = session.query(FinancialData).filter(
            FinancialData.symbol == symbol,
            FinancialData.timestamp >= start_date,
            FinancialData.timestamp <= end_date
        ).order_by(FinancialData.timestamp).limit(limit)
        
        financial_data = {model.timestamp: FinancialDataType.from_db_model(model) 
                         for model in financial_data_query}
        
        # Get market regimes
        market_regime_query = session.query(MarketRegime).filter(
            MarketRegime.symbol == symbol,
            MarketRegime.timestamp >= start_date,
            MarketRegime.timestamp <= end_date
        ).order_by(MarketRegime.timestamp)
        
        market_regimes = {model.timestamp: MarketRegimeType.from_db_model(model) 
                         for model in market_regime_query}
        
        # Initialize result
        result = []
        
        # Create integrated data points
        for timestamp, financial in financial_data.items():
            data_point = IntegratedDataPoint(
                timestamp=timestamp,
                financial_data=financial,
                market_regime=market_regimes.get(timestamp)
            )
            
            # Add planetary positions if requested
            if include_planetary:
                planetary_query = session.query(PlanetaryPosition).filter(
                    PlanetaryPosition.timestamp == timestamp
                )
                
                data_point.planetary_positions = [
                    PlanetaryPositionType.from_db_model(model) for model in planetary_query
                ]
            
            # Add astrological events if requested
            if include_events:
                # Get events on the same day
                day_start = datetime(timestamp.year, timestamp.month, timestamp.day)
                day_end = day_start + timedelta(days=1)
                
                events_query = session.query(AstrologicalEvent).filter(
                    AstrologicalEvent.timestamp >= day_start,
                    AstrologicalEvent.timestamp < day_end
                )
                
                data_point.astrological_events = [
                    AstrologicalEventType.from_db_model(model) for model in events_query
                ]
            
            # Add technical indicators if requested
            if include_indicators:
                indicators_query = session.query(TechnicalIndicator).filter(
                    TechnicalIndicator.symbol == symbol,
                    TechnicalIndicator.timestamp == timestamp
                )
                
                data_point.technical_indicators = [
                    TechnicalIndicatorType.from_db_model(model) for model in indicators_query
                ]
            
            # Add predictions if requested
            if include_predictions:
                predictions_query = session.query(Prediction).filter(
                    Prediction.symbol == symbol,
                    Prediction.timestamp == timestamp
                )
                
                data_point.predictions = [
                    PredictionType.from_db_model(model) for model in predictions_query
                ]
            
            result.append(data_point)
        
        return result


# Create GraphQL schema
schema = strawberry.Schema(query=Query)

# Create GraphQL router
graphql_app = GraphQLRouter(
    schema,
    context_getter=get_context,
)

# Create FastAPI app
app = FastAPI(title="Cosmic Market Oracle API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GraphQL endpoint
app.include_router(graphql_app, prefix="/graphql")

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Add version endpoint
@app.get("/version")
async def version():
    """Version endpoint."""
    return {"version": "1.0.0", "name": "Cosmic Market Oracle API"}


# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
