#!/usr/bin/env python
# Cosmic Market Oracle - Simplified API Application for Testing

"""
Simplified API Application for the Cosmic Market Oracle.

This module provides a basic API server for testing purposes,
without dependencies on the full astrological engine.
"""

import os
import logging
import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
import time

# Create FastAPI application
app = FastAPI(
    title="Cosmic Market Oracle API (Simplified)",
    description="AI-powered financial forecasting system integrating Vedic astrology with market data",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api")

# Define API models
class MarketPrediction(BaseModel):
    """Model for market prediction results"""
    date: datetime.date
    prediction: float
    confidence: float
    supporting_factors: List[Dict[str, Union[str, float]]]
    planetary_configurations: Dict[str, Union[str, float]]


# Middleware for request logging and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and their timing."""
    start_time = time.time()
    
    # Generate request ID
    request_id = f"{int(start_time * 1000)}"
    
    # Log the request
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    # Process the request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log the response time
        logger.info(f"Request {request_id} completed in {process_time:.3f}s with status {response.status_code}")
        
        # Add timing header to response
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        return response
    except Exception as e:
        # Log the error
        logger.error(f"Request {request_id} failed: {str(e)}")
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )


# Exception handler for all exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint providing API information"""
    logger.info("Root endpoint accessed")
    return {
        "name": "Cosmic Market Oracle API (Simplified)",
        "version": "0.1.0",
        "description": "AI-powered financial forecasting system integrating Vedic astrology with market data",
        "documentation": "/docs",
    }


# Market prediction endpoint
@app.get("/prediction/{date}", response_model=MarketPrediction)
async def get_prediction(date: str):
    """Get market prediction for a specific date"""
    logger.info(f"Getting market prediction for {date}")
    
    try:
        # Parse date
        parsed_date = datetime.datetime.fromisoformat(date)
        
        # This is a placeholder - actual implementation would use the prediction pipeline
        # For now, return a placeholder response
        return {
            "date": parsed_date.date(),
            "prediction": 0.75,  # Positive prediction
            "confidence": 0.85,
            "supporting_factors": [
                {"factor": "Jupiter-Saturn conjunction", "weight": 0.4},
                {"factor": "Venus retrograde ending", "weight": 0.3},
                {"factor": "Moon in favorable nakshatra", "weight": 0.2},
            ],
            "planetary_configurations": {
                "key_aspect": "Jupiter trine Saturn",
                "key_transit": "Mars entering Capricorn",
            }
        }
    except ValueError as e:
        logger.error(f"Invalid date format: {date}")
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting market prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment variables or use defaults
    host = os.environ.get("API_HOST", "localhost")
    port = int(os.environ.get("API_PORT", 8000))
    
    logger.info(f"Starting simplified API server on {host}:{port}")
    uvicorn.run("simple_app:app", host=host, port=port, reload=True)
