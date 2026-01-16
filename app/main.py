"""
FastAPI application entry point.
Handles:
- Application startup/shutdown lifecycle
- ML model loading
- Background scheduler initialization
- API endpoints
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from app.database import engine, Base, get_db
from app.models import SensorReading, DailyPrediction
from app.schemas import (
    SensorReadingCreate, 
    SensorReadingResponse, 
    DailyPredictionResponse,
    DashboardResponse
)
from app.ml_service import MLService, ml_service as global_ml_service
from app.scheduler_service import start_scheduler, stop_scheduler
from app import crud

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    Handles startup and shutdown events.
    """
    # ========== STARTUP ==========
    logger.info("Starting application...")
    
    # 1. Create database tables
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    # 2. Load ML artifacts
    logger.info("Loading ML artifacts...")
    global global_ml_service
    app.state.ml_service = MLService(
        model_path="ml_artifacts/water_quality_rf_model.pkl",
        encoder_path="ml_artifacts/label_encoder.pkl"
    )
    app.state.ml_service.load_artifacts()
    
    # Make ML service globally accessible
    import app.ml_service
    app.ml_service.ml_service = app.state.ml_service
    
    # 3. Start background scheduler
    logger.info("Starting background scheduler...")
    global scheduler
    scheduler = start_scheduler()
    
    logger.info("Application startup complete!")
    logger.info("=" * 50)
    
    yield  # Application runs here
    
    # ========== SHUTDOWN ==========
    logger.info("Shutting down application...")
    
    # Stop scheduler
    if scheduler:
        stop_scheduler(scheduler)
    
    logger.info("Application shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="Water Quality Monitoring System",
    description="IoT-based water quality monitoring with ML risk prediction",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (for frontend development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "message": "Water Quality Monitoring System API",
        "status": "running",
        "endpoints": {
            "submit_reading": "POST /api/readings",
            "latest_reading": "GET /api/readings/latest",
            "latest_prediction": "GET /api/predictions/latest",
            "dashboard": "GET /api/dashboard"
        }
    }


@app.post("/api/readings", response_model=SensorReadingResponse, 
          status_code=status.HTTP_201_CREATED)
async def submit_sensor_reading(
    reading: SensorReadingCreate,
    db: Session = Depends(get_db)
):
    """
    ESP32 endpoint: Submit sensor reading.
    Called every 3 hours by ESP32.
    
    Example payload:
    {
        "ph": 7.2,
        "tds": 350.5,
        "turbidity": 2.8,
        "temperature": 25.3
    }
    """
    try:
        db_reading = crud.create_sensor_reading(db, reading)
        logger.info(f"Received sensor reading from ESP32: ID={db_reading.id}")
        return db_reading
    except Exception as e:
        logger.error(f"Error saving sensor reading: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save sensor reading"
        )


@app.get("/api/readings/latest", response_model=SensorReadingResponse)
async def get_latest_reading(db: Session = Depends(get_db)):
    """
    Get the most recent sensor reading.
    Used by frontend dashboard.
    """
    reading = crud.get_latest_sensor_reading(db)
    if reading is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No sensor readings found"
        )
    return reading


@app.get("/api/predictions/latest", response_model=DailyPredictionResponse)
async def get_latest_prediction(db: Session = Depends(get_db)):
    """
    Get the most recent daily prediction.
    Used by frontend dashboard.
    """
    prediction = crud.get_latest_daily_prediction(db)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No predictions found"
        )
    return prediction


@app.get("/api/dashboard", response_model=DashboardResponse)
async def get_dashboard_data(db: Session = Depends(get_db)):
    """
    Get all data needed for dashboard in single request.
    Returns latest reading + latest prediction.
    """
    latest_reading = crud.get_latest_sensor_reading(db)
    latest_prediction = crud.get_latest_daily_prediction(db)
    
    return DashboardResponse(
        latest_reading=latest_reading,
        latest_prediction=latest_prediction
    )


@app.post("/api/predictions/trigger")
async def trigger_prediction_manually(db: Session = Depends(get_db)):
    """
    Manual trigger for daily prediction (for testing).
    In production, this runs automatically via scheduler.
    """
    from app.scheduler_service import run_daily_prediction
    try:
        run_daily_prediction()
        return {"message": "Daily prediction triggered successfully"}
    except Exception as e:
        logger.error(f"Error triggering prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
