"""
Database CRUD operations.
"""
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date, datetime, timedelta
from typing import List, Optional
from app.models import SensorReading, DailyPrediction
from app.schemas import SensorReadingCreate
import logging

logger = logging.getLogger(__name__)


def create_sensor_reading(db: Session, reading: SensorReadingCreate) -> SensorReading:
    """
    Create new sensor reading in database.
    
    Args:
        db: Database session
        reading: Sensor data from ESP32
        
    Returns:
        Created SensorReading object
    """
    db_reading = SensorReading(
        ph=reading.ph,
        tds=reading.tds,
        turbidity=reading.turbidity,
        temperature=reading.temperature
    )
    db.add(db_reading)
    db.commit()
    db.refresh(db_reading)
    logger.info(f"Created sensor reading: {db_reading.id}")
    return db_reading


def get_latest_sensor_reading(db: Session) -> Optional[SensorReading]:
    """
    Get the most recent sensor reading.
    
    Returns:
        Latest SensorReading or None
    """
    return db.query(SensorReading).order_by(SensorReading.timestamp.desc()).first()


def get_readings_for_date(db: Session, target_date: date) -> List[SensorReading]:
    """
    Get all sensor readings for a specific date.
    
    Args:
        target_date: Date to query
        
    Returns:
        List of SensorReading objects
    """
    start_datetime = datetime.combine(target_date, datetime.min.time())
    end_datetime = datetime.combine(target_date, datetime.max.time())
    
    readings = db.query(SensorReading).filter(
        SensorReading.timestamp >= start_datetime,
        SensorReading.timestamp <= end_datetime
    ).all()
    
    logger.info(f"Found {len(readings)} readings for {target_date}")
    return readings


def compute_daily_aggregates(db: Session, target_date: date) -> Optional[dict]:
    """
    Compute daily average of sensor parameters for a given date.
    
    Args:
        target_date: Date to aggregate
        
    Returns:
        Dictionary with averages or None if no data
    """
    start_datetime = datetime.combine(target_date, datetime.min.time())
    end_datetime = datetime.combine(target_date, datetime.max.time())
    
    # SQLAlchemy aggregation query
    result = db.query(
        func.avg(SensorReading.ph).label('avg_ph'),
        func.avg(SensorReading.tds).label('avg_tds'),
        func.avg(SensorReading.turbidity).label('avg_turbidity'),
        func.avg(SensorReading.temperature).label('avg_temperature'),
        func.count(SensorReading.id).label('count')
    ).filter(
        SensorReading.timestamp >= start_datetime,
        SensorReading.timestamp <= end_datetime
    ).first()
    
    if result.count == 0:
        logger.warning(f"No readings found for {target_date}")
        return None
    
    aggregates = {
        'avg_ph': float(result.avg_ph),
        'avg_tds': float(result.avg_tds),
        'avg_turbidity': float(result.avg_turbidity),
        'avg_temperature': float(result.avg_temperature),
        'reading_count': result.count
    }
    
    logger.info(f"Daily aggregates for {target_date}: {aggregates}")
    return aggregates


def create_daily_prediction(db: Session, target_date: date, 
                           aggregates: dict, prediction: str, 
                           confidence: Optional[float]) -> DailyPrediction:
    """
    Create or update daily prediction record.
    
    Args:
        target_date: Date of prediction
        aggregates: Daily aggregated sensor values
        prediction: ML prediction label
        confidence: Prediction confidence score
        
    Returns:
        Created/Updated DailyPrediction object
    """
    # Check if prediction already exists for this date
    existing = db.query(DailyPrediction).filter(
        DailyPrediction.date == target_date
    ).first()
    
    if existing:
        # Update existing prediction
        existing.avg_ph = aggregates['avg_ph']
        existing.avg_tds = aggregates['avg_tds']
        existing.avg_turbidity = aggregates['avg_turbidity']
        existing.avg_temperature = aggregates['avg_temperature']
        existing.prediction = prediction
        existing.prediction_confidence = confidence
        existing.reading_count = aggregates['reading_count']
        existing.created_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        logger.info(f"Updated daily prediction for {target_date}")
        return existing
    else:
        # Create new prediction
        db_prediction = DailyPrediction(
            date=target_date,
            avg_ph=aggregates['avg_ph'],
            avg_tds=aggregates['avg_tds'],
            avg_turbidity=aggregates['avg_turbidity'],
            avg_temperature=aggregates['avg_temperature'],
            prediction=prediction,
            prediction_confidence=confidence,
            reading_count=aggregates['reading_count']
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        logger.info(f"Created daily prediction for {target_date}: {prediction}")
        return db_prediction


def get_latest_daily_prediction(db: Session) -> Optional[DailyPrediction]:
    """
    Get the most recent daily prediction.
    
    Returns:
        Latest DailyPrediction or None
    """
    return db.query(DailyPrediction).order_by(DailyPrediction.date.desc()).first()
