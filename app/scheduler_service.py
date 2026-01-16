"""
Background scheduler for daily ML predictions.
Uses APScheduler to run daily aggregation and prediction task.
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import date, timedelta
from app.database import SessionLocal
from app.crud import compute_daily_aggregates, create_daily_prediction
from app.ml_service import ml_service
import logging

logger = logging.getLogger(__name__)


def run_daily_prediction():
    """
    Daily task: Aggregate previous day's sensor data and make ML prediction.
    
    This function:
    1. Gets yesterday's date
    2. Computes daily averages of sensor parameters
    3. Feeds averages to ML model
    4. Stores prediction in database
    
    Runs automatically at midnight (00:00) every day.
    """
    logger.info("=" * 50)
    logger.info("Starting daily prediction task")
    
    # Use yesterday's data (today's data is incomplete at midnight)
    yesterday = date.today() - timedelta(days=1)
    logger.info(f"Processing date: {yesterday}")
    
    db = SessionLocal()
    try:
        # Step 1: Compute daily aggregates
        aggregates = compute_daily_aggregates(db, yesterday)
        
        if aggregates is None:
            logger.warning(f"No sensor data available for {yesterday}. Skipping prediction.")
            return
        
        # Step 2: Make ML prediction
        prediction_label, confidence = ml_service.predict(
            ph=aggregates['avg_ph'],
            tds=aggregates['avg_tds'],
            turbidity=aggregates['avg_turbidity'],
            temperature=aggregates['avg_temperature']
        )
        
        # Step 3: Store prediction in database
        daily_prediction = create_daily_prediction(
            db=db,
            target_date=yesterday,
            aggregates=aggregates,
            prediction=prediction_label,
            confidence=confidence
        )
        
        logger.info(f"Daily prediction completed successfully:")
        logger.info(f"  Date: {daily_prediction.date}")
        logger.info(f"  Prediction: {daily_prediction.prediction}")
        logger.info(f"  Confidence: {daily_prediction.prediction_confidence:.2%}" if confidence else "  Confidence: N/A")
        logger.info(f"  Based on {daily_prediction.reading_count} readings")
        
    except Exception as e:
        logger.error(f"Error in daily prediction task: {e}", exc_info=True)
    finally:
        db.close()
    
    logger.info("Daily prediction task completed")
    logger.info("=" * 50)


def start_scheduler():
    """
    Initialize and start the background scheduler.
    Called once during FastAPI startup.
    """
    scheduler = BackgroundScheduler()
    
    # Schedule daily prediction at midnight (00:00)
    scheduler.add_job(
        func=run_daily_prediction,
        trigger=CronTrigger(hour=0, minute=0),  # Every day at 00:00
        id='daily_prediction_job',
        name='Daily Water Quality Prediction',
        replace_existing=True
    )
    
    scheduler.start()
    logger.info("Scheduler started - Daily prediction will run at 00:00")
    
    return scheduler


def stop_scheduler(scheduler: BackgroundScheduler):
    """
    Gracefully shutdown scheduler.
    Called during FastAPI shutdown.
    """
    if scheduler:
        scheduler.shutdown()
        logger.info("Scheduler stopped")
