from sqlalchemy import Column, Integer, Float, String, DateTime, Date
from datetime import datetime
from app.database import Base


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, index=True)
    ph = Column(Float, nullable=False)
    tds = Column(Float, nullable=False)  
    turbidity = Column(Float, nullable=False)  
    temperature = Column(Float, nullable=False)  
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self):
        return f"<SensorReading(id={self.id}, timestamp={self.timestamp})>"


class DailyPrediction(Base):
    __tablename__ = "daily_predictions"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, nullable=False, index=True)
    
    avg_ph = Column(Float, nullable=False)
    avg_tds = Column(Float, nullable=False)
    avg_turbidity = Column(Float, nullable=False)
    avg_temperature = Column(Float, nullable=False)
    
    prediction = Column(String, nullable=False) 
    prediction_confidence = Column(Float, nullable=True)  
    
    reading_count = Column(Integer, nullable=False)  # How many readings used
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<DailyPrediction(date={self.date}, prediction={self.prediction})>"
