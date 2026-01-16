# IoT-Based Water Quality Monitoring System

Real-time water quality monitoring system with daily ML risk prediction.

## Features

- **Real-time Monitoring**: ESP32 sends sensor data every 3 hours
- **Daily ML Prediction**: Automatic risk assessment using pre-trained Random Forest model
- **REST API**: FastAPI backend with SQLAlchemy ORM
- **Web Dashboard**: Real-time visualization of sensor data and predictions

## Project Structure
```
water-quality-monitoring/
├── app/                    # Backend application
├── ml_artifacts/           # Pre-trained ML models
├── static/                 # Frontend dashboard
└── data/                   # SQLite database (auto-created)
```

## Prerequisites

- Python 3.10+
- Pre-trained ML models:
  - `water_quality_rf_model.pkl`
  - `label_encoder.pkl`

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd water-quality-monitoring
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place ML artifacts in `ml_artifacts/` directory

## Running the Application

1. Start the backend server:
```bash
uvicorn app.main:app --reload
```

2. Access the dashboard:
```
http://localhost:8000/static/index.html
```

3. API documentation:
```
http://localhost:8000/docs
```

## API Endpoints

### Submit Sensor Reading (ESP32)
```http
POST /api/readings
Content-Type: application/json

{
    "ph": 7.2,
    "tds": 350.5,
    "turbidity": 2.8,
    "temperature": 25.3
}
```

### Get Latest Reading
```http
GET /api/readings/latest
```

### Get Latest Prediction
```http
GET /api/predictions/latest
```

### Get Dashboard Data
```http
GET /api/dashboard
```

## How It Works

### Daily Prediction Pipeline

1. **Data Collection**: ESP32 sends sensor readings every 3 hours
2. **Storage**: Each reading is stored in SQLite database
3. **Aggregation**: At midnight (00:00), system computes daily averages
4. **ML Inference**: Averages are fed to pre-trained Random Forest model
5. **Storage**: Prediction is stored with metadata
6. **Display**: Dashboard shows latest prediction

### Scheduler Architecture

- **APScheduler** runs background task at midnight
- Task aggregates previous day's data (yesterday, not today)
- Uses SQLAlchemy for efficient aggregation queries
- ML model loaded once at startup for performance

### ML Model Loading

Model is loaded **once** during FastAPI startup:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML artifacts at startup
    app.state.ml_service = MLService(...)
    app.state.ml_service.load_artifacts()
    yield
    # Cleanup on shutdown
```

## ESP32 Configuration

Update the following in ESP32 code:
- WiFi SSID and password
- Backend server IP address
- Sensor pin configurations

## Database Schema

### SensorReading
- id, ph, tds, turbidity, temperature, timestamp

### DailyPrediction
- id, date, avg_ph, avg_tds, avg_turbidity, avg_temperature
- prediction, prediction_confidence, reading_count, created_at

## Development

### Testing Daily Prediction Manually
```http
POST /api/predictions/trigger
```

### Viewing Logs
Backend logs show:
- Sensor reading submissions
- Daily prediction execution
- ML inference results

## Production Deployment

For production:
1. Use PostgreSQL instead of SQLite
2. Configure proper CORS origins
3. Add authentication/authorization
4. Use environment variables for configuration
5. Deploy with Gunicorn + Nginx
6. Enable HTTPS

## License

MIT License
