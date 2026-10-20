import sys
sys.path.insert(0, '.')
from src.prediction import AddictionPredictor

predictor = AddictionPredictor()

data = {
    'daily_screen_time': 5.0,
    'social_media_time': 2.0,
    'gaming_time': 1.0,
    'notifications_per_day': 50.0,
    'night_usage': 1.0,
    'app_launches': 0.0,
    'sleep_disturbance': 0.0,
    'unlock_frequency': 0.0,
    'weekend_usage_increase': 0.0,
    'session_duration': 0.0
}

res = predictor.predict(data)
print("Prediction for 5 hours Screen Time:", res['probabilities'])
