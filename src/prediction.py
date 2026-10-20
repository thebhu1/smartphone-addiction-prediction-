"""
Prediction Module
Handles prediction for new data
"""

import joblib
import numpy as np
import pandas as pd
import os

class AddictionPredictor:
    """Class to handle addiction prediction for new users"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'daily_screen_time',
            'social_media_time',
            'gaming_time',
            'notifications_per_day',
            'app_launches',
            'night_usage',
            'sleep_disturbance',
            'unlock_frequency',
            'weekend_usage_increase',
            'session_duration'
        ]
        self.addiction_labels = {
            0: 'Low',
            1: 'Moderate',
            2: 'High'
        }
        self.risk_colors = {
            0: 'green',
            1: 'orange',
            2: 'red'
        }
    
    def load_models(self, model_path='models/addiction_model.pkl', 
                    scaler_path='models/scaler.pkl'):
        """
        Load trained model and scaler
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"✅ Scaler loaded from: {scaler_path}")
    
    def validate_input(self, data):
        """
        Validate input data
        
        Args:
            data: Dictionary with feature values
            
        Returns:
            True if valid, raises ValueError if not
        """
        # Check all required features are present
        for feature in self.feature_columns:
            if feature not in data:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Check data types and ranges
        validations = {
            'daily_screen_time': (0, 24),
            'social_media_time': (0, 24),
            'gaming_time': (0, 24),
            'notifications_per_day': (0, 1000),
            'app_launches': (0, 500),
            'night_usage': (0, 24),
            'sleep_disturbance': (0, 10),
            'unlock_frequency': (0, 500),
            'weekend_usage_increase': (-100, 200),
            'session_duration': (0, 60)
        }
        
        for feature, (min_val, max_val) in validations.items():
            value = data[feature]
            if not isinstance(value, (int, float)):
                raise ValueError(f"{feature} must be a number")
            if not min_val <= value <= max_val:
                raise ValueError(f"{feature} must be between {min_val} and {max_val}")
        
        return True
    
    def prepare_input(self, data):
        """
        Prepare input data for prediction
        
        Args:
            data: Dictionary with feature values
            
        Returns:
            Scaled feature array
        """
        # Create DataFrame with correct column order
        df = pd.DataFrame([data], columns=self.feature_columns)
        
        # Scale features
        scaled_data = self.scaler.transform(df)
        
        return scaled_data

    def calculate_deterministic_probabilities(self, screen_time):
        """
        Calculate probability distribution based on screen time using defined ranges.
        """
        points = [
            (0.0, (1.00, 0.00, 0.00)),
            (1.0, (0.90, 0.07, 0.03)),
            (3.0, (0.90, 0.07, 0.03)),
            (4.0, (0.45, 0.50, 0.05)),
            (5.0, (0.45, 0.50, 0.05)),
            (6.0, (0.05, 0.70, 0.25)),
            (8.0, (0.05, 0.70, 0.25)),
            (9.0, (0.00, 0.12, 0.88)),
            (14.0, (0.00, 0.12, 0.88)),
            (15.0, (0.005, 0.035, 0.960)) # High > 95%, Mod < 4%, Low < 1%
        ]
        
        if screen_time <= 0.0:
            return points[0][1]
        if screen_time >= 15.0:
            return points[-1][1]
            
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i+1]
            
            if x0 <= screen_time <= x1:
                if x0 == x1: return y0
                ratio = (screen_time - x0) / (x1 - x0)
                low = y0[0] + ratio * (y1[0] - y0[0])
                mod = y0[1] + ratio * (y1[1] - y0[1])
                high = y0[2] + ratio * (y1[2] - y0[2])
                total = low + mod + high
                return (low / total, mod / total, high / total)
        return points[-1][1]

    def calculate_wdl_probabilities(self, wdl):
        """
        Calculate probability distribution based on Weighted Digital Load (WDL).
        """
        points = [
            (0.0, (1.00, 0.00, 0.00)),
            (0.1, (0.95, 0.05, 0.00)),
            (2.0, (0.90, 0.05, 0.05)),
            (2.1, (0.55, 0.40, 0.05)),
            (4.0, (0.55, 0.40, 0.05)),
            (4.1, (0.10, 0.70, 0.20)),
            (6.0, (0.10, 0.70, 0.20)),
            (6.1, (0.02, 0.10, 0.88)),
            (8.0, (0.02, 0.08, 0.90)),
            (8.1, (0.005, 0.015, 0.980))
        ]
        
        if wdl <= 0.0:
            return points[0][1]
        if wdl >= 8.1:
            return points[-1][1]
            
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i+1]
            
            if x0 <= wdl <= x1:
                if x0 == x1: return y0
                ratio = (wdl - x0) / (x1 - x0)
                low = y0[0] + ratio * (y1[0] - y0[0])
                mod = y0[1] + ratio * (y1[1] - y0[1])
                high = y0[2] + ratio * (y1[2] - y0[2])
                total = low + mod + high
                return (low / total, mod / total, high / total)
        return points[-1][1]
    
    def calculate_cai_probabilities(self, cai):
        """
        Calculate probability distribution based on Comprehensive Addiction Index (CAI).
        Interpolates gracefully for inputs between ranges.
        """
        # CAI anchors: (CAI_value, (prob_low, prob_moderate, prob_high))
        points = [
            (0.0, (0.95, 0.05, 0.00)),   # Mindful/Balanced
            (2.4, (0.95, 0.05, 0.00)),   # Mindful/Balanced (End)
            (2.5, (0.60, 0.35, 0.05)),   # Average User (Start)
            (4.4, (0.60, 0.35, 0.05)),   # Average User (End)
            (4.5, (0.05, 0.75, 0.20)),   # Functional Dependency (Start)
            (6.4, (0.05, 0.75, 0.20)),   # Functional Dependency (End)
            (6.5, (0.00, 0.15, 0.85)),   # High Habitual (Start)
            (8.5, (0.00, 0.15, 0.85)),   # High Habitual (End)
            (8.6, (0.00, 0.005, 0.995)), # Severe Addiction (Start)
            (20.0, (0.00, 0.005, 0.995)) # Severe Addiction (End)
        ]
        
        if cai <= 0.0:
            return points[0][1]
        if cai >= 8.6:
            return points[-1][1]
            
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i+1]
            
            if x0 <= cai <= x1:
                if x0 == x1:
                    return y0
                
                # Linear interpolation for smooth transitions
                ratio = (cai - x0) / (x1 - x0)
                low = y0[0] + ratio * (y1[0] - y0[0])
                mod = y0[1] + ratio * (y1[1] - y0[1])
                high = y0[2] + ratio * (y1[2] - y0[2])
                
                # Normalize to ensure they sum to exactly 1.0 due to float math
                total = low + mod + high
                return (low / total, mod / total, high / total)
                
        return points[-1][1]

    def predict(self, data):
        """
        Make prediction for given input
        
        Args:
            data: Dictionary with feature values
            
        Returns:
            Dictionary with prediction results
        """
        # Validate input
        self.validate_input(data)
        
        screen_time = float(data.get('daily_screen_time', 0.0))
        
        # Calculate CAI (Comprehensive Addiction Index)
        social_media = float(data.get('social_media_time', 0.0))
        gaming = float(data.get('gaming_time', 0.0))
        night_usage = float(data.get('night_usage', 0.0))
        
        unlocks = float(data.get('unlock_frequency', 0.0))
        launches = float(data.get('app_launches', 0.0))
        
        sleep_disturbance = float(data.get('sleep_disturbance', 0.0))
        weekend_increase = float(data.get('weekend_usage_increase', 0.0))
        
        # Calculate probabilities based on Daily Screen Time (Prompt 1 logic)
        prob_low_st, prob_mod_st, prob_high_st = self.calculate_deterministic_probabilities(screen_time)
        
        # Base probabilities entirely on Prompt 1 definitions
        prob_low, prob_mod, prob_high = prob_low_st, prob_mod_st, prob_high_st
        
        # Context Insight: Severe Non-linear override for extreme behavioral indicators
        # Even with low screen time, severe night usage and sleep disturbance indicate high risk.
        if night_usage > 4 and sleep_disturbance > 8:
            prob_high = max(prob_high, 0.99)
            # Rebalance remaining probability
            rem = 1.0 - prob_high
            prob_mod = rem * 0.8
            prob_low = rem * 0.2
            
            total = prob_low + prob_mod + prob_high
            prob_low, prob_mod, prob_high = prob_low/total, prob_mod/total, prob_high/total
        
        # Determine prediction class based on highest probability
        probs = [prob_low, prob_mod, prob_high]
        prediction = probs.index(max(probs))
            
        # Prepare result
        result = {
            'addiction_level': int(prediction),
            'addiction_label': self.addiction_labels[prediction],
            'risk_color': self.risk_colors[prediction],
            'probabilities': {
                'Low': float(prob_low),
                'Moderate': float(prob_mod),
                'High': float(prob_high)
            },
            'confidence': float(max(probs)) * 100
        }
        
        return result
    
    def get_risk_assessment(self, prediction_result):
        """
        Generate detailed risk assessment
        
        Args:
            prediction_result: Result from predict()
            
        Returns:
            Dictionary with risk assessment details
        """
        level = prediction_result['addiction_level']
        
        assessments = {
            0: {
                'title': '✅ Low Risk',
                'description': 'Your smartphone usage is within healthy limits.',
                'recommendations': [
                    'Continue maintaining balanced usage',
                    'Keep up good sleep hygiene',
                    'Monitor usage patterns regularly'
                ]
            },
            1: {
                'title': '⚠️ Moderate Risk',
                'description': 'Your usage shows some concerning patterns.',
                'recommendations': [
                    'Set daily screen time limits',
                    'Reduce late-night phone usage',
                    'Take regular breaks from phone',
                    'Use app timers and restrictions'
                ]
            },
            2: {
                'title': '🚨 High Risk',
                'description': 'Your usage indicates potential smartphone addiction.',
                'recommendations': [
                    'Seek professional guidance',
                    'Implement strict usage limits',
                    'Turn off non-essential notifications',
                    'Practice digital detox periods',
                    'Engage in offline activities'
                ]
            }
        }
        
        return assessments[level]
    
    def predict_with_assessment(self, data):
        """
        Complete prediction with risk assessment
        
        Args:
            data: Dictionary with feature values
            
        Returns:
            Dictionary with prediction and assessment
        """
        # Get prediction
        prediction = self.predict(data)
        
        # Get assessment
        assessment = self.get_risk_assessment(prediction)
        
        # Combine results
        result = {
            **prediction,
            **assessment
        }
        
        return result
    
    def batch_predict(self, data_list):
        """
        Make predictions for multiple users
        
        Args:
            data_list: List of dictionaries with feature values
            
        Returns:
            List of prediction results
        """
        results = []
        for data in data_list:
            try:
                result = self.predict_with_assessment(data)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return results