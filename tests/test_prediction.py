import pytest
import sys
import os

# Add root to python path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prediction import AddictionPredictor

@pytest.fixture
def predictor():
    predictor = AddictionPredictor()
    # Mock model/scaler presence bypass since deterministic logic handles it
    predictor.model = "Mocked"
    predictor.scaler = "Mocked"
    return predictor

def test_wdl_user_1(predictor):
    """
    User 1 (WDL Check): (9.41 * 0.6) + (7.76 * 0.4) = 8.75.
    Verify result is High Risk (Extreme).
    """
    wdl = (9.41 * 0.6) + (7.76 * 0.4)
    probs = predictor.calculate_wdl_probabilities(wdl)
    
    prob_low, prob_mod, prob_high = probs
    
    # 8.75 is > 8.1, so it should be the cap value
    assert prob_high == 0.980
    assert prob_mod == 0.015
    assert prob_low == 0.005

def test_wdl_user_3(predictor):
    """
    User 3 (WDL Check): (5.07 * 0.6) + (3.51 * 0.4) = 4.44.
    Verify result is Moderate Risk.
    """
    wdl = (5.07 * 0.6) + (3.51 * 0.4)
    probs = predictor.calculate_wdl_probabilities(wdl)
    
    prob_low, prob_mod, prob_high = probs
    
    # Range 4.1 to 6.0 specifies Moderate 0.70, High 0.20, Low 0.10
    assert pytest.approx(prob_mod) == 0.70
    assert pytest.approx(prob_high) == 0.20
    assert pytest.approx(prob_low) == 0.10

def test_cai_example_calculation(predictor):
    """
    Example Calculation (CAI Check):
    Usage: 2 (Social) + 1 (Gaming) + 1 (Night) = 4.0
    Interaction: (80 Unlocks / 20) + (100 Launches / 25) = 8.0
    Impact: 5 (Sleep) + (30% / 20) = 6.5
    Target CAI: 5.95. Verify result is Moderate Risk (Functional Dependency).
    """
    data = {
        'daily_screen_time': 4.0,  # Unused in CAI, but present for validation
        'social_media_time': 2.0,
        'gaming_time': 1.0,
        'night_usage': 1.0,
        'unlock_frequency': 80.0,
        'app_launches': 100.0,
        'sleep_disturbance': 5.0,
        'weekend_usage_increase': 30.0,
        'notifications_per_day': 0.0,
        'session_duration': 0.0
    }
    
    result = predictor.predict(data)
    
    # CAI = 5.95, which is in [4.5, 6.4]
    assert result['addiction_label'] == 'Moderate'
    assert result['probabilities']['Moderate'] == 0.75
    assert result['probabilities']['High'] == 0.20
    assert result['probabilities']['Low'] == 0.05

def test_edge_case_zero_usage(predictor):
    """
    Edge Case: 0 hrs usage (Verify 100% Low Risk).
    """
    # Deterministic mapping for 0 hrs usage should yield 100% Low Risk
    probs = predictor.calculate_deterministic_probabilities(0.0)
    
    prob_low, prob_mod, prob_high = probs
    
    assert prob_low == 1.00
    assert prob_mod == 0.00
    assert prob_high == 0.00

def test_non_linear_check(predictor):
    """
    Non-Linear Check: High Night Usage (>4) + High Sleep Disturbance (>8) 
    should trigger the High Risk boost regardless of total time.
    """
    data = {
        'daily_screen_time': 5.0,
        'social_media_time': 0.0,
        'gaming_time': 0.0,
        'night_usage': 5.0,      # > 4
        'unlock_frequency': 0.0,
        'app_launches': 0.0,
        'sleep_disturbance': 9.0, # > 8
        'weekend_usage_increase': 0.0,
        'notifications_per_day': 0.0,
        'session_duration': 0.0
    }
    
    # Base CAI would be very low (2.0 + 2.7 = 4.7), but the non-linear check should override to High Risk
    result = predictor.predict(data)
    
    assert result['addiction_label'] == 'High'
    assert result['probabilities']['High'] >= 0.99
