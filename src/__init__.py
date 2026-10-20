"""
Smartphone Addiction Prediction System
Source Package Initialization
"""

__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

from .data_preprocessing import DataPreprocessor
from .model_training import ModelTrainer
from .prediction import AddictionPredictor

__all__ = [
    'DataPreprocessor',
    'ModelTrainer',
    'AddictionPredictor'
]