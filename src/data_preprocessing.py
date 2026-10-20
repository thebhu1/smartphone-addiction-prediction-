"""
Data Preprocessing Module - WITH STRATIFIED SPLIT
Critical fix: Use stratified split to maintain class balance in test set
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, filepath):
        """Load dataset from CSV"""
        print(f"\n📂 Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        return df
    
    def check_data_quality(self, df):
        """Check for missing values and data quality"""
        print("\n🔍 Checking data quality...")
        
        # Check missing values
        missing = df.isnull().sum()
        if missing.any():
            print("⚠️  Missing values found:")
            print(missing[missing > 0])
        else:
            print("✅ No missing values")
        
        # Check class distribution
        print("\n📊 Class distribution:")
        class_dist = df['addiction_level'].value_counts().sort_index()
        for level in class_dist.index:
            count = class_dist[level]
            pct = (count / len(df)) * 100
            print(f"   Level {level}: {count:4d} ({pct:5.1f}%)")
        
        return df
    
    def split_data(self, df, test_size=0.20, random_state=42):
        """
        Split data into train and test sets with STRATIFIED sampling
        This is CRITICAL - maintains class distribution in both sets
        """
        print(f"\n✂️  Splitting data (test_size={test_size})...")
        
        # Separate features and target
        X = df.drop(['user_id', 'addiction_level'], axis=1)
        y = df['addiction_level']
        
        self.feature_columns = X.columns.tolist()
        
        # STRATIFIED SPLIT - This is the key!
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # ← CRITICAL: Maintains class balance
        )
        
        print(f"✅ Train set: {len(X_train)} samples")
        print(f"✅ Test set:  {len(X_test)} samples")
        
        # Show class distribution in both sets
        print("\n📊 Train set distribution:")
        train_dist = y_train.value_counts().sort_index()
        for level in train_dist.index:
            count = train_dist[level]
            pct = (count / len(y_train)) * 100
            print(f"   Level {level}: {count:4d} ({pct:5.1f}%)")
        
        print("\n📊 Test set distribution:")
        test_dist = y_test.value_counts().sort_index()
        for level in test_dist.index:
            count = test_dist[level]
            pct = (count / len(y_test)) * 100
            print(f"   Level {level}: {count:4d} ({pct:5.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Standardize features using StandardScaler"""
        print("\n⚖️  Scaling features...")
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("✅ Features scaled (mean=0, std=1)")
        
        return X_train_scaled, X_test_scaled
    
    def save_scaler(self, filepath='models/scaler.pkl'):
        """Save fitted scaler"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)
        print(f"💾 Scaler saved: {filepath}")
    
    def preprocess_pipeline(self, filepath, test_size=0.20, save_processed=True):
        """
        Complete preprocessing pipeline
        
        Args:
            filepath: Path to raw data CSV
            test_size: Proportion for test set (default 20%)
            save_processed: Whether to save processed data
            
        Returns:
            X_train, X_test, y_train, y_test (scaled)
        """
        print("\n" + "="*70)
        print(" "*15 + "DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        # Load data
        df = self.load_data(filepath)
        
        # Quality check
        df = self.check_data_quality(df)
        
        # Split data (STRATIFIED)
        X_train, X_test, y_train, y_test = self.split_data(
            df, 
            test_size=test_size
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Save scaler
        self.save_scaler()
        
        # Optionally save processed data
        if save_processed:
            print("\n💾 Saving processed data...")
            processed_dir = 'data/processed'
            os.makedirs(processed_dir, exist_ok=True)
            
            # Combine for saving
            train_df = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
            train_df['addiction_level'] = y_train.values
            
            test_df = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
            test_df['addiction_level'] = y_test.values
            
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
            combined_df.to_csv(f'{processed_dir}/processed_data.csv', index=False)
            print(f"✅ Saved: {processed_dir}/processed_data.csv")
        
        print("\n" + "="*70)
        print("✅ PREPROCESSING COMPLETE")
        print("="*70)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        filepath='data/raw/smartphone_usage.csv'
    )