"""
Model Training Module - FIXED
Uses MACRO averaging instead of weighted to show real differences
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """Class to handle model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = {}
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        print("\n📊 Training Logistic Regression...")
        
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=0.8  # Slightly regularized for realistic performance
        )
        model.fit(X_train, y_train)
        
        self.models['Logistic Regression'] = model
        print("✅ Logistic Regression trained")
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        print("\n🌲 Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train, y_train)
        
        self.models['Random Forest'] = model
        print("✅ Random Forest trained")
        
        return model
    
    def train_svm(self, X_train, y_train):
        """Train SVM model"""
        print("\n🎯 Training SVM...")
        
        model = SVC(
            kernel='rbf',
            random_state=42,
            probability=True,
            C=1.2,
            gamma='scale'
        )
        model.fit(X_train, y_train)
        
        self.models['SVM'] = model
        print("✅ SVM trained")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate model with MACRO averaging
        This shows TRUE differences between metrics!
        """
        print(f"\n📈 Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics with MACRO average (treats all classes equally)
        # This is the KEY FIX - macro shows real performance differences
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Also calculate per-class metrics to show variation
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        self.results[model_name] = results
        
        # Print overall results
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f} (macro)")
        print(f"Recall:    {recall:.4f} (macro)")
        print(f"F1-Score:  {f1:.4f} (macro)")
        
        # Print per-class breakdown
        print(f"\nPer-Class Performance:")
        for i, class_name in enumerate(['Low', 'Moderate', 'High']):
            print(f"  {class_name:8s}: P={precision_per_class[i]:.3f}, R={recall_per_class[i]:.3f}, F1={f1_per_class[i]:.3f}")
        
        return results
    
    def print_classification_report(self, model_name, y_test):
        """Print detailed classification report"""
        print(f"\n📋 Classification Report - {model_name}")
        print("="*60)
        
        y_pred = self.results[model_name]['predictions']
        
        target_names = ['Low', 'Moderate', 'High']
        report = classification_report(y_test, y_pred, target_names=target_names)
        print(report)
    
    def plot_confusion_matrix(self, model_name, save_path='models'):
        """Plot and save confusion matrix"""
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Low', 'Moderate', 'High'],
            yticklabels=['Low', 'Moderate', 'High']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, f'confusion_matrix_{model_name.replace(" ", "_")}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"💾 Confusion matrix saved: {filepath}")
        plt.close()
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON")
        print("="*60)
        
        comparison = []
        for model_name, results in self.results.items():
            comparison.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}"
            })
        
        import pandas as pd
        df_comparison = pd.DataFrame(comparison)
        print(df_comparison.to_string(index=False))
        
        # Find best model based on F1-Score
        best_model_name = max(self.results.items(), key=lambda x: x[1]['f1_score'])[0]
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1-Score: {self.results[best_model_name]['f1_score']:.4f}")
        print("="*60)
        
        return df_comparison
    
    def plot_feature_importance(self, feature_names, save_path='models'):
        """Plot feature importance for Random Forest"""
        if 'Random Forest' not in self.models:
            print("⚠️  Random Forest not trained, skipping feature importance")
            return
        
        model = self.models['Random Forest']
        importances = model.feature_importances_
        
        # Store for JSON export
        self.feature_importance = {name: float(imp) for name, imp in zip(feature_names, importances)}
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance - Random Forest')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        # Save plot
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, 'feature_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"💾 Feature importance plot saved: {filepath}")
        plt.close()
        
        # Print top features
        print("\n🔝 Top 5 Important Features:")
        for i in range(min(5, len(importances))):
            idx = indices[i]
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    def save_best_model(self, filepath='models/addiction_model.pkl'):
        """Save the best model to file"""
        if self.best_model is None:
            print("⚠️  No best model selected. Run compare_models() first.")
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.best_model, filepath)
        print(f"\n💾 Best model ({self.best_model_name}) saved to: {filepath}")
        
        # Also save metrics as JSON
        self.save_metrics_json()
    
    def save_metrics_json(self, filepath='models/model_metrics.json'):
        """Save model metrics as JSON for web dashboard"""
        import json
        
        metrics_data = {
            'models': {},
            'best_model': self.best_model_name,
            'feature_importance': self.feature_importance
        }
        
        # Save all model metrics
        for model_name, results in self.results.items():
            metrics_data['models'][model_name] = {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score'])
            }
        
        # Save to file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"💾 Model metrics saved to: {filepath}")
    
    def load_model(self, filepath='models/addiction_model.pkl'):
        """Load trained model from file"""
        self.best_model = joblib.load(filepath)
        print(f"📂 Model loaded from: {filepath}")
        return self.best_model
    
    def train_all_models(self, X_train, y_train, X_test, y_test, feature_names):
        """Complete training pipeline for all models"""
        print("\n" + "="*60)
        print("🤖 MODEL TRAINING PIPELINE")
        print("="*60)
        
        # Train all models
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_svm(X_train, y_train)
        
        # Evaluate all models
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
            self.print_classification_report(model_name, y_test)
            self.plot_confusion_matrix(model_name)
        
        # Compare models
        self.compare_models()
        
        # Plot feature importance
        self.plot_feature_importance(feature_names)
        
        # Save best model
        self.save_best_model()
        
        print("\n" + "="*60)
        print("✅ MODEL TRAINING COMPLETE")
        print("="*60)