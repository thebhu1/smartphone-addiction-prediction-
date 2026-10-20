"""
Main Training Script
Runs complete ML pipeline: preprocessing → training → evaluation
"""

import sys
import os
import json

# Add src to path
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer

def save_metrics_to_json(trainer, output_path='models/model_metrics.json'):
    """
    Save model metrics to JSON file for web dashboard
    
    Args:
        trainer: Trained ModelTrainer instance
        output_path: Path to save JSON file
    """
    metrics_data = {
        'models': {},
        'best_model': trainer.best_model_name,
        'feature_importance': {}
    }
    
    # Add metrics for each model
    for model_name, metrics in trainer.results.items():
        metrics_data['models'][model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score'])
        }
    
    # Add feature importance if available
    if hasattr(trainer, 'feature_importance') and trainer.feature_importance:
        metrics_data['feature_importance'] = {
            k: float(v) for k, v in trainer.feature_importance.items()
        }
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    print(f"   ✅ {output_path}")

def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print(" " * 10 + "📱 SMARTPHONE ADDICTION PREDICTION")
    print(" " * 15 + "Machine Learning Pipeline")
    print("="*70 + "\n")
    
    # Check if dataset exists
    dataset_path = 'data/raw/smartphone_usage.csv'
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found!")
        print(f"   Expected: {dataset_path}")
        print("\n💡 Run this command first:")
        print("   python generate_dataset.py")
        return
    
    try:
        # Step 1: Data Preprocessing
        print("STEP 1: DATA PREPROCESSING")
        print("-" * 70)
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            filepath=dataset_path,
            save_processed=True
        )
        
        # Step 2: Model Training
        print("\n\nSTEP 2: MODEL TRAINING & EVALUATION")
        print("-" * 70)
        
        trainer = ModelTrainer()
        trainer.train_all_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=preprocessor.feature_columns
        )
        
        # Step 3: Save Metrics for Web Dashboard
        print("\n\nSTEP 3: SAVING METRICS FOR WEB DASHBOARD")
        print("-" * 70)
        save_metrics_to_json(trainer)
        
        # Step 4: Summary
        print("\n\n" + "="*70)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print("\n📊 Summary:")
        print(f"   - Best Model: {trainer.best_model_name}")
        print(f"   - F1-Score: {trainer.results[trainer.best_model_name]['f1_score']:.4f}")
        print(f"   - Accuracy: {trainer.results[trainer.best_model_name]['accuracy']:.4f}")
        
        print("\n📁 Generated Files:")
        print("   ✅ models/addiction_model.pkl")
        print("   ✅ models/scaler.pkl")
        print("   ✅ models/model_metrics.json")
        print("   ✅ models/confusion_matrix_*.png")
        print("   ✅ models/feature_importance.png")
        print("   ✅ data/processed/processed_data.csv")
        
        print("\n🚀 Next Steps:")
        print("   1. Run: python app.py")
        print("   2. Open browser: http://localhost:5000")
        print("   3. Start making predictions!")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())