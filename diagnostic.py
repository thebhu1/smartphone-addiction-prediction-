"""
Diagnostic Checker - Verify everything is set up correctly
Run this to see what's wrong
"""

import os
import json

print("\n" + "="*70)
print(" "*20 + "🔍 DIAGNOSTIC CHECKER")
print("="*70 + "\n")

# Check 1: Dataset
print("📊 Checking Dataset...")
dataset_path = 'data/raw/smartphone_usage.csv'
if os.path.exists(dataset_path):
    import pandas as pd
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset found: {len(df)} samples")
    print(f"   Distribution: {df['addiction_level'].value_counts().sort_index().to_dict()}")
else:
    print(f"❌ Dataset NOT found at: {dataset_path}")

# Check 2: Models
print("\n🤖 Checking Model Files...")
model_files = [
    'models/addiction_model.pkl',
    'models/scaler.pkl',
    'models/model_metrics.json'
]

for file in model_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"✅ {file} ({size:,} bytes)")
    else:
        print(f"❌ {file} NOT FOUND")

# Check 3: Metrics JSON Content
print("\n📋 Checking Metrics Content...")
metrics_path = 'models/model_metrics.json'
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    print(f"✅ Metrics JSON loaded successfully")
    print(f"\nBest Model: {metrics.get('best_model', 'N/A')}")
    print(f"\nModel Scores:")
    for model_name, scores in metrics.get('models', {}).items():
        print(f"\n{model_name}:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f} ({value*100:.2f}%)")
    
    # Check if all scores are too similar
    print("\n🔍 Checking for 'fake looking' metrics...")
    all_scores = []
    for model_scores in metrics.get('models', {}).values():
        all_scores.extend([
            model_scores['accuracy'],
            model_scores['precision'],
            model_scores['recall'],
            model_scores['f1_score']
        ])
    
    if len(set([round(s, 3) for s in all_scores])) <= 3:
        print("⚠️  WARNING: All metrics are too similar!")
        print("   This looks like synthetic data issues.")
        print("   Run: python generate_dataset.py (with new generator)")
    else:
        print("✅ Metrics show realistic variation")
else:
    print(f"❌ Metrics file NOT found at: {metrics_path}")
    print("   Run: python train_model.py")

# Check 4: Confusion Matrices
print("\n📸 Checking Confusion Matrix Images...")
matrix_files = [
    'models/confusion_matrix_Logistic_Regression.png',
    'models/confusion_matrix_Random_Forest.png',
    'models/confusion_matrix_SVM.png'
]

for file in matrix_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"✅ {file} ({size:,} bytes)")
    else:
        print(f"❌ {file} NOT FOUND")

# Check 5: Flask App Status
print("\n🌐 Flask App Check...")
print("   To see updated metrics:")
print("   1. Stop Flask app (Ctrl+C)")
print("   2. Run: python app.py")
print("   3. Open: http://localhost:5000/model_performance")
print("   4. Hard refresh: Ctrl+Shift+R (or Cmd+Shift+R on Mac)")

print("\n" + "="*70)
print("🎯 RECOMMENDATIONS:")
print("="*70)

if not os.path.exists(metrics_path):
    print("\n1. ❌ Metrics file missing!")
    print("   ACTION: Run 'python train_model.py'")
elif os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    if len(df) < 2000:
        print("\n1. ⚠️  Dataset too small (< 2000 samples)")
        print("   ACTION: Run new 'python generate_dataset.py'")
    else:
        print("\n1. ✅ Dataset size looks good")
        print("   If metrics still look fake, regenerate data")

print("\n2. 🔄 Refresh browser:")
print("   - Press Ctrl+Shift+R (hard refresh)")
print("   - Or clear cache and reload")

print("\n3. 🔄 Restart Flask app:")
print("   - Stop current app (Ctrl+C)")
print("   - Run: python app.py")
print("   - Check: http://localhost:5000/api/model_metrics")

print("\n" + "="*70 + "\n")