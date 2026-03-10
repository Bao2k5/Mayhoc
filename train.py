import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from preprocess import engineer_features
import time

# --- CONFIG ---
DATA_PATH = Path('data/raw/UCI_Credit_Card.csv')
OUTPUT_DIR = Path('models')
REPORT_DIR = Path('reports')
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def train_pipeline():
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Please place 'UCI_Credit_Card.csv' in 'data/raw/' first.")
        return

    print("🚀 Starting Data Mining Training Pipeline...")
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records.")
    
    # 2. Feature Engineering
    print("🛠️ Engineering 52 features...")
    df = engineer_features(df)
    
    # 3. Prepare Features/Target
    target = 'default.payment.next.month'
    if target not in df.columns:
        # Check if column name is slightly different
        target = [c for c in df.columns if 'default' in c.lower()][0]
    
    X = df.drop(['ID', target] if 'ID' in df.columns else [target], axis=1)
    y = df[target]
    
    # Save feature names for the app
    with open('data/processed/feature_names.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    # 4. Split, Scale, and Handle Imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open(OUTPUT_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("⚖️ Applying SMOTE to handle class imbalance...")
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"Resampled training data: {len(X_train_resampled)} records.")

    
    # 5. Model Benchmarking
    models = {
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = []
    best_auc = 0
    best_model = None
    
    print("\n📊 Benchmarking Models:")
    for name, model in models.items():
        start = time.time()
        model.fit(X_train_resampled, y_train_resampled)
        duration = time.time() - start
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "Training Time (s)": duration
        }
        results.append(metrics)
        print(f"✅ {name:15} | AUC: {metrics['ROC-AUC']:.4f} | Time: {duration:.2f}s")
        
        if metrics['ROC-AUC'] > best_auc:
            best_auc = metrics['ROC-AUC']
            best_model = model
            best_name = name

    # 6. Advanced Hyperparameter Tuning for the Best Model (Optuna)
    print("\n🔍 Running Optuna Hyperparameter Tuning for LightGBM...")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300)
        }
        model = LGBMClassifier(**param, random_state=42)
        model.fit(X_train_resampled, y_train_resampled)
        preds = model.predict_proba(X_test_scaled)[:, 1]
        return roc_auc_score(y_test, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10) # 10 trials for demonstration speed
    print(f"✅ Optuna Best AUC: {study.best_value:.4f}")
    
    # Train robust model with best params
    print("🚀 Training Final Tuned Model...")
    tuned_model = LGBMClassifier(**study.best_params, random_state=42)
    tuned_model.fit(X_train_resampled, y_train_resampled)
    best_model = tuned_model
    best_name = "LightGBM (Tuned)"

    # 7. Save Best Model and Reports
    print(f"\n🏆 Final Production Model: {best_name} (AUC: {study.best_value:.4f})")
    with open(OUTPUT_DIR / 'best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    pd.DataFrame(results).to_csv(REPORT_DIR / 'model_comparison_results.csv', index=False)
    
    # Feature Importance (for best model)
    if hasattr(best_model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_

        }).sort_values(by='Importance', ascending=False)
        importance.to_csv(REPORT_DIR / 'feature_importance.csv', index=False)
    
    print("\n✨ Training Complete! Models and reports saved.")

if __name__ == "__main__":
    train_pipeline()
