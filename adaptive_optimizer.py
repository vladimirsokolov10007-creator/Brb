"""
Adaptive Model Optimization Script
Iteratively improves model based on feedback ROC-AUC scores
Target: ROC-AUC = 1.0
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

class AdaptiveConfig:
    """Configuration that adapts based on performance gap"""
    
    def __init__(self, gap):
        """
        gap: difference between target (1.0) and current ROC-AUC
        """
        self.gap = gap
        self.initial_gap = gap
        
    def get_lr_class_weight(self):
        """Adjust class weights for Logistic Regression"""
        if self.gap > 0.3:
            return 'balanced'
        else:
            return None
    
    def get_rf_params(self):
        """Adapt Random Forest hyperparameters"""
        params = {
            'n_estimators': 150 if self.gap > 0.25 else 100,
            'max_depth': 20 if self.gap > 0.3 else 15,
            'min_samples_split': 3 if self.gap > 0.25 else 5,
            'min_samples_leaf': 1 if self.gap > 0.3 else 2,
            'random_state': 42,
            'n_jobs': -1,
            'max_features': 'sqrt' if self.gap > 0.2 else 'log2'
        }
        return params
    
    def get_gb_params(self):
        """Adapt Gradient Boosting hyperparameters"""
        params = {
            'n_estimators': 200 if self.gap > 0.25 else 100,
            'learning_rate': 0.05 if self.gap > 0.3 else 0.1,
            'max_depth': 6 if self.gap > 0.25 else 5,
            'subsample': 0.8 if self.gap > 0.2 else 1.0,
            'random_state': 42
        }
        return params
    
    def get_xgb_params(self):
        """Adapt XGBoost hyperparameters"""
        params = {
            'n_estimators': 200 if self.gap > 0.25 else 100,
            'learning_rate': 0.05 if self.gap > 0.3 else 0.1,
            'max_depth': 6 if self.gap > 0.25 else 5,
            'subsample': 0.8 if self.gap > 0.2 else 1.0,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
        return params
    
    def get_calibration_method(self):
        """Choose calibration based on gap"""
        if self.gap > 0.2:
            return 'sigmoid'  # Platt scaling - good for large gaps
        else:
            return 'isotonic'  # Isotonic regression - good for fine-tuning
    
    def get_threshold_adjustment(self):
        """Adjust decision threshold"""
        if self.gap > 0.2:
            return 0.4  # Lower threshold to catch more positives
        else:
            return 0.5  # Standard threshold

# ============================================================================
# LOAD DATA AND PREPARE FEATURES
# ============================================================================

def load_and_prepare_data():
    """Load and prepare training and test data"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    train = pd.read_csv('train')
    test = pd.read_csv('test')
    
    # Identify numeric columns
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('retention')
    numeric_cols.remove('id')
    
    # Feature engineering
    train_eng = train.copy()
    test_eng = test.copy()
    
    for df in [train_eng, test_eng]:
        for col in numeric_cols:
            if (df[col] > 0).all():
                df[f'{col}_log'] = np.log1p(df[col])
            if (df[col] >= 0).all():
                df[f'{col}_sqrt'] = np.sqrt(df[col])
            df[f'{col}_squared'] = df[col] ** 2
    
    # Prepare features and target
    X = train_eng.drop(['retention', 'id'], axis=1)
    y = train_eng['retention']
    X_test = test_eng.drop('id', axis=1)
    test_ids = test_eng['id']
    
    # Align columns
    X_test = X_test[X.columns]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Features shape: {X_scaled.shape}")
    print(f"Test shape: {X_test_scaled.shape}")
    
    return X_scaled, y, X_test_scaled, test_ids, X.columns

# ============================================================================
# TRAIN ADAPTIVE MODEL
# ============================================================================

def train_adaptive_models(X_train, X_val, y_train, y_val, config):
    """Train models with adaptive hyperparameters"""
    
    print("\n" + "=" * 80)
    print(f"TRAINING ADAPTIVE MODELS (Gap: {config.gap:.4f})")
    print("=" * 80)
    
    models = {}
    
    # Logistic Regression with adaptive class weights
    print("\n→ Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight=config.get_lr_class_weight(),
        random_state=42
    )
    lr.fit(X_train, y_train)
    lr_pred = lr.predict_proba(X_val)[:, 1]
    lr_auc = roc_auc_score(y_val, lr_pred)
    models['LR'] = {'model': lr, 'auc': lr_auc, 'pred': lr_pred}
    print(f"  ROC-AUC: {lr_auc:.4f}")
    
    # Random Forest with adaptive params
    print("→ Random Forest...")
    rf = RandomForestClassifier(**config.get_rf_params())
    rf.fit(X_train, y_train)
    rf_pred = rf.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_pred)
    models['RF'] = {'model': rf, 'auc': rf_auc, 'pred': rf_pred}
    print(f"  ROC-AUC: {rf_auc:.4f}")
    
    # Gradient Boosting with adaptive params
    print("→ Gradient Boosting...")
    gb = GradientBoostingClassifier(**config.get_gb_params())
    gb.fit(X_train, y_train)
    gb_pred = gb.predict_proba(X_val)[:, 1]
    gb_auc = roc_auc_score(y_val, gb_pred)
    models['GB'] = {'model': gb, 'auc': gb_auc, 'pred': gb_pred}
    print(f"  ROC-AUC: {gb_auc:.4f}")
    
    # XGBoost with adaptive params
    print("→ XGBoost...")
    xgb_model = xgb.XGBClassifier(**config.get_xgb_params())
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_pred)
    models['XGB'] = {'model': xgb_model, 'auc': xgb_auc, 'pred': xgb_pred}
    print(f"  ROC-AUC: {xgb_auc:.4f}")
    
    return models

# ============================================================================
# CREATE ENSEMBLE WITH CALIBRATION
# ============================================================================

def create_calibrated_ensemble(X_train, X_val, y_train, y_val, models, config):
    """Create weighted ensemble with calibration"""
    
    print("\n" + "=" * 80)
    print("CREATING CALIBRATED ENSEMBLE")
    print("=" * 80)
    
    # Create voting ensemble with weights based on validation AUC
    ensemble = VotingClassifier(
        estimators=[
            ('lr', models['LR']['model']),
            ('rf', models['RF']['model']),
            ('gb', models['GB']['model']),
            ('xgb', models['XGB']['model'])
        ],
        voting='soft',
        weights=[
            models['LR']['auc'],
            models['RF']['auc'],
            models['GB']['auc'],
            models['XGB']['auc']
        ]
    )
    
    # Calibrate ensemble
    calibrator = CalibratedClassifierCV(
        ensemble,
        method=config.get_calibration_method(),
        cv=5
    )
    calibrator.fit(X_train, y_train)
    
    # Validate
    ensemble_pred = calibrator.predict_proba(X_val)[:, 1]
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    
    print(f"Ensemble ROC-AUC (before calibration): {roc_auc_score(y_val, ensemble.predict_proba(X_val)[:, 1]):.4f}")
    print(f"Ensemble ROC-AUC (after calibration): {ensemble_auc:.4f}")
    
    return calibrator

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

def generate_predictions(model, X_test, threshold=0.5):
    """Generate predictions with adaptive threshold"""
    proba = model.predict_proba(X_test)[:, 1]
    return proba

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_submission(test_ids, predictions, version, current_auc, gap):
    """Save submission file with metadata"""
    
    submission = pd.DataFrame({
        'id': test_ids,
        'retention': predictions
    })
    
    filename = f'submission_v{version}.csv'
    submission.to_csv(filename, index=False)
    
    # Save metadata
    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'current_auc': float(current_auc),
        'gap_to_target': float(gap),
        'target_auc': 1.0,
        'num_predictions': len(predictions),
        'mean_prediction': float(predictions.mean()),
        'std_prediction': float(predictions.std()),
        'min_prediction': float(predictions.min()),
        'max_prediction': float(predictions.max())
    }
    
    with open(f'submission_v{version}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Submission saved: {filename}")
    print(f"✓ Metadata saved: submission_v{version}_metadata.json")
    
    return filename

# ============================================================================
# LOAD PREVIOUS SUBMISSION
# ============================================================================

def load_previous_submission(version):
    """Load previous submission version"""
    try:
        filename = f'submission_v{version}.csv'
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        return None

# ============================================================================
# MAIN OPTIMIZATION LOOP
# ============================================================================

def main():
    """Main adaptive optimization loop"""
    
    print("\n" + "=" * 80)
    print("ADAPTIVE MODEL OPTIMIZATION SYSTEM")
    print("Target ROC-AUC: 1.0")
    print("=" * 80)
    
    # Load data once
    X_scaled, y, X_test_scaled, test_ids, feature_names = load_and_prepare_data()
    
    # Check for existing submissions
    version = 1
    while load_previous_submission(version) is not None:
        version += 1
    
    print(f"\nStarting from version: {version}")
    
    while True:
        print("\n" + "=" * 80)
        print(f"ITERATION {version}")
        print("=" * 80)
        
        # Get current ROC-AUC from user
        print(f"\nPредыдущая версия: submission_v{version-1}.csv" if version > 1 else "\nПервая версия")
        current_auc = float(input("\nВведите текущий ROC-AUC результат (0-1): "))
        
        # Validate input
        if not 0 <= current_auc <= 1:
            print("❌ Некорректное значение. Введите число от 0 до 1.")
            continue
        
        # Check if target reached
        if current_auc >= 0.99:
            print("\n✓ Достигнута целевая метрика ROC-AUC ≥ 0.99!")
            break
        
        # Calculate gap
        gap = 1.0 - current_auc
        print(f"\n→ Зазор до целевого результата: {gap:.4f}")
        
        # Create adaptive config
        config = AdaptiveConfig(gap)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train adaptive models
        models = train_adaptive_models(X_train, X_val, y_train, y_val, config)
        
        # Create calibrated ensemble
        ensemble = create_calibrated_ensemble(X_train, X_val, y_train, y_val, models, config)
        
        # Generate predictions for test set
        predictions = generate_predictions(ensemble, X_test_scaled, config.get_threshold_adjustment())
        
        # Save submission
        filename = save_submission(test_ids, predictions, version, current_auc, gap)
        
        # Print statistics
        print("\n" + "=" * 80)
        print("PREDICTION STATISTICS")
        print("=" * 80)
        print(f"Mean prediction: {predictions.mean():.4f}")
        print(f"Std prediction: {predictions.std():.4f}")
        print(f"Min prediction: {predictions.min():.4f}")
        print(f"Max prediction: {predictions.max():.4f}")
        print(f"Median prediction: {np.median(predictions):.4f}")
        
        # Summary
        print("\n" + "=" * 80)
        print("ИТОГИ ИТЕРАЦИИ")
        print("=" * 80)
        print(f"Версия: {version}")
        print(f"Текущий ROC-AUC: {current_auc:.4f}")
        print(f"Целевой ROC-AUC: 1.0000")
        print(f"Зазор: {gap:.4f}")
        print(f"Файл сохранён: {filename}")
        
        # Ask to continue
        continue_choice = input("\nПродолжить оптимизацию? (yes/no): ").lower()
        if continue_choice != 'yes':
            print("\n✓ Оптимизация завершена!")
            break
        
        version += 1
    
    print("\n" + "=" * 80)
    print("ПРОЦЕСС ЗАВЕРШЕН")
    print("=" * 80)

if __name__ == '__main__':
    main()
