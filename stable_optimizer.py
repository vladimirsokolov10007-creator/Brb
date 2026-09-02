"""
Stable Adaptive Model Optimization Script
Prevents overfitting with regularization, early stopping, and model stability checks
Target: ROC-AUC = 1.0 (stable improvement only)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import json
from datetime import datetime
import shutil
import os

# ============================================================================
# STABILITY TRACKING
# ============================================================================

class StabilityTracker:
    """Tracks model performance and prevents degradation"""
    
    def __init__(self):
        self.history = []
        self.best_auc = 0.0
        self.best_version = 0
        self.divergence_threshold = 0.05  # 5% drop threshold
        self.load_history()
    
    def load_history(self):
        """Load optimization history from previous runs"""
        try:
            with open('optimization_history.json', 'r') as f:
                self.history = json.load(f)
                if self.history:
                    self.best_auc = max([h['auc'] for h in self.history])
                    self.best_version = max([h['version'] for h in self.history])
        except FileNotFoundError:
            self.history = []
            self.best_auc = 0.0
            self.best_version = 0
    
    def save_history(self):
        """Save optimization history"""
        with open('optimization_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def record_iteration(self, version, current_auc, gap, ensemble_auc):
        """Record iteration results"""
        entry = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'input_auc': float(current_auc),
            'gap': float(gap),
            'ensemble_auc': float(ensemble_auc),
            'best_auc_so_far': float(max(self.best_auc, ensemble_auc))
        }
        self.history.append(entry)
        self.save_history()
        return entry
    
    def check_divergence(self, current_auc):
        """Check if model is diverging (getting worse)"""
        if not self.history:
            return False, 0.0
        
        degradation = self.best_auc - current_auc
        is_diverging = degradation > self.divergence_threshold
        
        return is_diverging, degradation
    
    def get_improvement(self, new_auc):
        """Calculate improvement from best known"""
        if self.best_auc == 0:
            return new_auc
        return new_auc - self.best_auc

# ============================================================================
# ADAPTIVE CONFIGURATION WITH REGULARIZATION
# ============================================================================

class StableAdaptiveConfig:
    """Adaptive configuration with built-in regularization"""
    
    def __init__(self, gap, iteration_num):
        self.gap = gap
        self.iteration_num = iteration_num
        
        # More conservative as iterations increase
        self.regularization_strength = min(0.5, 0.1 * iteration_num)
    
    def get_lr_params(self):
        """Logistic Regression with regularization"""
        params = {
            'max_iter': 1000,
            'C': 1.0 / (1.0 + self.regularization_strength),  # Inverse regularization
            'class_weight': 'balanced' if self.gap > 0.3 else None,
            'random_state': 42,
            'solver': 'lbfgs'
        }
        return params
    
    def get_rf_params(self):
        """Random Forest with regularization"""
        params = {
            'n_estimators': max(50, 150 - 10 * self.iteration_num),
            'max_depth': max(5, 20 - self.iteration_num),
            'min_samples_split': 5 + self.iteration_num,  # Increase to prevent overfitting
            'min_samples_leaf': 2 + int(self.iteration_num / 2),
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'warm_start': True
        }
        return params
    
    def get_gb_params(self):
        """Gradient Boosting with early stopping"""
        params = {
            'n_estimators': max(50, 200 - 20 * self.iteration_num),
            'learning_rate': 0.01 + (0.1 - 0.01) * np.exp(-self.iteration_num / 3),
            'max_depth': max(3, 6 - int(self.iteration_num / 3)),
            'subsample': 0.7 + 0.3 * np.exp(-self.iteration_num / 2),
            'min_samples_leaf': 5 + self.iteration_num,
            'random_state': 42,
            'validation_fraction': 0.1,
            'n_iter_no_change': 20  # Early stopping
        }
        return params
    
    def get_xgb_params(self):
        """XGBoost with regularization"""
        params = {
            'n_estimators': max(50, 200 - 20 * self.iteration_num),
            'learning_rate': 0.01 + (0.1 - 0.01) * np.exp(-self.iteration_num / 3),
            'max_depth': max(3, 6 - int(self.iteration_num / 3)),
            'subsample': 0.7 + 0.3 * np.exp(-self.iteration_num / 2),
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1 * self.iteration_num,  # L1 regularization
            'reg_lambda': 1.0 + 0.1 * self.iteration_num,  # L2 regularization
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
        return params
    
    def get_calibration_method(self):
        """Calibration method selection"""
        if self.gap > 0.2:
            return 'sigmoid'
        else:
            return 'isotonic'

# ============================================================================
# DATA LOADING
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
# MODEL TRAINING WITH CROSS-VALIDATION
# ============================================================================

def train_stable_models(X_train, X_val, y_train, y_val, config):
    """Train models with stability checks"""
    
    print("\n" + "=" * 80)
    print(f"TRAINING STABLE MODELS (Gap: {config.gap:.4f}, Iteration: {config.iteration_num})")
    print("=" * 80)
    
    models = {}
    
    # Logistic Regression
    print("\n→ Logistic Regression...")
    lr = LogisticRegression(**config.get_lr_params())
    lr.fit(X_train, y_train)
    lr_pred = lr.predict_proba(X_val)[:, 1]
    lr_auc = roc_auc_score(y_val, lr_pred)
    
    # Cross-validation score
    lr_cv = cross_val_score(lr, X_train, y_train, cv=5, scoring='roc_auc').mean()
    print(f"  ROC-AUC: {lr_auc:.4f} (CV: {lr_cv:.4f})")
    
    models['LR'] = {'model': lr, 'auc': lr_auc, 'pred': lr_pred, 'cv_auc': lr_cv}
    
    # Random Forest
    print("→ Random Forest...")
    rf = RandomForestClassifier(**config.get_rf_params())
    rf.fit(X_train, y_train)
    rf_pred = rf.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_pred)
    
    rf_cv = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc').mean()
    print(f"  ROC-AUC: {rf_auc:.4f} (CV: {rf_cv:.4f})")
    
    models['RF'] = {'model': rf, 'auc': rf_auc, 'pred': rf_pred, 'cv_auc': rf_cv}
    
    # Gradient Boosting
    print("→ Gradient Boosting...")
    try:
        gb = GradientBoostingClassifier(**config.get_gb_params())
        gb.fit(X_train, y_train)
        gb_pred = gb.predict_proba(X_val)[:, 1]
        gb_auc = roc_auc_score(y_val, gb_pred)
        
        gb_cv = cross_val_score(gb, X_train, y_train, cv=5, scoring='roc_auc').mean()
        print(f"  ROC-AUC: {gb_auc:.4f} (CV: {gb_cv:.4f})")
        
        models['GB'] = {'model': gb, 'auc': gb_auc, 'pred': gb_pred, 'cv_auc': gb_cv}
    except Exception as e:
        print(f"  Warning: {e}")
    
    # XGBoost
    print("→ XGBoost...")
    try:
        xgb_model = xgb.XGBClassifier(**config.get_xgb_params())
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
        xgb_auc = roc_auc_score(y_val, xgb_pred)
        
        xgb_cv = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='roc_auc').mean()
        print(f"  ROC-AUC: {xgb_auc:.4f} (CV: {xgb_cv:.4f})")
        
        models['XGB'] = {'model': xgb_model, 'auc': xgb_auc, 'pred': xgb_pred, 'cv_auc': xgb_cv}
    except Exception as e:
        print(f"  Warning: {e}")
    
    return models

# ============================================================================
# CREATE BLENDED ENSEMBLE
# ============================================================================

def create_stable_ensemble(X_train, X_val, y_train, y_val, models, config, old_ensemble=None):
    """Create ensemble with model blending for stability"""
    
    print("\n" + "=" * 80)
    print("CREATING STABLE ENSEMBLE")
    print("=" * 80)
    
    # Create voting ensemble
    estimators = [(name, m['model']) for name, m in models.items()]
    weights = [m['cv_auc'] for _, m in models.items()]  # Use CV scores as weights
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights
    )
    
    ensemble.fit(X_train, y_train)
    ensemble_pred_before = ensemble.predict_proba(X_val)[:, 1]
    ensemble_auc_before = roc_auc_score(y_val, ensemble_pred_before)
    
    print(f"New Ensemble ROC-AUC (before calibration): {ensemble_auc_before:.4f}")
    
    # Calibrate
    calibrator = CalibratedClassifierCV(
        ensemble,
        method=config.get_calibration_method(),
        cv=5
    )
    calibrator.fit(X_train, y_train)
    
    ensemble_pred = calibrator.predict_proba(X_val)[:, 1]
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    
    print(f"New Ensemble ROC-AUC (after calibration): {ensemble_auc:.4f}")
    
    # Blend with old ensemble if available
    if old_ensemble is not None:
        print("\nBlending with previous model...")
        old_pred = old_ensemble.predict_proba(X_val)[:, 1]
        old_auc = roc_auc_score(y_val, old_pred)
        
        # 70% new, 30% old (conservative blending)
        blended_pred = 0.7 * ensemble_pred + 0.3 * old_pred
        blended_auc = roc_auc_score(y_val, blended_pred)
        
        print(f"Old Ensemble ROC-AUC: {old_auc:.4f}")
        print(f"Blended Ensemble ROC-AUC: {blended_auc:.4f}")
        
        # Use blended if better, else use new
        if blended_auc >= ensemble_auc:
            print("✓ Using blended model (more stable)")
            # Return wrapper that blends predictions
            class BlendedEnsemble:
                def __init__(self, new_ens, old_ens):
                    self.new_ens = new_ens
                    self.old_ens = old_ens
                
                def predict_proba(self, X):
                    new_pred = self.new_ens.predict_proba(X)
                    old_pred = self.old_ens.predict_proba(X)
                    blended = 0.7 * new_pred + 0.3 * old_pred
                    return blended
            
            return BlendedEnsemble(calibrator, old_ensemble), blended_auc
        else:
            print("✓ Using new model (significant improvement)")
            return calibrator, ensemble_auc
    
    return calibrator, ensemble_auc

# ============================================================================
# LOAD PREVIOUS MODEL
# ============================================================================

def load_previous_model(version):
    """Load previous calibrated ensemble if exists"""
    try:
        import pickle
        filename = f'model_v{version}.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Could not load previous model: {e}")
    return None

def save_model(model, version):
    """Save calibrated ensemble"""
    try:
        import pickle
        filename = f'model_v{version}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model saved: {filename}")
    except Exception as e:
        print(f"Could not save model: {e}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_submission(test_ids, predictions, version, current_auc, gap, ensemble_auc):
    """Save submission with metadata"""
    
    submission = pd.DataFrame({
        'id': test_ids,
        'retention': predictions
    })
    
    filename = f'submission_v{version}.csv'
    submission.to_csv(filename, index=False)
    
    metadata = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'input_auc': float(current_auc),
        'ensemble_auc': float(ensemble_auc),
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
    
    print(f"✓ Submission saved: {filename}")
    
    return filename

# ============================================================================
# ROLLBACK MECHANISM
# ============================================================================

def rollback_if_needed(current_auc, best_auc, version, tracker):
    """Rollback to previous version if divergence detected"""
    
    is_diverging, degradation = tracker.check_divergence(current_auc)
    
    if is_diverging:
        print("\n" + "!" * 80)
        print("⚠️  DIVERGENCE DETECTED!")
        print(f"Current AUC ({current_auc:.4f}) degraded from best ({best_auc:.4f})")
        print(f"Degradation: {degradation:.4f} ({degradation/best_auc*100:.2f}%)")
        print("!" * 80)
        
        prev_version = version - 1
        if prev_version > 0:
            print(f"\n→ Rolling back to submission_v{prev_version}.csv")
            
            # Copy previous submission
            shutil.copy(f'submission_v{prev_version}.csv', f'submission_v{version}.csv')
            shutil.copy(f'submission_v{prev_version}_metadata.json', f'submission_v{version}_metadata.json')
            
            print(f"✓ Rolled back to v{prev_version}")
            return True
    
    return False

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    """Main stable optimization loop"""
    
    print("\n" + "=" * 80)
    print("STABLE ADAPTIVE MODEL OPTIMIZATION")
    print("With anti-overfitting safeguards")
    print("Target ROC-AUC: 1.0 (stable only)")
    print("=" * 80)
    
    # Load data
    X_scaled, y, X_test_scaled, test_ids, feature_names = load_and_prepare_data()
    
    # Initialize stability tracker
    tracker = StabilityTracker()
    version = tracker.best_version + 1
    
    print(f"\nStarting from version: {version}")
    if tracker.best_auc > 0:
        print(f"Best AUC so far: {tracker.best_auc:.4f}")
    
    # Load previous model
    old_ensemble = load_previous_model(version - 1) if version > 1 else None
    
    while True:
        print("\n" + "=" * 80)
        print(f"ITERATION {version}")
        print("=" * 80)
        
        # Get feedback
        print(f"\nПредыдущая версия: submission_v{version-1}.csv" if version > 1 else "\nПервая версия")
        current_auc = float(input("\nВведите текущий ROC-AUC результат (0-1): "))
        
        if not 0 <= current_auc <= 1:
            print("❌ Некорректное значение.")
            continue
        
        if current_auc >= 0.99:
            print("\n✓ Достигнута целевая метрика!")
            break
        
        # Check for divergence and rollback if needed
        if version > 1 and tracker.best_auc > 0:
            if rollback_if_needed(current_auc, tracker.best_auc, version, tracker):
                version += 1
                continue
        
        gap = 1.0 - current_auc
        print(f"→ Gap to target: {gap:.4f}")
        
        # Create config
        config = StableAdaptiveConfig(gap, version - 1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        models = train_stable_models(X_train, X_val, y_train, y_val, config)
        
        # Create ensemble
        ensemble, ensemble_auc = create_stable_ensemble(
            X_train, X_val, y_train, y_val, models, config, old_ensemble
        )
        
        # Generate predictions
        predictions = ensemble.predict_proba(X_test_scaled)[:, 1]
        
        # Save
        save_submission(test_ids, predictions, version, current_auc, gap, ensemble_auc)
        save_model(ensemble, version)
        
        # Record
        tracker.record_iteration(version, current_auc, gap, ensemble_auc)
        
        # Statistics
        print("\n" + "=" * 80)
        print("ITERATION SUMMARY")
        print("=" * 80)
        print(f"Input ROC-AUC: {current_auc:.4f}")
        print(f"Ensemble ROC-AUC: {ensemble_auc:.4f}")
        print(f"Improvement: {tracker.get_improvement(ensemble_auc):.4f}")
        print(f"Mean prediction: {predictions.mean():.4f}")
        print(f"Std prediction: {predictions.std():.4f}")
        
        # Update old ensemble for next iteration
        old_ensemble = ensemble
        
        # Continue?
        if input("\nПродолжить? (yes/no): ").lower() != 'yes':
            break
        
        version += 1
    
    print("\n" + "=" * 80)
    print("✓ OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best version: submission_v{tracker.best_version}.csv")
    print(f"Best AUC: {tracker.best_auc:.4f}")

if __name__ == '__main__':
    main()
