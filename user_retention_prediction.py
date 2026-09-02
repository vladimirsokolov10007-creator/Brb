"""
User Retention Prediction Model
Predicts whether a user will return to the app in the next period based on user activity features.
Target metric: ROC-AUC
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
import xgboost as xgb

# ============================================================================
# 1. ЗАГРУЗКА И ИССЛЕДОВАНИЕ ДАННЫХ
# ============================================================================

def load_data():
    """Load train and test data"""
    train = pd.read_csv('train')
    test = pd.read_csv('test')
    sample_submission = pd.read_csv('sample_submission')
    
    return train, test, sample_submission

def explore_data(train, test):
    """Data exploration and quality checks"""
    print("=" * 80)
    print("TRAIN DATA INFO")
    print("=" * 80)
    print(f"Shape: {train.shape}")
    print(f"\nFirst rows:\n{train.head()}")
    print(f"\nData types:\n{train.dtypes}")
    print(f"\nMissing values:\n{train.isnull().sum()}")
    print(f"\nBasic statistics:\n{train.describe()}")
    
    # Check column names
    print(f"\nColumn names: {train.columns.tolist()}")
    
    # Target column
    target_col = 'retention' if 'retention' in train.columns else 'is_returned'
    print(f"Target distribution:\n{train[target_col].value_counts()}")
    print(f"Target proportions:\n{train[target_col].value_counts(normalize=True)}")
    
    print("\n" + "=" * 80)
    print("TEST DATA INFO")
    print("=" * 80)
    print(f"Shape: {test.shape}")
    print(f"\nFirst rows:\n{test.head()}")
    
    return train, test, target_col

# ============================================================================
# 2. ПРЕДОБРАБОТКА И ОЧИСТКА ДАННЫХ
# ============================================================================

def detect_and_handle_anomalies(train, test, target_col):
    """Detect and handle anomalies and outliers"""
    
    # Identify numeric columns (excluding target and ID)
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    
    print("\n" + "=" * 80)
    print("ANOMALY DETECTION")
    print("=" * 80)
    
    # Check for extreme values (using IQR method)
    anomaly_count = 0
    for col in numeric_cols:
        Q1 = train[col].quantile(0.25)
        Q3 = train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        anomalies = train[(train[col] < lower_bound) | (train[col] > upper_bound)]
        if len(anomalies) > 0:
            anomaly_count += 1
            print(f"\n{col}: Found {len(anomalies)} anomalies")
            print(f"  Expected range: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  Anomaly range: [{anomalies[col].min():.2f}, {anomalies[col].max():.2f}]")
    
    if anomaly_count == 0:
        print("No extreme anomalies detected using IQR method (3*IQR threshold)")
    
    return numeric_cols

def preprocess_data(train, test, numeric_cols):
    """Data preprocessing: handle missing values and prepare features"""
    
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    
    # Handle missing values
    train_processed = train.copy()
    test_processed = test.copy()
    
    # Fill NaN with median for numeric columns
    missing_found = False
    for col in numeric_cols:
        if train_processed[col].isnull().sum() > 0:
            median_val = train_processed[col].median()
            train_processed[col].fillna(median_val, inplace=True)
            test_processed[col].fillna(median_val, inplace=True)
            print(f"Filled {col} missing values with median: {median_val:.2f}")
            missing_found = True
    
    if not missing_found:
        print("No missing values found in numeric columns")
    
    return train_processed, test_processed

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

def create_engineered_features(train, test, numeric_cols):
    """Create new features through transformations"""
    
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    
    train_eng = train.copy()
    test_eng = test.copy()
    
    for df in [train_eng, test_eng]:
        for col in numeric_cols:
            # Log transformation for right-skewed features (handling zero values)
            if (df[col] > 0).all():
                df[f'{col}_log'] = np.log1p(df[col])
            
            # Square root transformation
            if (df[col] >= 0).all():
                df[f'{col}_sqrt'] = np.sqrt(df[col])
            
            # Polynomial features for key metrics
            df[f'{col}_squared'] = df[col] ** 2
    
    # Get all engineered features
    engineered_cols = [col for col in train_eng.columns if col not in train.columns]
    print(f"Created {len(engineered_cols)} engineered features")
    print(f"Total features now: {train_eng.shape[1]}")
    
    return train_eng, test_eng, engineered_cols

# ============================================================================
# 4. БАЗОВЫЕ МОДЕЛИ (BASELINE)
# ============================================================================

def train_baseline_models(X_train, X_val, y_train, y_val):
    """Train and evaluate baseline models"""
    
    print("\n" + "=" * 80)
    print("BASELINE MODELS")
    print("=" * 80)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred)
        results[name] = {
            'model': model,
            'auc': auc_score,
            'predictions': y_pred
        }
        print(f"{name}: ROC-AUC = {auc_score:.4f}")
    
    return results

# ============================================================================
# 5. ПРОДВИНУТЫЕ МОДЕЛИ (OPTIMIZATION)
# ============================================================================

def train_advanced_models(X_train, X_val, y_train, y_val):
    """Train and evaluate advanced models"""
    
    print("\n" + "=" * 80)
    print("ADVANCED MODELS (Gradient Boosting)")
    print("=" * 80)
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        ),
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred)
        results[name] = {
            'model': model,
            'auc': auc_score,
            'predictions': y_pred
        }
        print(f"{name}: ROC-AUC = {auc_score:.4f}")
    
    return results

# ============================================================================
# 6. ВЫБОР ЛУЧШЕЙ МОДЕЛИ И ФИНАЛЬНЫЙ ПРОГНОЗ
# ============================================================================

def train_final_model(X_train, y_train, best_model_class, best_params=None):
    """Train final model on all training data"""
    
    if best_params:
        final_model = best_model_class(**best_params)
    else:
        final_model = best_model_class()
    
    final_model.fit(X_train, y_train)
    return final_model

def generate_submission(test_ids, predictions, output_path='submission.csv'):
    """Generate submission file"""
    
    submission = pd.DataFrame({
        'id': test_ids,
        'retention': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Submission shape: {submission.shape}")
    print(f"Submission preview:\n{submission.head()}")
    print(f"\nPrediction statistics:")
    print(f"Mean: {predictions.mean():.4f}")
    print(f"Std: {predictions.std():.4f}")
    print(f"Min: {predictions.min():.4f}")
    print(f"Max: {predictions.max():.4f}")
    
    return submission

# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================

def main():
    """Execute complete ML pipeline"""
    
    # 1. Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    train, test, sample_submission = load_data()
    
    # 2. Explore data
    train, test, target_col = explore_data(train, test)
    
    # 3. Detect anomalies
    numeric_cols = detect_and_handle_anomalies(train, test, target_col)
    
    # 4. Preprocess data
    train, test = preprocess_data(train, test, numeric_cols)
    
    # 5. Feature engineering
    train, test, engineered_cols = create_engineered_features(train, test, numeric_cols)
    
    # 6. Prepare features and target
    X = train.drop([target_col, 'id'], axis=1)
    y = train[target_col]
    X_test = test.drop('id', axis=1)
    test_ids = test['id']
    
    # Align columns
    X_test = X_test[X.columns]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 7. Train baseline models
    baseline_results = train_baseline_models(X_train, X_val, y_train, y_val)
    
    # 8. Train advanced models
    advanced_results = train_advanced_models(X_train, X_val, y_train, y_val)
    
    # 9. Combine all results
    all_results = {**baseline_results, **advanced_results}
    best_model_name = max(all_results, key=lambda x: all_results[x]['auc'])
    best_auc = all_results[best_model_name]['auc']
    
    print("\n" + "=" * 80)
    print("MODEL SELECTION")
    print("=" * 80)
    print(f"Best model: {best_model_name}")
    print(f"Best ROC-AUC on validation: {best_auc:.4f}")
    
    # 10. Train final model on all data
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)
    
    best_model = all_results[best_model_name]['model']
    
    # Retrain on full training data for final predictions
    final_model = train_final_model(X_scaled, y, type(best_model))
    
    # 11. Generate predictions on test set
    y_pred_test = final_model.predict_proba(X_test_scaled)[:, 1]
    
    # 12. Create submission
    submission = generate_submission(test_ids, y_pred_test, 'submission.csv')
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()
