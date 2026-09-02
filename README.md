# User Retention Prediction

Predict whether users will return to the app based on their activity features.

## Quick Start

### 1. Install dependencies (one-time)
```bash
pip install -r requirements.txt
```

### 2. Run the model
```bash
python user_retention_prediction.py
```

That's it! The script will:
- Load data from `train`, `test`, and `sample_submission`
- Analyze and preprocess data
- Engineer features (log, sqrt, polynomial, interactions)
- Train 5 models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
- Select the best model based on ROC-AUC
- Generate `submission.csv` with predictions

## Output

**File:** `submission.csv`
- `user_id`: User identifier
- `is_returned`: Probability (0-1) that user will return

## Model Performance

All models are evaluated on ROC-AUC metric during validation phase.
