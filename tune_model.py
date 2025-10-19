import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("🚀 Initializing hyperparameter tuning pipeline...")

# --- 1. Load Your Final Dataset ---
try:
    embeddings = np.load('final_full_paper_embeddings.npy')
    labels_df = pd.read_csv('final_full_paper_labels.csv')
    labels = labels_df['status'].values
except FileNotFoundError:
    print("❌ Error: Dataset files not found. Please run 'process_data.py' first.")
    exit()

print(f"✅ Dataset loaded. Shape: {embeddings.shape}")

# --- 2. Define the Parameter Grid ---
# These are the parameter ranges we want to test.
# Start with smaller ranges to tune faster, then expand if needed.
param_grid = {
    'max_depth': [3, 4, 5],             # Depth of trees
    'learning_rate': [0.1, 0.05, 0.01], # Step size shrinkage
    'n_estimators': [100, 150, 200],    # Number of trees
    'subsample': [0.8, 0.9, 1.0],       # Fraction of samples used per tree
    'colsample_bytree': [0.8, 0.9, 1.0],# Fraction of features used per tree
    'gamma': [0, 0.1, 0.2]              # Minimum loss reduction for split
}

# --- 3. Initialize XGBoost and GridSearchCV ---
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# Use Stratified K-Fold for cross-validation to maintain class balance
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("⚙️  Starting GridSearchCV (this might take a while)...")
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy', # We want to optimize for accuracy
    cv=cv,              # Cross-validation strategy
    n_jobs=-1,          # Use all available CPU cores
    verbose=1           # Show progress
)

# --- 4. Run the Grid Search ---
grid_search.fit(embeddings, labels)

# --- 5. Print Best Parameters and Score ---
print("\n✅ GridSearchCV complete.")
print(f"Best Parameters found: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_ * 100:.2f}%")

# --- 6. Train Final Model with Best Parameters (Optional but Recommended) ---
best_model = grid_search.best_estimator_ # The model trained with the best parameters
print("\nRetraining final model on the entire dataset with best parameters...")
best_model.fit(embeddings, labels) # Retrain on ALL data

# --- 7. Save the Tuned Model ---
tuned_model_filename = 'ai_reviewer_model_tuned.joblib'
joblib.dump(best_model, tuned_model_filename)
print(f"\n💾 Tuned model saved to '{tuned_model_filename}'.")

# You can optionally evaluate this final model on a held-out test set if you split one earlier
# (This example uses cross-validation scores as the primary evaluation)