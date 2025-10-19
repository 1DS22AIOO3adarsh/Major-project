import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("🚀 Initializing model training pipeline...")

# --- 1. Load Your Final Dataset ---
try:
    embeddings = np.load('final_full_paper_embeddings.npy')
    labels_df = pd.read_csv('final_full_paper_labels.csv')
    labels = labels_df['status'].values
except FileNotFoundError:
    print("❌ Error: Dataset files not found. Please run 'process_data.py' first.")
    exit()

print(f"✅ Dataset loaded successfully. Shape of embeddings: {embeddings.shape}, Number of labels: {len(labels)}")

# --- 2. Split Data into Training and Testing Sets ---
# We'll train the model on 80% of the data and test it on the unseen 20%.
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, 
    labels, 
    test_size=0.2, 
    random_state=42, # for reproducibility
    stratify=labels # ensures same class proportion in train/test
)
print("   - Data split into training and testing sets.")

# --- 3. Initialize and Train the XGBoost Model ---
print("⚙️  Training the XGBoost classification model...")
# The model will learn to associate the embedding patterns (X_train) with the labels (y_train).
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False # avoids a deprecation warning
)
model.fit(X_train, y_train)
print("✅ Model training complete.")

# --- 4. Evaluate the Model's Performance ---
print("\n📊 Evaluating model performance on the unseen test set...")
predictions = model.predict(X_test)

# Calculate and print the key metrics
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=['Rejected (0)', 'Accepted (1)']))

# --- 5. Save the Trained Model for Later Use (Optional but Recommended) ---
import joblib
model_filename = 'ai_reviewer_model.joblib'
joblib.dump(model, model_filename)
print(f"\n💾 Model saved to '{model_filename}'. You can now use this file in your web application.")