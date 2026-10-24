import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    confusion_matrix, 
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import joblib

# ==============================================================================
# PREREQUISITE DATA SETUP (Steps 1 to 7 Overview)
# ==============================================================================

# Load dataset
df = pd.read_csv("coffee_shop_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Feature Engineering
df['order_hour'] = df['timestamp'].dt.hour
df['day_of_week_num'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)

df['time_of_day_bucket'] = df['order_hour'].apply(
    lambda h: 'Morning_Rush' if 7 <= h <= 10 else ('Evening_Peak' if 16 <= h <= 21 else 'Off_Peak')
)

df['is_loyalty_member'] = (df['customer_type'] == 'Loyalty Member').astype(int)
df['effective_discount_rate'] = (df['discount_amount'] / df['gross_sales']).fillna(0)

# Target Variable (High-value transaction classification: top 25% quantile)
df['is_high_value_order'] = (df['net_sales'] > df['net_sales'].quantile(0.75)).astype(int)

# One-Hot Encoding
feature_cols = [
    'store_location', 'order_channel', 'payment_method', 'product_category',
    'unit_price', 'quantity', 'gross_sales', 'discount_amount',
    'effective_discount_rate', 'caffeine_mg', 'post_txn_stock_level',
    'order_hour', 'day_of_week_num', 'is_weekend', 'time_of_day_bucket',
    'is_loyalty_member'
]

X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df['is_high_value_order']

# Stratified Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Feature Selection via ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)


# ==============================================================================
# STEP 8: TRAIN THE MODEL
# Initialize and fit the selected algorithm (Random Forest Classifier)
# ==============================================================================

print("--- Step 8: Training Baseline Model ---")
# Instantiate Random Forest with default/baseline parameters
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=12, 
    random_state=42, 
    n_jobs=-1
)

# Train the algorithm on prepared, selected features
rf_model.fit(X_train_selected, y_train)
print("Baseline Random Forest model training complete.\n")


# ==============================================================================
# STEP 9: EVALUATE MODEL PERFORMANCE
# Test model predictions on unseen testing data and analyze metrics
# ==============================================================================

print("--- Step 9: Evaluating Model Performance ---")

# Generate class predictions and class probabilities
y_pred = rf_model.predict(X_test_selected)
y_prob = rf_model.predict_proba(X_test_selected)[:, 1]

# Calculate core metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}\n")

print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot and save Confusion Matrix & ROC Curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.set_theme(style="whitegrid")

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='#8B4513', linewidth=2, label=f'ROC-AUC = {auc:.4f}')
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
axes[1].set_title('ROC Curve', fontsize=12, fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig("model_evaluation_metrics.png", dpi=300)
plt.close()
print("Evaluation plots saved as 'model_evaluation_metrics.png'.\n")


# ==============================================================================
# STEP 10: HYPERPARAMETER TUNING
# Optimize model settings using Grid Search with Cross-Validation
# ==============================================================================

print("--- Step 10: Hyperparameter Tuning via GridSearchCV ---")

# Define hyperparameter search grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [8, 12],
    'min_samples_split': [2, 5]
}

# Run 3-fold cross-validated grid search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train_selected, y_train)

# Extract best tuned estimator
best_rf_model = grid_search.best_estimator_

print(f"Best Hyperparameters Found: {grid_search.best_params_}")
print(f"Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}\n")


# ==============================================================================
# STEP 11: DEPLOY THE MODEL
# Serialize and export model artifacts for integration into APIs/servers
# ==============================================================================

print("--- Step 11: Exporting Model Artifacts for Deployment ---")

# Save tuned model artifact
joblib.dump(best_rf_model, "random_forest_coffee_model.pkl")

# Save fitted feature selector transformer to ensure production data aligns
joblib.dump(selector, "feature_selector.pkl")

print("Saved 'random_forest_coffee_model.pkl' and 'feature_selector.pkl' successfully.\n")


# ==============================================================================
# STEP 12: MONITOR AND MAINTAIN THE MODEL
# Production monitoring architecture reference
# ==============================================================================

def monitor_production_data(new_raw_df, baseline_mean_price=4.75):
    """
    Simulated runtime function to track data drift and concept drift in production.
    """
    print("--- Step 12: Production Data Drift Check ---")
    current_mean_price = new_raw_df['unit_price'].mean()
    drift_threshold = 0.20 # 20% shift trigger
    
    price_change = abs(current_mean_price - baseline_mean_price) / baseline_mean_price
    
    if price_change > drift_threshold:
        print(f"WARNING: Data Drift Detected! Price shifted by {price_change*100:.2f}%. Triggering model retraining pipeline...")
        # Trigger automated pipeline execution or alert engineer
    else:
        print(f"Production inputs healthy. Current mean unit price variation: {price_change*100:.2f}% (Below threshold).")

# Demonstrate drift check using test set sample
sample_batch = df.sample(100, random_state=42)
monitor_production_data(sample_batch)