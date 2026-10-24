import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 1. Load Dataset
df = pd.read_csv("coffee_shop_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Feature Engineering
df['order_hour'] = df['timestamp'].dt.hour
df['day_of_week_num'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)

df['time_of_day_bucket'] = df['order_hour'].apply(
    lambda h: 'Morning_Rush' if 7 <= h <= 10 else ('Evening_Peak' if 16 <= h <= 21 else 'Off_Peak')
)

df['is_loyalty_member'] = (df['customer_type'] == 'Loyalty Member').astype(int)
df['effective_discount_rate'] = (df['discount_amount'] / df['gross_sales']).fillna(0)

# Target Definition (High-value transaction classification)
df['is_high_value_order'] = (df['net_sales'] > df['net_sales'].quantile(0.75)).astype(int)

# 3. Categorical Feature Encoding
feature_cols = [
    'store_location', 'order_channel', 'payment_method', 'product_category',
    'unit_price', 'quantity', 'gross_sales', 'discount_amount',
    'effective_discount_rate', 'caffeine_mg', 'post_txn_stock_level',
    'order_hour', 'day_of_week_num', 'is_weekend', 'time_of_day_bucket',
    'is_loyalty_member'
]

X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df['is_high_value_order']

# 4. Stratified Data Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. ANOVA Feature Selection
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 6. Fit Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train_selected, y_train)

# 7. Predictions & Evaluation
y_pred = rf_model.predict(X_test_selected)
y_prob = rf_model.predict_proba(X_test_selected)[:, 1]

print("--- Classification Performance Report ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")