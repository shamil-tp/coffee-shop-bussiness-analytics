import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# 1. Load Dataset
df = pd.read_csv("coffee_shop_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. Feature Engineering
df['order_hour'] = df['timestamp'].dt.hour
df['day_of_week_num'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)

# Peak hour bucket
df['time_of_day_bucket'] = df['order_hour'].apply(
    lambda h: 'Morning_Rush' if 7 <= h <= 10 else ('Evening_Peak' if 16 <= h <= 21 else 'Off_Peak')
)

# Behavioral flags
df['is_loyalty_member'] = (df['customer_type'] == 'Loyalty Member').astype(int)
df['effective_discount_rate'] = (df['discount_amount'] / df['gross_sales']).fillna(0)

# Target Variable (High-value order classification)
df['is_high_value_order'] = (df['net_sales'] > df['net_sales'].quantile(0.75)).astype(int)

# 3. Encoding Categorical Features
feature_cols = [
    'store_location', 'order_channel', 'payment_method', 'product_category',
    'unit_price', 'quantity', 'gross_sales', 'discount_amount',
    'effective_discount_rate', 'caffeine_mg', 'post_txn_stock_level',
    'order_hour', 'day_of_week_num', 'is_weekend', 'time_of_day_bucket',
    'is_loyalty_member'
]

X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df['is_high_value_order']

# 4. Stratified Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Feature Selection via ANOVA F-test (Fitted on Train Set Only)
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Convert selected numpy arrays back into pandas DataFrames using selected feature names
selected_cols = X_train.columns[selector.get_support()]

X_train_df = pd.DataFrame(X_train_selected, columns=selected_cols)
X_train_df['is_high_value_order'] = y_train.values
X_train_df.to_csv("coffee_shop_train_selected.csv", index=False)

X_test_df = pd.DataFrame(X_test_selected, columns=selected_cols)
X_test_df['is_high_value_order'] = y_test.values
X_test_df.to_csv("coffee_shop_test_selected.csv", index=False)

print("Saved selected train and test datasets to CSV successfully!")

print(f"Original features: {X_train.shape[1]} -> Selected top features: {X_train_selected.shape[1]}")