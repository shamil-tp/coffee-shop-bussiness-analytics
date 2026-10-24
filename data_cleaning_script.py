import pandas as pd

# 1. Load the two selected datasets
df_coffee = pd.read_excel("Coffee Shop Sales.xlsx")
df_sales = pd.read_csv("sales_data_sample-selected-columns.csv")

# -------------------------------------------------------------------
# CLEAN & HARMONIZE DATASET 1: Coffee Shop Sales.xlsx
# -------------------------------------------------------------------
# Fix column headers
df_coffee.columns = df_coffee.columns.str.strip().str.lower()

# Convert dates and times
df_coffee['transaction_date'] = pd.to_datetime(df_coffee['transaction_date'], errors='coerce')

# Calculate total sales line item
df_coffee['total_sales'] = df_coffee['transaction_qty'] * df_coffee['unit_price']

# Standardize column mapping for merging
df_coffee_clean = df_coffee.rename(columns={
    'transaction_id': 'order_id',
    'transaction_date': 'order_date',
    'transaction_qty': 'quantity',
    'unit_price': 'price_per_unit',
    'product_category': 'category',
    'product_type': 'sub_category',
    'product_detail': 'product_name'
})

# Select essential standardized columns
cols_coffee = ['order_id', 'order_date', 'quantity', 'price_per_unit', 'total_sales', 'store_location', 'product_name', 'category']
df_coffee_clean = df_coffee_clean[[c for c in cols_coffee if c in df_coffee_clean.columns]]
df_coffee_clean['source_dataset'] = 'Coffee Shop Sales'

# -------------------------------------------------------------------
# CLEAN & HARMONIZE DATASET 2: sales_data_sample-selected-columns.csv
# -------------------------------------------------------------------
# Standardize headers
df_sales.columns = df_sales.columns.str.strip().str.upper()

# Handle clipped date strings (### display issues in Excel are preserved as original strings in raw CSVs)
df_sales['ORDERDATE'] = pd.to_datetime(df_sales['ORDERDATE'], errors='coerce')

# Standardize column mapping
df_sales_clean = df_sales.rename(columns={
    'ORDERNUMBER': 'order_id',
    'ORDERDATE': 'order_date',
    'QUANTITYORDERED': 'quantity',
    'PRICEEACH': 'price_per_unit',
    'SALES': 'total_sales',
    'STATUS': 'order_status'
})

# Select essential standardized columns
cols_sales = ['order_id', 'order_date', 'quantity', 'price_per_unit', 'total_sales', 'order_status']
df_sales_clean = df_sales_clean[[c for c in cols_sales if c in df_sales_clean.columns]]
df_sales_clean['source_dataset'] = 'Sales Data Sample'

# -------------------------------------------------------------------
# COMBINE & FINAL CLEANUP
# -------------------------------------------------------------------
# Combine datasets vertically into one master DataFrame
df_merged = pd.concat([df_coffee_clean, df_sales_clean], ignore_index=True)

# Drop any rows where critical numeric identifiers or dates are completely null
df_merged = df_merged.dropna(subset=['order_id', 'order_date'])

# Ensure strict numeric data types
df_merged['quantity'] = pd.to_numeric(df_merged['quantity'], errors='coerce').fillna(0).astype(int)
df_merged['price_per_unit'] = pd.to_numeric(df_merged['price_per_unit'], errors='coerce')
df_merged['total_sales'] = pd.to_numeric(df_merged['total_sales'], errors='coerce')

# Save merged dataset
df_merged.to_csv("unified_sales_dataset.csv", index=False)
print("Successfully cleaned and merged the two datasets!")