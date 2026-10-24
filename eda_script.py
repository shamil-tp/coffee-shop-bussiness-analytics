import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("coffee_shop_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['month_year'] = df['timestamp'].dt.to_period('M')
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()

# Set style
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# 1. Monthly Revenue Trend
monthly_revenue = df.groupby('month_year')['net_sales'].sum().reset_index()
monthly_revenue['month_year_str'] = monthly_revenue['month_year'].astype(str)
sns.lineplot(data=monthly_revenue, x='month_year_str', y='net_sales', marker='o', color='#8B4513', ax=axes[0], linewidth=2.5)
axes[0].set_title('Monthly Net Revenue Trend (2025 - 2026)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Month', fontsize=11)
axes[0].set_ylabel('Total Net Sales ($)', fontsize=11)
axes[0].tick_params(axis='x', rotation=45)

# 2. Revenue by Store Location
store_revenue = df.groupby('store_location')['net_sales'].sum().sort_values(ascending=False).reset_index()
sns.barplot(data=store_revenue, x='net_sales', y='store_location', palette='YlOrBr_r', ax=axes[1])
axes[1].set_title('Total Net Revenue by Store Location', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Total Net Sales ($)', fontsize=11)
axes[1].set_ylabel('Store Location', fontsize=11)

# Annotate store bars
for index, value in enumerate(store_revenue['net_sales']):
    axes[1].text(value + 5000, index, f"${value:,.2f}", va='center', fontweight='bold', fontsize=9)

# 3. Hourly Purchasing Trends (Peak Hours)
hourly_sales = df.groupby('hour')['net_sales'].sum().reset_index()
sns.barplot(data=hourly_sales, x='hour', y='net_sales', color='#D2691E', ax=axes[2])
axes[2].set_title('Revenue by Hour of Day (Peak Purchasing Hours)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Hour of Day (24-Hour Format)', fontsize=11)
axes[2].set_ylabel('Total Net Sales ($)', fontsize=11)

plt.tight_layout()
chart_path = "coffee_shop_eda_charts.png"
plt.savefig(chart_path, dpi=300)
plt.close()

# Calculate Summary Stats for text response
total_revenue = df['net_sales'].sum()
total_transactions = len(df)
avg_ticket = df['net_sales'].mean()
top_product_cat = df.groupby('product_category')['net_sales'].sum().sort_values(ascending=False)
top_items = df.groupby('product_name')['quantity'].sum().sort_values(ascending=False).head(5)

print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Total Transactions: {total_transactions:,}")
print(f"Average Order Value: ${avg_ticket:.2f}")
print("\nRevenue by Store:")
print(store_revenue.to_string(index=False))
print("\nPeak Hours (Top 5 Hours by Sales):")
print(hourly_sales.sort_values('net_sales', ascending=False).head(5).to_string(index=False))
print("\nCategory Performance:")
print(top_product_cat.to_string())