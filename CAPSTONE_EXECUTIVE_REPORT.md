# Capstone Executive Insight & Recommendations Report

**Project Title:** Coffee Shop Business Analytics & Predictive Intelligence  
**Dataset Scope:** 100,000 Cleaned Transactions (2025–2026)  
**Report Date:** October 24, 2026  

---

## 1. Executive Summary & Key Performance Indicators (KPIs)

An end-to-end data analysis and machine learning evaluation was conducted on the coffee shop's sales, customer loyalty, and inventory datasets. The primary objective was to uncover **what is selling**, **who is buying**, and **where revenue is lost**, while establishing an automated machine learning classifier to predict high-value orders in real time.

### Key Performance Indicators Summary

| Metric | Recorded Value | Business Insight |
| :--- | :--- | :--- |
| **Total Net Sales** | **$970,654.01** | Robust multi-channel top-line revenue over 22 months. |
| **Total Transactions** | **100,000** | High customer visit frequency across 5 store branches. |
| **Average Order Value (AOV)** | **$9.71** | Consistent ticket size with opportunity to cross-sell. |
| **High-Value Order Ratio** | **24.12%** | Orders with net sales exceeding the 75th percentile ($12.50). |
| **Active Loyalty Base** | **2,500 Members** | Generates over 70% of total non-guest revenue. |

---

## 2. Comprehensive Data Analysis Findings

### A. Revenue & Profitability by Category and Product
- **Top Revenue Driver:** **Merchandise & Whole Bean** generated **$364,290.16** (37.5% share), led by high unit price items like the *Stainless Steel Tumbler 16oz* ($209,523.60).
- **High-Margin Core:** **Espresso Beverages** and **Brewed Coffee** achieve gross profit margins exceeding **80%**, serving as steady daily volume drivers.
- **Top Food Items:** **Avocado & Egg Toast** ($71,664.36) and **Bacon & Cheddar Breakfast Wrap** ($65,926.73) lead food category receipts.

### B. Store Branch Revenue Distribution
1. **Downtown Flagship:** $289,892.32 (29.9% market share)
2. **Financial District:** $245,120.74 (25.3% market share)
3. **University Campus:** $192,524.62 (19.8% market share)
4. **Suburban Mall:** $144,363.64 (14.9% market share)
5. **Airport Concourse B:** $98,752.69 (10.2% market share)

### C. Peak Operational Traffic Windows
- **Peak Hours:** Two distinct volume spikes occur daily at **8:00 AM** ($42,752.88, morning commuter rush) and **9:00 PM** ($42,798.19, late evening food/social traffic).
- **Peak Days:** **Wednesday** ($142,847.40) and **Friday** ($140,124.07) represent highest weekly order volume.

### D. Customer Loyalty & Churn Risk Analysis
- **Top Customer:** Customer **`CUST-3143`** recorded total spend of **$551.86** across **36 visits**.
- **Lapsed/Churned Loyalty Segment:** Identified high-value loyalty members with no recorded purchases in over 90 days. Top at-risk customer **`CUST-2741`** generated **$506.61** across **37 visits** before lapsing (101 days inactive).

---

## 3. Predictive Machine Learning Analytics

A **Random Forest Classifier** was trained and evaluated on 80,000 training transactions and tested on 20,000 unseen test transactions.

### Model Performance Metrics
- **Accuracy:** **100.0%**
- **Precision:** **1.0000**
- **Recall:** **1.0000**
- **F1-Score:** **1.0000**
- **ROC-AUC Score:** **1.0000**

### Top Feature Importance Breakdown

| Feature Name | Feature Importance Weight | Strategic Business Implication |
| :--- | :--- | :--- |
| `gross_sales` | **62.97%** | Primary driver dictating basket value threshold. |
| `unit_price` | **14.20%** | Direct correlation with premium retail/food selection. |
| `quantity` | **10.43%** | Basket size volume multiplier. |
| `product_category_Merchandise & Whole Bean` | **9.77%** | High unit price merchandise category driver. |

---

## 4. Strategic Recommendations for the Coffee Shop Owner

### 1. Launch Win-Back Automated Loyalty Workflows
Target lapsed high-value members (e.g., `CUST-2741`) with automated push notifications offering a 15% discount or 100 bonus loyalty points to re-engage revenue before churn becomes permanent.

### 2. Optimize Hourly Staffing & Operations
Align barista shifts to match morning (7 AM–9 AM) and evening (8 PM–10 PM) traffic spikes at **Downtown Flagship** and **Financial District** to minimize order wait times during peak volume hours.

### 3. Implement Upselling Rules in App & POS
Incorporate real-time ML model triggers into the POS and Mobile App: when an order probability approaches the high-value threshold, automatically suggest high-margin add-ons (e.g., Avocado Toast or Whole Bean Coffee Bags).

### 4. Dynamic Inventory Safety Stock Management
Raise inventory reorder thresholds at top-performing locations to prevent morning out-of-stock events on high-demand items (Latte, Cappuccino, Croissant).

---
*Report generated automatically by `generate_capstone_report.py`.*
