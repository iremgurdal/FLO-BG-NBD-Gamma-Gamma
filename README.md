# FLO CLTV Prediction

This project aims to calculate the **Customer Lifetime Value (CLTV)** for FLO, a fashion e-commerce company, using the **BG/NBD** and **Gamma-Gamma** models from the `lifetimes` Python library.

---

## 🔍 Business Problem

FLO wants to identify which customers bring more long-term value to the company and create personalized marketing strategies accordingly. For this, CLTV prediction is needed using historical purchasing behavior.

---

## 📦 Dataset Overview

The dataset contains customer behavior from 2020–2021 who purchased both online and offline.

**Key columns include:**

- `order_num_total_ever_online` / `offline`: total purchases
- `customer_value_total_ever_online` / `offline`: total amount spent
- `first_order_date` / `last_order_date`: dates of transactions
- `order_channel`: platform (Android, iOS, etc.)

---

## ⚙️ Project Workflow

### 1. Data Preparation
- Missing values handled
- Outliers capped using IQR method
- Features engineered for total orders and total value
- Date columns converted

### 2. Feature Engineering for CLTV
- Weekly recency, tenure (T), frequency and average monetary calculated
- Dataframe created for modeling

### 3. CLTV Modeling
- **BG/NBD Model** → Predict expected number of purchases
- **Gamma-Gamma Model** → Predict average revenue per purchase
- Combined to compute 6-month CLTV

### 4. Segmentation
- Customers divided into 4 segments (`A`, `B`, `C`, `D`) by CLTV quartiles
- Mean statistics calculated per segment

---

## 💡 Key Insights

- Segment A contains the highest value customers
- Segment D represents the lowest CLTV group
- Marketing strategies can prioritize A and B segments for retention and upsell campaigns

---

## 🧪 How to Run

1. Clone the repo or download files
2. Install required libraries:
```bash
pip install pandas matplotlib lifetimes scikit-learn
