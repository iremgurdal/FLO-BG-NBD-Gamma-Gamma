# FLO-BG-NBD-Gamma-Gamma
# FLO CLTV Prediction

This project predicts Customer Lifetime Value (CLTV) for FLO, a leading fashion e-commerce company, using BG/NBD and Gamma-Gamma models.

## 🔍 Business Problem

FLO aims to plan its sales and marketing roadmap based on the potential future value of its current customers.

## 📊 Dataset Info

The dataset includes customer behavior for online and offline transactions during 2020–2021.

Main features:
- Total purchases (online & offline)
- Total spent amounts
- First/last purchase dates
- Channels used

## 📌 Project Steps

1. **Data Preparation**
   - Outlier detection & treatment
   - Feature engineering
   - Date conversion

2. **CLTV Structure**
   - Recency, frequency, T, and monetary calculation
   - Weekly scale

3. **Modeling**
   - BG/NBD: Expected purchase estimation
   - Gamma-Gamma: Average transaction value
   - 6-month CLTV calculation

4. **Segmentation**
   - Customers are segmented into 4 groups (A–D) by CLTV
   - Summary statistics per segment

## 🧰 Technologies Used

- Python
- pandas, matplotlib
- lifetimes (BG/NBD, Gamma-Gamma)
- scikit-learn

## 📂 File

- `FLO_CLTV_Prediction.py`: Full code implementation

---

