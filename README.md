# 📊 Customer Churn Prediction Dashboard

## 📌 Project Overview
This is an end-to-end Data Science project designed to predict customer churn for a telecom company. The goal is to help businesses identify at-risk customers proactively and take data-driven retention actions.

The project covers the complete Data Science lifecycle: **Data Cleaning, EDA, Model Building, Explainability (SHAP), and Deployment via Streamlit.**

## 🚀 Key Features
- **Interactive Dashboard:** Built with Streamlit for a seamless user experience.
- **Exploratory Data Analysis (EDA):** 8+ interactive Plotly charts showing business insights.
- **Model Comparison:** Evaluated Logistic Regression, Random Forest, XGBoost, and LightGBM.
- **Best Model:** LightGBM Classifier (Accuracy: 91%, F1-Score: 0.83).
- **Live Prediction with SHAP:** Not only predicts churn but also explains *why* a customer is at risk using SHAP values.
- **Batch Prediction:** Users can upload a CSV file to predict churn for thousands of customers at once and download the results.

## 🛠️ Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, LightGBM
- **Explainability:** SHAP
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Web Framework:** Streamlit

## 📂 Dataset
- **Source:** Kaggle (Telecom Customer Churn Dataset)
- **Size:** 50,000 records with 33 features.

## ⚙️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/jadhavaakash41074-dev/customer-churn-prediction.git

2. Install requirements:
    pip install requirements.txt