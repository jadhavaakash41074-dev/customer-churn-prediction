# 📊 Telecom Customer Churn Prediction

## 📌 Project Overview

Telecom Customer Churn Prediction is a Machine Learning project that aims to predict whether a customer is likely to leave an Indian Telecom company.

The project uses customer information, service details, billing information, and payment methods to identify customers who are likely to churn.

It also focuses on identifying the major factors responsible for customer churn using Explainable AI (SHAP) and presenting the results through an interactive Streamlit dashboard.

## 🎯 Objectives

- Predict whether a telecom customer will leave the company.
- Identify the major factors responsible for customer churn.
- Perform Exploratory Data Analysis (EDA) to discover business patterns.
- Build and compare multiple Machine Learning models.
- Use SHAP for model explainability.
- Develop an interactive dashboard using Streamlit.
- Provide useful business insights to help reduce customer churn.

## 📁 Dataset Information

- **Dataset Name:** Indian Telecom Customer Churn Dataset
- **Source:** Kaggle
- **Dataset Size:** Approximately 50,000 rows
- **File Location:** `data/raw/telco_churn.csv`
- **Target Variable:** `Churn`

### Target Variable Distribution

| Customer Type | Percentage |
|---|---:|
| No Churn | 73% |
| Churn | 27% |

The dataset contains customer-related information such as tenure, contract type, monthly charges, internet service, and payment method.

*Note: Actual dataset columns and distributions will be verified during preprocessing.*

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.x | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| XGBoost | Advanced classification |
| SHAP | Model explainability |
| Streamlit | Interactive dashboard |
| Jupyter Notebook | Data analysis |
| VS Code | Development environment |
| Git & GitHub | Version control |

## 📂 Project Structure

```text
Telecom-Customer-Churn-Prediction/
│
├── data/
│   └── raw/
│       └── telco_churn.csv
│
├── notebooks/
│   └── 01_EDA.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── trained_model.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

*Some files and folders will be added as the project progresses.*

## 🔍 Project Workflow

### Phase 1: Exploratory Data Analysis (EDA)

- Load and inspect the dataset.
- Check missing values and duplicate records.
- Analyze the target variable distribution.
- Perform univariate and bivariate analysis.
- Create histograms and boxplots.
- Analyze categorical variables against churn.
- Generate a correlation heatmap.
- Identify important business patterns.

### Phase 2: Data Preprocessing

- Handle missing values.
- Remove duplicate records if required.
- Encode categorical variables.
- Scale numerical features where required.
- Split the dataset into training and testing sets.

### Phase 3: Machine Learning Model Development

The following algorithms will be implemented:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

The models will be trained and compared to identify the best-performing model.

### Phase 4: Model Explainability

SHAP (SHapley Additive exPlanations) will be used to:

- Identify important churn-related features.
- Understand how features influence predictions.
- Explain individual customer predictions.
- Generate feature importance visualizations.

### Phase 5: Interactive Dashboard

A Streamlit dashboard will be developed with the following features:

- Customer churn overview.
- Churn distribution charts.
- Customer behavior analysis.
- Important churn factors.
- Model performance metrics.
- SHAP explainability visualizations.
- Customer-level churn prediction form.

### Phase 6: Deployment

- Upload the project to GitHub.
- Prepare the application for deployment.
- Deploy the Streamlit application using Streamlit Community Cloud.

## 📊 Business Questions

This project aims to answer the following business questions:

1. **Tenure:** Are new customers more likely to leave the company?
2. **Contract Type:** Do month-to-month customers churn more than long-term contract customers?
3. **Monthly Charges:** Is there a relationship between monthly charges and customer churn?
4. **Internet Service:** Which internet service category has a higher churn rate?
5. **Payment Method:** Does the payment method influence customer churn?

## 📈 Model Evaluation Metrics

The Machine Learning models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

**Important:** Recall and ROC-AUC will receive special attention because churn represents approximately 27% of the dataset.

The final model will be selected based on its overall performance and ability to identify customers who are likely to churn.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Telecom-Customer-Churn-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Telecom-Customer-Churn-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/01_EDA.ipynb
```

### 7. Run the Streamlit Application

After completing the application development:

```bash
streamlit run app/app.py
```

## 🎯 Expected Outcomes

- A trained Machine Learning model for customer churn prediction.
- Identification of major customer churn factors.
- Comparative analysis of different classification algorithms.
- SHAP-based model explainability.
- An interactive Streamlit dashboard.
- A deployed web application for customer churn prediction.
- Business insights to support customer retention strategies.

## 🚀 Future Scope

- Integrate real-time customer data.
- Improve prediction performance through hyperparameter tuning.
- Add customer segmentation.
- Develop automated customer retention recommendations.
- Integrate the model with telecom customer management systems.
- Monitor model performance and data drift over time.




⭐ If you find this project interesting, consider giving the repository a star!

**Thank you for visiting this project!**
