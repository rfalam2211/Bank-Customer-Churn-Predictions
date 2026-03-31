# 🏦 Bank Customer Churn Prediction

> **Machine Learning project** — Predicting whether a bank customer will churn (leave) or stay, using classification algorithms with a full end-to-end pipeline: EDA → Preprocessing → Modeling → Hyperparameter Tuning → Deployment.

---

## 📂 Repository Structure

| File / Folder | Description |
| :--- | :--- |
| `P1M2_riko_fadilah.ipynb` | Main notebook: EDA, Feature Engineering, Preprocessing, Modeling & Evaluation |
| `P1M2_riko_fadilah_inf.ipynb` | Inference notebook for testing the final model on new data |
| `Deployment/` | Streamlit web app (EDA dashboard + real-time prediction) |
| `Bank Customer Churn Prediction.csv` | Raw dataset (10,000 rows × 12 columns) |
| `final_churn_model.pkl` | Trained & serialized Gradient Boosting pipeline |
| `Dockerfile` | Docker configuration for containerized deployment |
| `requirements.txt` | Python dependencies |

---

## 🧐 Problem Background

In the banking industry, **customer retention** is critical. Acquiring new customers costs significantly more than retaining existing ones. This project develops a **Machine Learning classification model** to predict customer churn behavior based on demographic, financial, and membership data — enabling banks to proactively implement retention strategies for at-risk customers.

---

## 📊 Dataset

**Source:** [Bank Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)

| Attribute | Detail |
| :--- | :--- |
| **Rows** | 10,000 |
| **Features** | 12 columns (credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary, churn) |
| **Target** | `churn` — binary (0 = Retained, 1 = Churned) |
| **Class Distribution** | ~79.6 % Retained, ~20.4 % Churned (imbalanced) |
| **Data Quality** | Clean — no missing values |

---

## 🔍 Key EDA Insights

1. **Class Imbalance** — Only ~20 % of customers churned, requiring SMOTE oversampling during training.
2. **Age is the strongest churn predictor** — Churned customers are significantly older (mean ≈ 45) vs retained (mean ≈ 37).
3. **Customers with a balance tend to churn more** than zero-balance customers.
4. **Customers with 3–4 products** show much higher churn rates.
5. **Correlations between features are weak**, meaning all features can be included without multicollinearity concerns.
6. **Geography matters** — Germany has notably higher churn rates than France and Spain.

---

## 🛠️ Methodology & Pipeline

### Preprocessing
- Dropped irrelevant column (`customer_id`).
- **StandardScaler** for numerical features (`credit_score`, `age`, `balance`, `estimated_salary`).
- **OneHotEncoder** for categorical features (`country`, `gender`).
- **SMOTE** oversampling to handle class imbalance.
- All preprocessing steps are encapsulated in an **imblearn Pipeline** for reproducibility.

### Models Evaluated
| Algorithm | Notes |
| :--- | :--- |
| K-Nearest Neighbors (KNN) | Baseline comparison |
| Support Vector Machine (SVM) | With probability estimates |
| Decision Tree | Interpretable baseline |
| Random Forest | Ensemble method |
| **Gradient Boosting** ✅ | **Best performer — selected as final model** |

### Model Selection & Tuning
- Evaluated using **classification reports**, **confusion matrices**, **ROC curves & AUC scores**, and **cross-validation**.
- **GridSearchCV** was used for hyperparameter tuning on the best model.
- 🏆 **Gradient Boosting** was selected as the final model after evaluation and tuning.

---

## 🎯 Project Deliverables

1. **Trained ML Pipeline** (`final_churn_model.pkl`) — A serialized pipeline containing preprocessing + SMOTE + Gradient Boosting, ready for inference.
2. **Interactive Web App** — A Streamlit application with two pages:
   - **EDA Page** — Visualizes data distributions, churn rates, correlations (9 chart types).
   - **Prediction Page** — Input customer data via sidebar widgets and get real-time churn predictions with probability scores and retention recommendations.
3. **Docker Support** — `Dockerfile` for containerized deployment.

---

## 💻 Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Language** | Python 🐍 |
| **Data Processing** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` |
| **ML & Preprocessing** | `scikit-learn`, `imblearn` (SMOTE, Pipeline) |
| **Model Persistence** | `joblib` |
| **Web Dashboard** | `streamlit` |
| **Containerization** | Docker |

---

## 🚀 Live Demo

🌍 **[HuggingFace Spaces — Bank Customer Churn Prediction App](https://huggingface.co/spaces/rfalam/Bank_Customer_Churn_Prediction)**

---

## ⚙️ How to Run Locally

1. **Clone & install dependencies**
   ```bash
   git clone https://github.com/rfalam2211/Bank-Customer-Churn-Predictions.git
   cd Bank-Customer-Churn-Predictions
   pip install -r requirements.txt
   ```

2. **Launch the Streamlit app**
   ```bash
   streamlit run Deployment/streamlit_app.py
   ```

3. **Or use Docker**
   ```bash
   docker build -t churn-app .
   docker run -p 8501:8501 churn-app
   ```

---

*Created by **Riko Fadilah Alam***
