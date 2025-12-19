# 🏦 Bank Customer Churn Prediction

_Machine Learning Project to predict whether a customer will stop subscribing (churn) or not._

---

## 📂 Repository Structure

| File/Folder | Description |
| :--- | :--- |
| `README.md` | Documentation and general project overview |
| `P1M2_riko_fadilah.ipynb` | Main notebook containing EDA, Preprocessing, and Modeling |
| `P1M2_riko_fadilah_inf.ipynb` | Dedicated notebook for model inference/testing |
| `Deployment/` | Directory containing Streamlit web application files |
| `Bank Customer Churn Prediction.csv` | Raw Dataset used |

---

## 🧐 Problem Background

In the banking industry, customer retention is key. This notebook contains the development of a **Machine Learning Model** to help banks predict customer behavior (especially churn risk) based on their profile data. These insights are useful for more strategic business decision-making.

## 🎯 Project Output

The deliverables of this project include:
1.  **Machine Learning Model** that has been trained and evaluated for performance.
2.  **Web Deployment** using Streamlit that allows users to input new customer data and get predictions in *real-time*.

## 📊 Dataset Information

The data used comes from Kaggle:
👉 [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)

*   **Total Data**: 10,000 rows.
*   **Features**: 12 Columns (including demographic, financial, and membership status data).
*   **Target**: `Churn` (Prediction of whether the customer leaves or not).
*   **Data Quality**: Clean (no missing values).

## 🛠️ Methodology

We experimented with various classification algorithms, including:
*   K-Nearest Neighbors (KNN)
*   Support Vector Machine (SVM)
*   Decision Tree
*   Random Forest
*   **Gradient Boosting**

🏆 **Result**: After evaluation and *hyperparameter tuning*, **Gradient Boosting** was selected as the best model for this dataset.

## 💻 Tech Stack

*   **Programming Language**: Python 🐍
*   **IDE**: Visual Studio Code / Jupyter
*   **Key Libraries**:
    *   `pandas`, `numpy`: Data processing
    *   `matplotlib`, `seaborn`: Data visualization
    *   `scikit-learn`, `imblearn`: Modeling & Preprocessing
    *   `joblib`: Model persistence
    *   `streamlit`: Web dashboard creation

## 🚀 Deployment Link

The application can be accessed online via the following link:
🌍 **[HuggingFace Spaces - Bank Customer Churn App](https://huggingface.co/spaces/rfalam/Bank_Customer_Churn_Prediction)**

---

## ⚙️ How to Run (Local)

If you want to run this project on your local machine:

1.  **Install Required Libraries**
    Ensure Python is installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Streamlit Application**
    ```bash
    streamlit run Deployment/streamlit_app.py
    ```

---
*Created by Riko Fadilah Alam*
