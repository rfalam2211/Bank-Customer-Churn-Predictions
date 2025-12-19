import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def run():
    # Create Title
    st.title('Bank Customer Churn Prediction')

    # Create Subheader
    st.subheader('Predicting Customer Churn for Bank')
    
    # Show Dataframe
    # Use relative path that works for both local (if run from parent) and HF deployment structure
    try:
        df = pd.read_csv('Bank Customer Churn Prediction.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('Deployment/Bank Customer Churn Prediction.csv') # Fallback if run from parent
        except:
            try:
                # Fallback for the specific path seen in your HF code
                df = pd.read_csv('./src/Bank Customer Churn Prediction.csv')
            except:
                 st.error("Dataset file not found. Please ensure 'Bank Customer Churn Prediction.csv' is uploaded.")
                 return

    st.dataframe(df)

    # 1. Barplot: Churn vs Non-Churn Distribution
    st.write('### 1. Distribution of Churn vs Non-Churn Customers')
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.countplot(x='churn', data=df, palette='pastel', ax=ax1)
    ax1.set_title('Distribution of Churn vs Non-Churn', fontsize=16)
    ax1.set_xlabel('Churn Status (0 = Retained, 1 = Churned)', fontsize=12)
    st.pyplot(fig1)

    # 2. Histograms: Numerical Feature Analysis
    st.write('### 2. Numerical Features Analysis')
    numerical_features = ['credit_score', 'age', 'balance', 'estimated_salary']
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle('Numerical Feature Distributions', fontsize=20)
    axes = axes.flatten()
    for i, feature in enumerate(numerical_features):
        df[feature].hist(bins=20, ax=axes[i], color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution of {feature.replace("_", " ").title()}', fontsize=12)
        axes[i].set_xlabel(feature.replace("_", " ").title())
        axes[i].set_ylabel('Frequency')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig2)

    # 3. Boxplot: Age Distribution for Churn
    st.write('### 3. Age Distribution Analysis for Churned Customers')
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    sns.boxplot(x='churn', y='age', data=df, palette='coolwarm', ax=ax3)
    ax3.set_title('Age Distribution: Churn vs Non-Churn', fontsize=16)
    ax3.set_xlabel('Churn Status (0 = Retained, 1 = Churned)', fontsize=12)
    ax3.set_ylabel('Age', fontsize=12)
    st.pyplot(fig3)

    # 4. Barplot: Churn Rate by Balance Status
    st.write("### 4. Churn Rate: Customers with Balance vs Zero Balance")
    df['has_balance'] = df['balance'] > 0
    churn_rate_by_balance_status = df.groupby('has_balance')['churn'].mean()
    
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.barplot(x=churn_rate_by_balance_status.index, y=churn_rate_by_balance_status.values, palette='coolwarm', ax=ax4)
    ax4.set_title('Churn Rate: Zero Balance vs. With Balance', fontsize=16)
    ax4.set_xticklabels(['Zero Balance', 'With Balance'])
    ax4.set_xlabel('Balance Status', fontsize=12)
    ax4.set_ylabel('Average Churn Rate', fontsize=12)
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    st.pyplot(fig4)

    # 5. Boxplot: Balance Distribution
    st.write('### 5. Balance Distribution Analysis')
    fig5, ax5 = plt.subplots(figsize=(10, 7))
    sns.boxplot(x='churn', y='balance', data=df, palette='magma', ax=ax5)
    ax5.set_title('Balance Distribution: Churn vs Non-Churn', fontsize=16)
    ax5.set_xlabel('Churn Status (0 = Retained, 1 = Churned)', fontsize=12)
    ax5.set_ylabel('Bank Balance', fontsize=12)   
    st.pyplot(fig5)

    # 6. Countplot: Products Number
    st.write('### 6. Churn Analysis Based on Number of Products')
    fig6, ax6 = plt.subplots(figsize=(10, 7))
    sns.countplot(x='products_number', hue='churn', data=df, palette='plasma', ax=ax6)
    ax6.set_title('Churn Count by Number of Products', fontsize=16)
    ax6.set_xlabel('Number of Products Owned', fontsize=12)
    ax6.set_ylabel('Customer Count', fontsize=12)  
    st.pyplot(fig6)

    # 7. Scatterplot: Age vs Balance
    st.write('### 7. Age and Balance Distribution by Churn Status')
    g = sns.FacetGrid(df, col="churn", hue="churn", height=6, palette='viridis')
    g.map(sns.scatterplot, "age", "balance", alpha=0.6)
    g.fig.suptitle('Interaction of Age and Balance (Split by Churn Status)', y=1.03, fontsize=16)
    g.set_axis_labels("Age", "Bank Balance")
    g.add_legend(title='Churn Status')
    st.pyplot(g.fig)

    # 8. Boxplot: Tenure vs Credit Score
    st.write('### 8. Tenure vs Credit Score Analysis')
    fig8, ax8 = plt.subplots(figsize=(14, 8))
    sns.boxplot(x='tenure', y='credit_score', data=df, palette='coolwarm', ax=ax8)
    ax8.set_title('Credit Score Distribution by Tenure', fontsize=16)
    ax8.set_xlabel('Tenure (Years)', fontsize=12)
    ax8.set_ylabel('Credit Score', fontsize=12)
    st.pyplot(fig8)

    # 9. Heatmap: Correlation
    st.write('### 9. Numerical Features Correlation Heatmap')
    correlation_df = df[numerical_features + ['churn']]
    corr_matrix = correlation_df.corr()
    fig9, ax9 = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax9)
    ax9.set_title('Correlation Matrix: Numerical Features vs Churn', fontsize=16)
    st.pyplot(fig9)

if __name__ == '__main__':
    run()
