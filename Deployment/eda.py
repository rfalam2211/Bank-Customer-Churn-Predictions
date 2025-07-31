import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

def run():
    # Membuat title
    st.title('Bank Customer Churn Prediction')

    # Membuat subheader
    st.subheader ('Page ini isinya EDA dari Bank Customer Churn Prediction')
    
    # Menampilkan Dataframe
    df = pd.read_csv('./src/Bank Customer Churn Prediction.csv')
    st.dataframe(df)

    # Membuat barplot Distribusi Nasabah Churn vs Non-Churn
    st.write('### Distribusi Nasabah Churn vs Non-Churn')
    fig = plt.figure(figsize=(10,8))
    sns.countplot(x='churn', data=df, palette='pastel')
    plt.title('Distribusi Pelanggan Churn vs Non-Churn', fontsize=16)
    plt.xlabel('Status Churn (0 = Bertahan, 1 = Churn)', fontsize=12)
    st.pyplot(fig)


    # Membuat barplot Analisis Fitur Numerikal
    st.write(('### Analisis kolom numerikal'))
    numerical_features = ['credit_score', 'age', 'balance', 'estimated_salary']
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle('Distribusi Fitur Numerik', fontsize=20)
    axes = axes.flatten()
    for i, feature in enumerate(numerical_features):
        df[feature].hist(bins=20, ax=axes[i], color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribusi {feature.replace("_", " ").title()}', fontsize=12)
        axes[i].set_xlabel(feature.replace("_", " ").title())
        axes[i].set_ylabel('Frekuensi')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig2)


    # Membuat boxplot Analisis Distribusi usia pelanggan yang churn
    st.write(('### Analisis distribusi usia pada pelanggan yang churn'))
    fig3 = plt.figure(figsize=(10, 7))
    sns.boxplot(x='churn', y='age', data=df, palette='coolwarm')
    plt.title('Distribusi Usia antara Pelanggan Churn dan Non-Churn', fontsize=16)
    plt.xlabel('Status Churn (0 = Bertahan, 1 = Churn)', fontsize=12)
    plt.ylabel('Usia', fontsize=12)
    st.pyplot(fig3)

    # Melihat perbandingan Analisis Distribusi usia pelanggan yang churn
    st.write("### Tingkat Churn: Nasabah dengan Saldo vs Tanpa Saldo")
    df['has_balance'] = df['balance'] > 0
    churn_rate_by_balance_status = df.groupby('has_balance')['churn'].mean()
    
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.barplot(x=churn_rate_by_balance_status.index, y=churn_rate_by_balance_status.values, palette='coolwarm', ax=ax4)
    ax4.set_title('Tingkat Churn: Pelanggan Saldo Nol vs. Saldo Terisi', fontsize=16)
    ax4.set_xticklabels(['Saldo Nol', 'Saldo Terisi'])
    ax4.set_xlabel('Status Saldo', fontsize=12)
    ax4.set_ylabel('Tingkat Churn Rata-rata', fontsize=12)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    st.pyplot(fig4)

    # Membuat boxplot distribusi saldo antara nasabah Churn dan non-Churn
    st.write(('### Analisis distribusi saldo antara nasabah Churn dan non-Churn'))
    fig5 = plt.figure(figsize=(10, 7))
    sns.boxplot(x='churn', y='balance', data=df, palette='magma')
    plt.title('Distribusi Saldo antara Nasabah Churn dan Non-Churn', fontsize=16)
    plt.xlabel('Status Churn (0 = Bertahan, 1 = Churn)', fontsize=12)
    plt.ylabel('Saldo Bank', fontsize=12)   
    st.pyplot(fig5)

    # Membuat sebaran data distribusi tingkat churn berdasarkan jumlah produk bank yang dimiliki
    st.write(('### Analisis usia dan saldo berdasarkan stasus churn'))
    fig6 =plt.figure(figsize=(10, 7))
    sns.countplot(x='products_number', hue='churn', data=df, palette='plasma')
    plt.title('Tingkat Churn Berdasarkan Jumlah Produk Bank', fontsize=16)
    plt.xlabel('Jumlah Produk yang Dimiliki', fontsize=12)
    plt.ylabel('Jumlah Pelanggan', fontsize=12)  
    st.pyplot(fig6)

    # Membuat sebaran data usia dan saldo berdasarkan stasus churn
    st.write(('### Analisis distribusi tingkat churn berdasarkan jumlah produk bank yang dimiliki'))
    g = sns.FacetGrid(df, col="churn", hue="churn", height=6, palette='viridis')
    g.map(sns.scatterplot, "age", "balance", alpha=0.6)
    g.fig.suptitle('Interaksi Usia dan Saldo (Dipisahkan Berdasarkan Status Churn)', y=1.03, fontsize=16)
    g.set_axis_labels("Usia", "Saldo Bank")
    g.add_legend(title='Status Churn')
    st.pyplot(g.fig)

    # Analisis boxplot fitur tenure dengan credit_score
    st.write(('### Analisis fitur tenure dengan credit_score'))
    fig8, ax8 = plt.subplots(figsize=(14, 8))
    sns.boxplot(x='tenure', y='credit_score', data=df, palette='coolwarm', ax=ax8)
    ax8.set_title('Distribusi Skor Kredit Berdasarkan Lama Berlangganan (Tenure)', fontsize=16)
    ax8.set_xlabel('Lama Berlangganan (Tahun)', fontsize=12)
    ax8.set_ylabel('Skor Kredit', fontsize=12)
    st.pyplot(fig8)

    # Heatmap korelasi dari kolom numerikal
    st.write('### 9. Heatmap Korelasi Fitur Numerikal')
    correlation_df = df[numerical_features + ['churn']]
    corr_matrix = correlation_df.corr()
    fig9, ax9 = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax9)
    ax9.set_title('Matriks Korelasi Antar Fitur Numerik dan Churn', fontsize=16)
    st.pyplot(fig9)

if __name__ == '__main__':
    run()