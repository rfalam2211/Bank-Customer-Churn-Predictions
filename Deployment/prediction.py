
import streamlit as st
import pandas as pd
import joblib
import numpy as np


def run():

    st.title('Aplikasi Prediksi Customer Churn Bank')
    st.markdown("""
    Aplikasi ini menggunakan model Machine Learning untuk memprediksi apakah seorang pelanggan berisiko akan berhenti berlangganan (churn). 
    Silakan masukkan data pelanggan pada panel di sebelah kiri.
    """)
    st.divider()
   
    @st.cache_resource
    def load_model():
        """Fungsi untuk memuat model pipeline dari file."""
        model_filename = 'final_churn_model.pkl' 
        try:
            pipeline = joblib.load(model_filename)
            return pipeline
        except FileNotFoundError:
            st.error(f"File model '{model_filename}' tidak ditemukan. Pastikan file berada di folder yang sama.")
            return None

    model = load_model()

    # Panel Input di Sidebar
    st.sidebar.header('Masukkan Data Pelanggan')

    def user_inputs():
        """Fungsi untuk membuat semua widget input dari pengguna."""
        credit_score = st.sidebar.slider('Skor Kredit (Credit Score)', 300, 850, 650)
        country = st.sidebar.selectbox('Negara (Country)', ('France', 'Germany', 'Spain'))
        gender = st.sidebar.selectbox('Jenis Kelamin (Gender)', ('Male', 'Female'))
        age = st.sidebar.slider('Usia (Age)', 18, 100, 40)
        tenure = st.sidebar.slider('Masa Jabatan (Tenure)', 0, 10, 5)
        balance = st.sidebar.number_input('Saldo Bank (Balance)', 0.0, 250000.0, 60000.0, step=1000.0)
        products_number = st.sidebar.slider('Jumlah Produk (Products Number)', 1, 4, 1)
        credit_card = st.sidebar.selectbox('Memiliki Kartu Kredit?', (1, 0), format_func=lambda x: 'Ya' if x == 1 else 'Tidak')
        active_member = st.sidebar.selectbox('Anggota Aktif?', (1, 0), format_func=lambda x: 'Ya' if x == 1 else 'Tidak')
        estimated_salary = st.sidebar.number_input('Estimasi Gaji (Estimated Salary)', 0.0, 200000.0, 50000.0, step=1000.0)

        data = {
            'credit_score': credit_score, 'country': country, 'gender': gender, 'age': age,
            'tenure': tenure, 'balance': balance, 'products_number': products_number,
            'credit_card': credit_card, 'active_member': active_member, 'estimated_salary': estimated_salary
        }
        
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_inputs()

    # Proses Prediksi dan Tampilan Hasil
    st.subheader('Data Pelanggan yang Dimasukkan:')
    st.write(input_df)

    if st.button('Prediksi Sekarang'):
        if model is not None:
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            st.subheader('Hasil Prediksi:')
            
            if prediction[0] == 1:
                st.error('**Hasil: Pelanggan ini diprediksi AKAN CHURN**')
                st.write(f"**Probabilitas untuk Churn:** `{prediction_proba[0][1]*100:.2f}%`")
                st.warning("""
                **Rekomendasi:** Pelanggan ini memiliki risiko tinggi untuk pergi. Pertimbangkan untuk menghubungi pelanggan ini dan menawarkan program retensi.
                """)
            else:
                st.success('**Hasil: Pelanggan ini diprediksi TIDAK AKAN CHURN**')
                st.write(f"**Probabilitas untuk Churn:** `{prediction_proba[0][1]*100:.2f}%`")
                st.info("""
                **Rekomendasi:** Pelanggan ini cenderung loyal. Pertahankan kualitas layanan atau tawarkan produk tambahan.
                """)
        else:
            st.error("Model tidak dapat dimuat. Proses prediksi tidak dapat dilanjutkan.")


if __name__ == '__main__':
    run()