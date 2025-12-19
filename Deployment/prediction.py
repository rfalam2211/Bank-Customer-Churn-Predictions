
import streamlit as st
import pandas as pd
import joblib
import numpy as np


def run():


    st.title('Bank Customer Churn Prediction App')
    st.markdown("""
    This app uses a Machine Learning model to predict whether a customer is at risk of churning.
    Please input customer data in the sidebar.
    """)
    st.divider()
   
    @st.cache_resource
    def load_model():
        """Function to load the pipeline model from file."""
        model_filename = 'final_churn_model.pkl' 
        try:
            pipeline = joblib.load(model_filename)
            return pipeline
        except FileNotFoundError:
            st.error(f"Model file '{model_filename}' not found. Ensure the file is in the same folder.")
            return None

    model = load_model()

    # Input Panel in Sidebar
    st.sidebar.header('Customer Data Input')

    def user_inputs():
        """Function to capture all user inputs."""
        credit_score = st.sidebar.slider('Credit Score', 300, 850, 650)
        country = st.sidebar.selectbox('Country', ('France', 'Germany', 'Spain'))
        gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
        age = st.sidebar.slider('Age', 18, 100, 40)
        tenure = st.sidebar.slider('Tenure (Years)', 0, 10, 5)
        balance = st.sidebar.number_input('Balance', 0.0, 250000.0, 60000.0, step=1000.0)
        products_number = st.sidebar.slider('Number of Products', 1, 4, 1)
        credit_card = st.sidebar.selectbox('Has Credit Card?', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No')
        active_member = st.sidebar.selectbox('Is Active Member?', (1, 0), format_func=lambda x: 'Yes' if x == 1 else 'No')
        estimated_salary = st.sidebar.number_input('Estimated Salary', 0.0, 200000.0, 50000.0, step=1000.0)

        data = {
            'credit_score': credit_score, 'country': country, 'gender': gender, 'age': age,
            'tenure': tenure, 'balance': balance, 'products_number': products_number,
            'credit_card': credit_card, 'active_member': active_member, 'estimated_salary': estimated_salary
        }
        
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_inputs()

    # Prediction Process and Result Display
    st.subheader('Feature for Prediction:')
    st.write(input_df)

    if st.button('Predict'):
        if model is not None:
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            st.subheader('Prediction Result:')
            
            if prediction[0] == 1:
                st.error('**Result: This customer is predicted to CHURN**')
                st.write(f"**Churn Probability:** `{prediction_proba[0][1]*100:.2f}%`")
                st.warning("""
                **Recommendation:** This customer is high risk. Consider offering a retention program or incentives.
                """)
            else:
                st.success('**Result: This customer is predicted to STAY**')
                st.write(f"**Churn Probability:** `{prediction_proba[0][1]*100:.2f}%`")
                st.info("""
                **Recommendation:** This customer is likely loyal. Maintain service quality or offer cross-sell products.
                """)
        else:
            st.error("Model could not be loaded. Prediction cannot proceed.")


if __name__ == '__main__':
    run()