# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model (tuned Random Forest model)
try:
    model = joblib.load('random_forest_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'random_forest_model.pkl' is in the same directory.")
    st.stop()

# Load the target encoder to inverse transform predictions
try:
    target_classes = ['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I', 'Obesity_Type_II',
                      'Obesity_Type_III', 'Overweight_Level_I', 'Overweight_Level_II']

    from sklearn.preprocessing import LabelEncoder
    target_encoder = LabelEncoder()
    target_encoder.fit(target_classes)

except FileNotFoundError:
    st.warning("Target encoder file not found. Using default class labels.")
    target_encoder = None
    target_classes = ['0', '1', '2', '3', '4', '5', '6']  # Fallback to encoded labels

# Streamlit UI
st.set_page_config(page_title="Prediksi Tingkat Obesitas", layout="wide")

st.title('Prediksi Tingkat Obesitas')

st.write("""
Selamat datang di aplikasi prediksi tingkat obesitas!
Aplikasi ini menggunakan model machine learning (Random Forest) yang telah dilatih
menggunakan fitur-fitur yang paling berpengaruh untuk memprediksi kategori obesitas
berdasarkan input Anda.

Silakan masukkan informasi berikut:
""")

# Input features from user - ONLY the influential features
st.header("Masukkan Data Diri dan Kebiasaan")

col1, col2 = st.columns(2)

with col1:
    weight = st.number_input('Berat (kg)', min_value=10.0, max_value=250.0, value=70.0,
                             help="Masukkan berat badan Anda dalam kilogram.")
    faf = st.slider('Frekuensi Aktivitas Fisik (skala 0-3)', 0.0, 3.0, 1.0,
                    help="Skala 0 (tidak pernah) hingga 3 (setiap hari).")
    fcvc = st.slider('Konsumsi Sayur dan Buah (skala 1-3)', 1.0, 3.0, 2.0,
                     help="Skala 1 (jarang) hingga 3 (sering).")

with col2:
    family_history_with_overweight = st.selectbox('Riwayat Keluarga Obesitas?', ['no', 'yes'],
                                                  help="Apakah ada riwayat obesitas dalam keluarga Anda?")
    mtrans = st.selectbox('Transportasi Utama',
                          ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'],
                          help="Pilih moda transportasi yang paling sering Anda gunakan.")

# Map categorical inputs to numerical values based on the encoding used during training
family_history_with_overweight_map = {'no': 0, 'yes': 1}
mtrans_map = {'Public_Transportation': 3, 'Walking': 4, 'Automobile': 0, 'Motorbike': 2, 'Bike': 1}

# Create a DataFrame with the input data - ONLY with selected features
input_data = pd.DataFrame({
    'Weight': [weight],
    'family_history_with_overweight': [family_history_with_overweight_map[family_history_with_overweight]],
    'FAF': [faf],
    'MTRANS': [mtrans_map[mtrans]],
    'FCVC': [fcvc]
})

# Scaling
min_vals_hardcoded = pd.Series({
    'Weight': 39.0,
    'family_history_with_overweight': 0.0,
    'FAF': 0.0,
    'MTRANS': 0.0,
    'FCVC': 1.0
})

max_vals_hardcoded = pd.Series({
    'Weight': 250.0,
    'family_history_with_overweight': 1.0,
    'FAF': 3.0,
    'MTRANS': 4.0,
    'FCVC': 3.0
})

# Apply manual scaling using hardcoded min/max
input_data_scaled = (input_data - min_vals_hardcoded) / (max_vals_hardcoded - min_vals_hardcoded)

# Make prediction
if st.button('Prediksi Tingkat Obesitas'):
    input_data_scaled = input_data_scaled[['Weight', 'family_history_with_overweight', 'FAF', 'MTRANS', 'FCVC']]

    prediction_encoded = model.predict(input_data_scaled)

    if target_encoder:
        prediction_label = target_encoder.inverse_transform(prediction_encoded)
        st.success(f'Hasil Prediksi: **{prediction_label[0]}**')
    else:
        st.success(f'Hasil Prediksi (Encoded): **{prediction_encoded[0]}**')

st.markdown("---")
st.write("Model dilatih menggunakan data obesitas dan fitur-fitur yang paling berpengaruh.")