import streamlit as st
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt

# Load trained model
@st.cache_resource
def load_model():
    return NBEATSModel.load("nbeats_model.pth")

st.title("📈 Dashboard Prediksi Konsumsi BBM (N-BEATS Model)")

# Input data
st.subheader("Masukkan Data Historis")
input_values = st.text_area(
    "Masukkan angka historis (pisahkan dengan koma)",
    value="100, 120, 130, 140, 160, 180, 200"
)

n_steps = st.slider("Jumlah langkah prediksi ke depan (n):", 1, 30, 7)

# Tombol prediksi
if st.button("Prediksi"):
    try:
        # Parse input
        input_list = [float(x.strip()) for x in input_values.split(",")]
        input_array = np.array(input_list).reshape(-1, 1)

        # Buat timeseries & scaling
        input_series = TimeSeries.from_values(input_array)
        scaler = Scaler()
        input_series_scaled = scaler.fit_transform(input_series)

        # Load model dan prediksi
        model = load_model()
        forecast = model.predict(n=n_steps)
        forecast_values = scaler.inverse_transform(forecast).values()

        # Tampilkan hasil prediksi
        st.success("Prediksi berhasil!")
        pred_df = pd.DataFrame(forecast_values, columns=["Prediksi"])
        st.dataframe(pred_df)

        # Plot
        fig, ax = plt.subplots()
        input_series.plot(label="Data Historis", ax=ax)
        forecast.plot(label="Prediksi", ax=ax)
        plt.title("Hasil Prediksi Konsumsi")
        plt.xlabel("Waktu")
        plt.ylabel("Nilai")
        plt.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
