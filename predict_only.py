from flask import Flask, jsonify, request
import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler

app = Flask(__name__)

# Load trained model
def load_model():
    return NBEATSModel.load("nbeats_model.pth")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data input dari user
        data = request.get_json()
        input_data = np.array(data["input"]).reshape(-1, 1)
        n_predictions = data.get("n", 7)

        # Buat timeseries dan lakukan scaling
        input_series = TimeSeries.from_values(input_data)
        scaler = Scaler()
        input_series_scaled = scaler.fit_transform(input_series)

        # Load model dan prediksi
        model = load_model()
        forecast = model.predict(n=n_predictions)

        # Kembalikan hasil prediksi
        forecast_values = scaler.inverse_transform(forecast).values()
        return jsonify({'prediction': forecast_values.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Starting the Flask prediction API...")
    print("Open in browser or send POST to http://127.0.0.1:5000/predict")
    app.run(debug=True)
