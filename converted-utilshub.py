
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
import torch
from flask import Flask, jsonify, request
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from darts.metrics import mae, rmse



dataset = pd.read_csv("dataset.csv")
dataset.head()

dataset.info()

total_columns = [col for col in dataset.columns if col.endswith('- total')]

# Filter the dataset to include only the year and '-total' columns
filtered_data = dataset[['Year'] + total_columns]

total_columns

# Melt the dataset
melted_data = pd.melt(filtered_data, id_vars=["Year"], var_name="Fuel_Type", value_name="Consumption")
# Convert 'Year' to datetime type
melted_data['Year'] = pd.to_datetime(melted_data['Year'], format='%Y')
# Display the melted data
melted_data.head()

melted_data['Consumption'] = melted_data['Consumption'].replace({',': ''}, regex=True)
melted_data['Consumption'] = pd.to_numeric(melted_data['Consumption'], errors='coerce')

melted_data

# Step 1: Prepare Data (use index as time)
melted_data['Index'] = melted_data.index  # Adding index as a column for TimeSeries
# Step 2: Convert to TimeSeries using the index as time
series = TimeSeries.from_dataframe(melted_data, time_col="Index", value_cols="Consumption")
# Step 4: Normalization
scaler = Scaler()
series_scaled = scaler.fit_transform(series)
# Step 5: Train/Test split (using last 42 values for validation)
train, val = series_scaled[:-42], series_scaled[-42:]
# Objective function for Optuna
def objective(trial):
    # Hyperparameter tuning
    input_chunk_length = trial.suggest_int('input_chunk_length', 24, 120)
    output_chunk_length = trial.suggest_int('output_chunk_length', 6, 24)
    n_epochs = trial.suggest_int('n_epochs', 100, 500)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    # Step 6: N-BEATS model with parameters from Optuna
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        n_epochs=n_epochs,
        random_state=42,
        batch_size=batch_size
    )
    # Step 7: Fit the model
    model.fit(train, verbose=False)
    # Step 8: Evaluate the model using validation data
    forecast = model.predict(n=len(val))
    # Calculate the Mean Absolute Percentage Error (MAPE) as a performance metric
    mape_value = np.mean(np.abs((forecast.values() - val.values()) / val.values())) * 100
    return mape_value
# Step 9: Set up Optuna study and optimize
study = optuna.create_study(direction='minimize')  # We want to minimize the MAPE
study.optimize(objective, n_trials=50)  # Number of trials (you can increase this for more thorough search)
# Step 10: Display the best parameters found by Optuna
print("Best trial:")
best_trial = study.best_trial
print(f"  Value (MAPE): {best_trial.value}")
print(f"  Params: {best_trial.params}")
# Step 11: Train the model with the best parameters
best_params = best_trial.params
best_model = NBEATSModel(
    input_chunk_length=best_params['input_chunk_length'],
    output_chunk_length=best_params['output_chunk_length'],
    n_epochs=best_params['n_epochs'],
    batch_size=best_params['batch_size'],
    random_state=42
)
best_model.fit(train, verbose=True)
# Step 12: Forecasting using the best model
forecast = best_model.predict(n=7)
# Step 13: Plotting the results
series_scaled.plot(label="Actual")
forecast.plot(label="Forecast")
plt.xlabel("Index")
plt.ylabel("Consumption")
plt.title("Fuel Consumption Forecast using Optimized N-BEATS")
plt.legend()
plt.show()


param = {'input_chunk_length': 115, 'output_chunk_length': 13, 'n_epochs': 115, 'batch_size': 30}
best_model = NBEATSModel(
    **param
)
best_model.fit(train, verbose=True)
# Step 12: Forecasting using the best model
forecast = best_model.predict(n=42)
# Step 13: Plotting the results
series_scaled.plot(label="Actual")
forecast.plot(label="Forecast")
plt.xlabel("Index")
plt.ylabel("Consumption")
plt.title("Fuel Consumption Forecast using Optimized N-BEATS")
plt.legend()
plt.show()
print(f"MAE: {mae(val, forecast)}")
print(f"RMSE: {rmse(val, forecast)}")


best_model.save("nbeats_model.pth")

# Step 1: Prepare Data (use index as time)
melted_data['Index'] = melted_data.index  # Adding index as a column for TimeSeries
# Step 2: Convert to TimeSeries using the index as time
series = TimeSeries.from_dataframe(melted_data, time_col="Index", value_cols="Consumption")
# Step 4: Normalization
scaler = Scaler()
series_scaled = scaler.fit_transform(series)

app = Flask(__name__)
# Load the trained model
def load_model():
    best_model = NBEATSModel.load("nbeats_model.pth")
    return best_model
# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json()
        # Extract necessary input from the request
        input_data = np.array(data["input"]).reshape(-1, 1)  # Input data
        n_predictions = data.get("n", 7)
        input_series = TimeSeries.from_values(input_data)
        scaler = Scaler()
        input_series_scaled = scaler.fit_transform(input_series)
        # Get the model and make the prediction
        model = load_model()
        forecast = model.predict(n=n_predictions)  # Predict `n_predictions` future values
        # Inverse transform the forecast
        forecast_values = scaler.inverse_transform(forecast.values())
        # Return the predicted values as a JSON response
        return jsonify({'prediction': forecast_values.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})
# Run the Flask application
if __name__ == '__main__':
    # Print the URL of the API after the app starts
    print("Starting the Flask API...")
    print(f"API is running at: http://127.0.0.1:5000/predict")
    app.run(debug=True)

