import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from sklearn.metrics import r2_score
from meteostat import Point, Daily


data = pd.read_csv('data/attendance.csv', header=None,)
data = data.drop(data.columns[[0,3]], axis=1)
data.columns = ['ds', 'y']
data['ds'] = pd.to_datetime(data['ds'], yearfirst=True)
data['y'] = data['y'].str.replace(',', '')
data['y'] = data['y'].astype(float)

m = Prophet()
m.add_country_holidays(country_name='US')
m.fit(data)

start_date = data['ds'].min()
end_date = data['ds'].max()
location = Point(33.0980, -116.9967, 150) # Safari park coordinates
weather_data = Daily(location, start_date, end_date)
weather_data = weather_data.fetch()

# Convert temperature to Fahrenheit
weather_data['tavg_fahrenheit'] = weather_data['tavg'] * 9/5 + 32

data['temp'] = weather_data['tavg_fahrenheit'].values
data['prcp'] = weather_data['prcp'].fillna(0).values
m.add_regressor('prcp', prior_scale=0.1)
print(data.to_string())
print(weather_data.to_string())

future = m.make_future_dataframe(periods=0)
future['floor'] = 0
forecast = m.predict(future)

temp_df = pd.DataFrame({
    'prediction_actual_difference': data['y'] - forecast['yhat'],
    'temp': data['temp']
})

temp_df['temp'] = temp_df['temp'].astype(float)

prec_df = pd.DataFrame({
    'prediction_actual_difference': data['y'] - forecast['yhat'],
    'precipitation': weather_data['prcp'].fillna(0).values
})

plt.figure(figsize=(12, 6))
plt.scatter(temp_df['temp'], temp_df['prediction_actual_difference'], alpha=0.5)
plt.title('Temperature vs Prediction-Actual Difference')
plt.xlabel('Average Temperature (°F)')
plt.ylabel('Prediction - Actual Difference')
plt.grid()
plt.savefig('static/temperature_vs_difference.png') 

plt.figure(figsize=(12, 6))
plt.scatter(prec_df['precipitation'], prec_df['prediction_actual_difference'], alpha=0.5)
plt.title('Precipitation vs Prediction-Actual Difference')
plt.xlabel('Daily Precipitation (mm)')
plt.ylabel('Prediction - Actual Difference')
plt.grid()
plt.savefig('static/precipitaion_vs_difference.png') 


# Make predictions for the same dates as the historical data
historical_forecast = m.predict(data[['ds']])

# Calculate R-squared
actual_values = data['y'].values
predicted_values = historical_forecast['yhat'].values
r_squared = r2_score(actual_values, predicted_values)

print(f"R-squared value: {r_squared:.4f}")
print(f"Model explains {r_squared*100:.2f}% of the variance in the data")

# Also show some sample comparisons
comparison_df = pd.DataFrame({
    'Date': data['ds'].dt.strftime('%Y-%m-%d'),
    'Actual': actual_values,
    'Predicted': predicted_values.round(0),
    'Difference': (actual_values - predicted_values).round(0)
})

print("\nSample of actual vs predicted values:")
print(comparison_df.head(10).to_string(index=False))

# Future predictions
future = m.make_future_dataframe(periods=17)
future['floor'] = 0
forecast = m.predict(future)
print("\nFuture predictions:")
print(forecast[['ds', 'yhat']].tail(17).to_string())

