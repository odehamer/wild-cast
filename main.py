import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from sklearn.metrics import r2_score

data = pd.read_csv('data/attendance.csv', header=None,)
data = data.drop(data.columns[[0,3]], axis=1)
data.columns = ['ds', 'y']
data['ds'] = pd.to_datetime(data['ds'], yearfirst=True)
data['y'] = data['y'].str.replace(',', '')
data['y'] = data['y'].astype(float)

m = Prophet()
m.add_country_holidays(country_name='US')
m.fit(data)

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

