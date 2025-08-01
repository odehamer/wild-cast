import os
import django
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta, date
from meteostat import Point, Daily
import requests

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wildcast.settings')
django.setup()

from dashboard.models import DailyAttendance, Location, AttendancePrediction, SevenDayPrediction

def db_to_df():
    df = pd.DataFrame({
        'ds': DailyAttendance.objects.values_list('date', flat=True),
        'y': DailyAttendance.objects.values_list('count', flat=True),
        'high_temp': DailyAttendance.objects.values_list('high_temp', flat=True),
        'precipitation': DailyAttendance.objects.values_list('precipitation', flat=True)
    })

    return df

def get_weather_gov_forecast():
    """Get weather forecast from Weather.gov API for Safari Park location."""
    # Safari Park coordinates: 33.0980, -116.9967
    lat, lon = 33.0980, -116.9967
    
    try:
        # First, get the grid point for this location
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        points_response = requests.get(points_url, headers={'User-Agent': 'WildCast/1.0'})
        
        if points_response.status_code != 200:
            return None
            
        points_data = points_response.json()
        forecast_url = points_data['properties']['forecast']
        
        # Get the forecast
        forecast_response = requests.get(forecast_url, headers={'User-Agent': 'WildCast/1.0'})
        
        if forecast_response.status_code != 200:
            return None
            
        forecast_data = forecast_response.json()
        periods = forecast_data['properties']['periods']
        
        # Process the forecast data
        forecasts = []
        for period in periods:
            # Weather.gov returns day/night pairs, we want daily forecasts
            if period['isDaytime']:
                date_str = period['startTime'][:10]  # Extract YYYY-MM-DD
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                # Extract temperature and precipitation info
                temp = period['temperature']  # Weather.gov already provides daily high for daytime periods
                
                # Look for precipitation in detailed forecast
                precip = 0.0
                detailed_forecast = period['detailedForecast'].lower()
                if 'rain' in detailed_forecast or 'shower' in detailed_forecast:
                    # Simple heuristic for precipitation in mm
                    if 'heavy' in detailed_forecast:
                        precip = 12.7  # ~0.5 inches = 12.7mm
                    elif 'light' in detailed_forecast:
                        precip = 2.5   # ~0.1 inches = 2.5mm
                    else:
                        precip = 6.4   # ~0.25 inches = 6.4mm
                
                forecasts.append({
                    'date': date_obj,
                    'temperature': temp,  # This is the daily high temperature
                    'precipitation': precip,
                    'description': period['shortForecast']
                })
        
        return forecasts
    
    except Exception as e:
        print(f"Weather.gov API error: {e}")
        return None


def make_weather_predictions():
    weather_forecasts = get_weather_gov_forecast()
    next_week_dates = [datetime.now().date() + timedelta(days=i) for i in range(0, 7)]

    df = db_to_df()
    df = df.dropna()
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = df['y'].astype(float)
    df['high_temp'] = df['high_temp'].fillna(0).astype(float).values
    df['precipitation'] = df['precipitation'].fillna(0).astype(float).values

    m = Prophet()
    m.add_country_holidays(country_name='US')
    m.add_regressor('high_temp', prior_scale=0.1)
    m.add_regressor('precipitation', prior_scale=0.1)
    
    m.fit(df)
    
    future = pd.DataFrame({'ds': next_week_dates})
    future['floor'] = 0

    temperatures = []
    precipitation = []
    for i in range(len(weather_forecasts)):
        temperatures.append(weather_forecasts[i]['temperature'])
        precipitation.append(weather_forecasts[i]['precipitation'])

    future['high_temp'] = temperatures
    future['precipitation'] = precipitation
    
    forecast = m.predict(future)

    SevenDayPrediction.objects.all().delete()
    for i in range(len(forecast)):
        SevenDayPrediction.objects.create(
            date=forecast['ds'].iloc[i],
            location=Location.objects.get(name='Safari Park'),
            value=forecast['yhat'].iloc[i],
            high_temp=temperatures[i],
            precipitation=precipitation[i]
        )
    print("7-Day predictions created successfully.")
    
def make_attendance_predictions():
    df = db_to_df()
    df = df.dropna()
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = df['y'].astype(float)
    df.drop(df.columns[[2, 3]], axis=1, inplace=True)

    dates = [datetime.now().date() + timedelta(days=i) for i in range(0, 365)]

    m = Prophet()
    m.add_country_holidays(country_name='US')
    m.fit(df)

    future = pd.DataFrame({'ds': dates})
    future['floor'] = 0
    forecast = m.predict(future)

    AttendancePrediction.objects.all().delete()
    for i in range(len(forecast)):
        AttendancePrediction.objects.create(
            date=forecast['ds'].iloc[i],
            location=Location.objects.get(name='Safari Park'),
            value=forecast['yhat'].iloc[i]
        )

make_weather_predictions()
make_attendance_predictions()