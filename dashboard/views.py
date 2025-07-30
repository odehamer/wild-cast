from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta, date
from meteostat import Point, Daily
import requests
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wildcast.settings')
django.setup()

from dashboard.models import DailyAttendance, Location


def get_data():
    """Load and preprocess the attendance data."""
    data = pd.read_csv('data/attendance.csv', header=None)
    data.columns = ['ds', 'y']
    data['ds'] = pd.to_datetime(data['ds'], yearfirst=True)
    data['y'] = data['y'].str.replace(',', '')
    data['y'] = data['y'].astype(float)
    return data

def get_today():
    """Get today's date."""
    return datetime.now().date()

def get_weather_data(start_date, end_date):
    """Get weather data using meteostat for the given date range."""
    location = Point(33.0980, -116.9967, 150)
    weather_data = Daily(location, start_date, end_date)
    weather_data = weather_data.fetch()
    
    # Convert to the expected format using daily highs instead of averages
    weather_df = pd.DataFrame({
        'ds': weather_data.index,
        'tmax': weather_data['tmax'].fillna(0).astype(float) * 9/5 + 32,  # Convert daily high to Fahrenheit
        'prcp': weather_data['prcp'].fillna(0).astype(float)  # Keep precipitation in mm
    })
    weather_df['ds'] = pd.to_datetime(weather_df['ds'])
    
    return weather_df

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

def get_forecast_weather_for_date(target_date):
    """Get weather forecast for a specific date using Weather.gov."""
    forecasts = get_weather_gov_forecast()
    
    if not forecasts:
        # Fallback to default values
        return {'temperature': 75.0, 'precipitation': 0.0}  # Default daily high
    
    # Convert target_date to date object if it's datetime
    if isinstance(target_date, datetime):
        target_date = target_date.date()
    
    # Find forecast for the target date
    for forecast in forecasts:
        if forecast['date'] == target_date:
            return {
                'temperature': forecast['temperature'],  # Daily high from Weather.gov
                'precipitation': forecast['precipitation']
            }
    
    # If no exact match, return fallback
    return {'temperature': 75.0, 'precipitation': 0.0}  # Default daily high

def get_prediction_for_date(date_param):
    """Get the attendance prediction for a specific date."""
    data = get_data()

    # Convert date to datetime if it's a date object
    if isinstance(date_param, date) and not isinstance(date_param, datetime):
        date_param = datetime.combine(date_param, datetime.min.time())

    start_date = data['ds'].min()
    end_date = data['ds'].max()
    location = Point(33.0980, -116.9967, 150) # Safari park coordinates
    weather_data = Daily(location, start_date, end_date)
    weather_data = weather_data.fetch()
    
    m = Prophet()
    m.add_country_holidays(country_name='US')
    m.add_regressor('temp', prior_scale=0.1)
    m.add_regressor('prcp', prior_scale=0.1)
    data['temp'] = weather_data['tmax'].fillna(0).astype(float).values * 9/5 + 32  # Convert daily high to Fahrenheit
    data['prcp'] = weather_data['prcp'].fillna(0).astype(float).values
    m.fit(data)

    future = pd.DataFrame({'ds': [date_param]})
    future['floor'] = 0

    # Use Weather.gov for future forecasts
    weather_forecast = get_forecast_weather_for_date(date_param)
    future['temp'] = weather_forecast['temperature']
    future['prcp'] = weather_forecast['precipitation']
    temp_display = weather_forecast['temperature']
    prcp_display = weather_forecast['precipitation']

    forecast = m.predict(future)
    return {
        'date': date_param.strftime('%m/%d/%Y'),
        'prediction': forecast['yhat'].iloc[0],
        'upper_bound': forecast['yhat_upper'].iloc[0],
        'lower_bound': forecast['yhat_lower'].iloc[0],
        'temperature': temp_display,
        'precipitation': prcp_display
        } 

def get_todays_prediction():
    today = get_today()
    return get_prediction_for_date(today)

def get_tomorrows_prediction(): 
    tomorrow = get_today() + timedelta(days=1)
    return get_prediction_for_date(tomorrow)

def get_next_week_prediction():
    """Get predictions for the next 7 days as a DataFrame."""
    today = get_today()
    
    # Convert today to datetime for consistency
    if isinstance(today, date) and not isinstance(today, datetime):
        today_dt = datetime.combine(today, datetime.min.time())
    else:
        today_dt = today
    
    next_week_dates = [today_dt + timedelta(days=i) for i in range(0, 7)]
    
    # Get the training data with weather
    data = get_data()
    start_date = data['ds'].min()
    end_date = data['ds'].max()
    location = Point(33.0980, -116.9967, 150)
    weather_data = Daily(location, start_date, end_date)
    weather_data = weather_data.fetch()
    
    m = Prophet()
    m.add_country_holidays(country_name='US')
    m.add_regressor('temp', prior_scale=0.1)
    m.add_regressor('prcp', prior_scale=0.1)
    data['temp'] = weather_data['tmax'].fillna(0).astype(float).values * 9/5 + 32  # Convert daily high to Fahrenheit
    data['prcp'] = weather_data['prcp'].fillna(0).astype(float).values
    m.fit(data)
    
    future = pd.DataFrame({'ds': next_week_dates})
    future['floor'] = 0

    # Get Weather.gov forecasts for the week
    weather_gov_forecasts = get_weather_gov_forecast()
    
    temperatures = []
    precipitations = []
    
    for date_dt in next_week_dates:
        date_obj = date_dt.date()
        
        # Look for Weather.gov forecast for this date
        found_forecast = False
        if weather_gov_forecasts:
            for forecast in weather_gov_forecasts:
                if forecast['date'] == date_obj:
                    temperatures.append(forecast['temperature'])  # Daily high from Weather.gov
                    precipitations.append(forecast['precipitation'])
                    found_forecast = True
                    break
        
        # Fallback if no forecast found
        if not found_forecast:
            temperatures.append(75.0)  # Default daily high temperature
            precipitations.append(0.0)  # Default precipitation
    
    future['temp'] = temperatures
    future['prcp'] = precipitations

    forecast = m.predict(future)
    
    predictions_df = pd.DataFrame({
        'date': forecast['ds'].dt.strftime('%m/%d/%Y'),
        'day_of_week': forecast['ds'].dt.day_name(),
        'prediction': forecast['yhat'],
        'temperature': temperatures,
        'precipitation': precipitations
    })
    
    return predictions_df

def get_plot():
    """Generate and save the forecast plot."""
    import os
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Get the training data with weather
    data = get_data()
    start_date = data['ds'].min()
    end_date = data['ds'].max()
    location = Point(33.0980, -116.9967, 150)
    weather_data = Daily(location, start_date, end_date)
    weather_data = weather_data.fetch()
    
    m = Prophet()
    m.add_country_holidays(country_name='US')
    m.add_regressor('temp', prior_scale=0.1)
    m.add_regressor('prcp', prior_scale=0.1)
    data['temp'] = weather_data['tmax'].fillna(0).astype(float).values * 9/5 + 32  # Convert daily high to Fahrenheit
    data['prcp'] = weather_data['prcp'].fillna(0).astype(float).values
    m.fit(data)
    
    # Convert get_today() to datetime for consistency
    today = get_today()
    if isinstance(today, date) and not isinstance(today, datetime):
        today_dt = datetime.combine(today, datetime.min.time())
    else:
        today_dt = today
    
    # Create future dates starting from today for next 30 days
    plotdf = pd.DataFrame({'ds': pd.date_range(start=today_dt, periods=30)})
    plotdf['floor'] = 0
    plotdf['temp'] = 75.0  # Default daily high temperature
    plotdf['prcp'] = 0.0   # Default precipitation
    forecast = m.predict(plotdf)
    
    # Create a custom plot with only future data
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the forecast line
    ax.plot(forecast['ds'], forecast['yhat'], color='#1f77b4', linewidth=2, label='Forecast')
    
    # Plot the uncertainty interval
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                   color='#1f77b4', alpha=0.2, label='Confidence Interval')
    
    plt.title('30-Day Visitor Forecast', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Expected Visitors', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    plot_path = os.path.join(static_dir, 'forecast_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return 'static/forecast_plot.png'
    
def get_historical_weather_for_date(target_date):
    """Get historical weather data for a specific date."""
    try:
        location = Point(33.0980, -116.9967, 150)
        weather_data = Daily(location, target_date, target_date)
        weather_data = weather_data.fetch()
        
        if not weather_data.empty:
            temp = weather_data['tmax'].fillna(70).astype(float).iloc[0] * 9/5 + 32  # Daily high in Fahrenheit
            prec = weather_data['prcp'].fillna(0).astype(float).iloc[0]
            return {'temperature': temp, 'precipitation': prec}
        else:
            return {'temperature': 75.0, 'precipitation': 0.0}
    except Exception:
        return {'temperature': 75.0, 'precipitation': 0.0}

def input(request):
    context = {}
    
    if request.method == 'POST':
        date_str = request.POST.get('date')
        attendance_str = request.POST.get('attendance')
        
        if date_str and attendance_str:
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                attendance_count = int(attendance_str.replace(',', ''))
                
                # Get historical weather data for this date
                weather_data = get_historical_weather_for_date(date_obj)
                
                # Check if entry already exists and update or create
                entry, created = DailyAttendance.objects.get_or_create(
                    date=date_obj,
                    location=Location.objects.get(name='Safari Park'),
                    defaults={
                        'count': attendance_count,
                        'high_temp': weather_data['temperature'],
                        'precipitation': weather_data['precipitation']
                    }
                )
                
                if not created:
                    # Update existing entry
                    entry.count = attendance_count
                    entry.high_temp = weather_data['temperature']
                    entry.precipitation = weather_data['precipitation']
                    entry.save()
                    context['success'] = f"Updated attendance data for {date_obj.strftime('%B %d, %Y')} with {attendance_count} visitors!"
                else:
                    context['success'] = f"Added attendance data for {date_obj.strftime('%B %d, %Y')} with {attendance_count} visitors!"
                
            except ValueError as e:
                context['error'] = f"Error processing data: {e}"
            except Location.DoesNotExist:
                context['error'] = "Safari Park location not found in database."
            except Exception as e:
                context['error'] = f"Unexpected error: {e}"
    
    # Get recent entries for display
    try:
        recent_entries = DailyAttendance.objects.filter(
            location__name='Safari Park'
        ).order_by('-date')[:10]
        context['recent_entries'] = recent_entries
    except:
        context['recent_entries'] = []
    
    return render(request, 'dashboard/input.html', context)

def homepage(request):
    """Return a nicely formatted hello world message with predictions."""
    
    # Get both today's and tomorrow's predictions
    todays_data = get_todays_prediction()
    tomorrows_data = get_tomorrows_prediction()
    
    # Get next week's predictions
    next_week_df = get_next_week_prediction()
    
    # Generate the forecast plot
    plot_path = get_plot()
    
    # Find busiest and slowest days
    busiest_day = next_week_df.loc[next_week_df['prediction'].idxmax()]
    slowest_day = next_week_df.loc[next_week_df['prediction'].idxmin()]
    
    # Convert DataFrame to list of dictionaries for template iteration
    next_week_data = next_week_df.to_dict('records')
    
    context = {
        'todays_data': todays_data,
        'tomorrows_data': tomorrows_data,
        'next_week_df': next_week_data,
        'busiest_day': busiest_day,
        'slowest_day': slowest_day,
        'plot_path': plot_path,
    }
    
    return render(request, 'dashboard/homepage.html', context)
