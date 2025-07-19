from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta

def get_data():
    """Load and preprocess the attendance data."""
    data = pd.read_csv('data/attendance.csv', header=None)
    data = data.drop(data.columns[[0,3]], axis=1)
    data.columns = ['ds', 'y']
    data['ds'] = pd.to_datetime(data['ds'], yearfirst=True)
    data['y'] = data['y'].str.replace(',', '')
    data['y'] = data['y'].astype(float)
    return data

def get_today():
    """Get today's date."""
    return datetime.now().date()

def get_prediction_for_date(date):
    """Get the attendance prediction for a specific date."""
    m = Prophet()
    m.add_country_holidays(country_name='US')
    m.fit(get_data())

    future = pd.DataFrame({'ds': [date]})
    future['floor'] = 0
    forecast = m.predict(future)
    return {
        'date': date.strftime('%m/%d/%Y'),
        'prediction': forecast['yhat'].iloc[0],
        'upper_bound': forecast['yhat_upper'].iloc[0],
        'lower_bound': forecast['yhat_lower'].iloc[0]
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
    
    next_week_dates = [today + timedelta(days=i) for i in range(1, 8)]
    
    m = Prophet()
    m.add_country_holidays(country_name='US')
    m.fit(get_data())
    
    future = pd.DataFrame({'ds': next_week_dates})
    future['floor'] = 0
    forecast = m.predict(future)
    
    predictions_df = pd.DataFrame({
        'date': forecast['ds'].dt.strftime('%m/%d/%Y'),
        'day_of_week': forecast['ds'].dt.day_name(),
        'prediction': forecast['yhat'],
    })
    
    return predictions_df

def get_plot():
    """Generate and save the forecast plot."""
    import os
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    data = get_data()
    m = Prophet()
    m.add_country_holidays(country_name='US')
    m.fit(data)
    
    # Create future dates starting from today for next 30 days
    plotdf = pd.DataFrame({'ds': pd.date_range(start=get_today(), periods=30)})
    plotdf['floor'] = 0
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

def hello_world(request):
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
    
    # Generate HTML rows for the week table
    week_rows = ""
    for _, row in next_week_df.iterrows():
        # Determine row class based on busiest/slowest
        row_class = ""
        if row['date'] == busiest_day['date']:
            row_class = "busiest-day"
        elif row['date'] == slowest_day['date']:
            row_class = "slowest-day"
            
        week_rows += f"""
        <tr class="{row_class}">
            <td>{row['day_of_week']}</td>
            <td>{row['date']}</td>
            <td class="prediction-cell">{int(row['prediction'])}</td>
        </tr>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wild Cast - Visitor Attendance Predictions</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }}
            .container {{
                text-align: center;
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                max-width: 800px;
            }}
            h1 {{
                color: #333;
                margin-bottom: 20px;
                font-size: 2.5em;
            }}
            p {{
                color: #666;
                font-size: 1.2em;
                margin-bottom: 30px;
            }}
            .logo {{
                font-size: 3em;
                margin-bottom: 20px;
            }}
            .subtitle {{
                color: #888;
                font-style: italic;
            }}
            .predictions-container {{
                display: flex;
                gap: 20px;
                justify-content: center;
                margin: 20px 0;
                flex-wrap: wrap;
            }}
            .prediction-box {{
                background: #f8f9fa;
                border: 2px solid #667eea;
                border-radius: 8px;
                padding: 20px;
                flex: 1;
                min-width: 250px;
                max-width: 350px;
            }}
            .prediction-box.today {{
                border-color: #28a745;
            }}
            .prediction-box.tomorrow {{
                border-color: #667eea;
            }}
            .prediction-value {{
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .prediction-value.today {{
                color: #28a745;
            }}
            .prediction-value.tomorrow {{
                color: #667eea;
            }}
            .prediction-value.upper-bound {{
                font-size: 1em;
                color: #dc3545;
                font-weight: normal;
                margin: 5px 0;
            }}
            .prediction-value.lower-bound {{
                font-size: 1em;
                color: #6c757d;
                font-weight: normal;
                margin: 5px 0;
            }}
            .prediction-date {{
                color: #666;
                font-size: 1.1em;
            }}
            .week-forecast {{
                margin-top: 40px;
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
            }}
            .week-forecast h3 {{
                color: #333;
                margin-bottom: 20px;
            }}
            .forecast-table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 6px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .forecast-table th {{
                background: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
            }}
            .forecast-table td {{
                padding: 12px;
                border-bottom: 1px solid #eee;
            }}
            .forecast-table tr:last-child td {{
                border-bottom: none;
            }}
            .forecast-table tr:hover {{
                background: #f5f5f5;
            }}
            .prediction-cell {{
                font-weight: bold;
                color: #667eea;
                text-align: center;
            }}
            .busiest-day {{
                background: #fff3cd !important;
                border-left: 4px solid #f39c12 !important;
            }}
            .busiest-day:hover {{
                background: #ffeaa7 !important;
            }}
            .busiest-day .prediction-cell {{
                color: #e67e22;
                font-weight: bold;
            }}
            .slowest-day {{
                background: #d1ecf1 !important;
                border-left: 4px solid #17a2b8 !important;
            }}
            .slowest-day:hover {{
                background: #bee5eb !important;
            }}
            .slowest-day .prediction-cell {{
                color: #138496;
                font-weight: bold;
            }}
            .forecast-chart {{
                margin-top: 40px;
                background: #f8f9fa;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
            }}
            .forecast-chart h3 {{
                color: #333;
                margin-bottom: 20px;
            }}
            .forecast-chart img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                background: white;
                padding: 15px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">🦁</div>
            <h1>Wild Cast</h1>
            <p>Welcome to <strong>Wild Cast</strong> - Your Wildlife Attendance Intelligence</p>
            <p class="subtitle">Your assitant in predicting Safari Park attendance</p>
            
            <div class="predictions-container">
                <div class="prediction-box today">
                    <h3>Today's Prediction</h3>
                    <div class="prediction-value today">{int(todays_data['prediction'])}</div>
                    <div class="prediction-value upper-bound">High End: {int(todays_data['upper_bound'])}</div>
                    <div class="prediction-value lower-bound">Low End: {int(todays_data['lower_bound'])}</div>
                    <div class="prediction-date">for {todays_data['date']}</div>
                </div>
                
                <div class="prediction-box tomorrow">
                    <h3>Tomorrow's Prediction</h3>
                    <div class="prediction-value tomorrow">{int(tomorrows_data['prediction'])}</div>
                    <div class="prediction-value upper-bound">High End: {int(tomorrows_data['upper_bound'])}</div>
                    <div class="prediction-value lower-bound">Low End: {int(tomorrows_data['lower_bound'])}</div>
                    <div class="prediction-date">for {tomorrows_data['date']}</div>
                </div>
            </div>
            
                        <div class="week-forecast">
                <h3>📅 7-Day Visitor Forecast</h3>
                <table class="forecast-table">
                    <thead>
                        <tr>
                            <th>Day</th>
                            <th>Date</th>
                            <th>Expected Visitors</th>
                        </tr>
                    </thead>
                    <tbody>
                        {week_rows}
                    </tbody>
                </table>
            </div>
            
            <div class="forecast-chart">
                <h3>📈 30-Day Forecast Trend</h3>
                <img src="{plot_path}?{todays_data['date']}" alt="30-Day Visitor Forecast Chart" />
                <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
                    Blue line shows expected visitors, gray area shows confidence range
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    return HttpResponse(html)
