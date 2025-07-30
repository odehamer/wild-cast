import os
import django
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wildcast.settings')
django.setup()

from dashboard.models import DailyAttendance

def db_to_df():
    df = pd.DataFrame({
        'ds': DailyAttendance.objects.values_list('date', flat=True),
        'y': DailyAttendance.objects.values_list('count', flat=True),
        'high_temp': DailyAttendance.objects.values_list('high_temp', flat=True),
        'precipitation': DailyAttendance.objects.values_list('precipitation', flat=True)
    })

    return df

