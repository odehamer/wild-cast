from datetime import datetime
from meteostat import Point, Daily
import csv
import os
import django

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wildcast.settings')
django.setup()

from dashboard.models import DailyAttendance
from dashboard.models import Location

cords = Point(33.0980, -116.9967, 150) # Safari park coordinates

Location.objects.get_or_create(
    name='Safari Park',
    defaults={'description': 'San Diego Safari Park, located in Escondido, California.'}
)

with open('data/attendance.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)
    for row in data:
        try:
            d = datetime.strptime(row[0], '%m/%d/%Y')
        except ValueError:
            try:
                d = datetime.strptime(row[0], '%m/%d/%y')
            except ValueError:
                print(f"Could not parse date: {row[0]}")
                continue
        
        row[0] = d
        row[1] = int(row[1].replace(',', ''))

        weather_data = Daily(cords, row[0], row[0])
        weather_data = weather_data.fetch()

        temp = weather_data['tmax'].fillna(0).astype(float) * 9/5 + 32
        prec = weather_data['prcp'].fillna(0).astype(float)

        DailyAttendance.objects.create(
            date=row[0],
            count=row[1],
            location = Location.objects.get(name='Safari Park'),
            high_temp=temp,
            precipitation=prec
        )

print("Daily attendance data loaded successfully.")