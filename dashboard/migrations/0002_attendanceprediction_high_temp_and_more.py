# Generated by Django 5.2.4 on 2025-07-29 04:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='attendanceprediction',
            name='high_temp',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='attendanceprediction',
            name='precipitation',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='dailyattendance',
            name='high_temp',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='dailyattendance',
            name='precipitation',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
