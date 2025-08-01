from django.db import models

# Create your models here.

class Location(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name

class DailyAttendance(models.Model):
    date = models.DateField()
    location = models.ForeignKey(Location, on_delete=models.CASCADE)
    count = models.IntegerField()
    high_temp = models.FloatField(blank=True, null=True)
    precipitation = models.FloatField(blank=True, null=True)

class AttendancePrediction(models.Model):
    date = models.DateField()
    location = models.ForeignKey(Location, on_delete=models.CASCADE)
    value = models.FloatField()


    def __str__(self):
        return f"Prediction for {self.date} at {self.location}: {self.value}"
    
class SevenDayPrediction(models.Model):
    date = models.DateField()
    location = models.ForeignKey(Location, on_delete=models.CASCADE)
    value = models.FloatField()
    high_temp = models.FloatField(blank=True, null=True)
    precipitation = models.FloatField(blank=True, null=True)

    def __str__(self):
        return f"7-Day Prediction for {self.date} at {self.location}: {self.value}"