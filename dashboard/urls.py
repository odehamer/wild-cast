from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('input/', views.input, name='input'),
]
