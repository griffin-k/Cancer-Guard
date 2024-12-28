from django.urls import path
from . import views

urlpatterns = [
    path('', views.lung_cancer_prediction, name='cancer_prediction'),

]