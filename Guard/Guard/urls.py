from django.contrib import admin
from django.urls import path
from django.urls import include
from Dashboard.views import lung_cancer_prediction

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',lung_cancer_prediction , name='dashboard')

]
