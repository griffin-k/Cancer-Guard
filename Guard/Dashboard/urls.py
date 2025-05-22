from django.urls import path
from . import views

urlpatterns = [
    path('home', views.view_home, name="home"),
    path("cancer_prediction/", views.lung_cancer_prediction, name='cancer_prediction'),
    path("generate_pdf/", views.generate_pdf, name="generate_pdf"),
    path("ask_questions/", views.view_question, name="ask_question"),
    path("support/", views.view_support, name="support"),
    path("about/", views.view_about, name="about"),
    path("", views.view_index, name="index"),
]
