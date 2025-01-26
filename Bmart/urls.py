from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Replace 'home' with your view function
    # path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
]