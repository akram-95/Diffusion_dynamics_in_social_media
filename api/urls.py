from django.urls import path
from api import views

urlpatterns = [
    path('api/detect_topics', views.detect_topics),
    path('api/detect_topics_bertopic', views.detect_topics_bertopic),
]
