from django.urls import path
from . import views as Eviews
urlpatterns = [
    path('extract/',Eviews.upload_files ),
    
]