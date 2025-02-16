from django.urls import path
from . import views as Eviews
urlpatterns = [
    path('extract/',Eviews.upload_files ),
    path('extractBills/',Eviews.upload_Bills),
    # path('testing/',Eviews.testing),
    
]