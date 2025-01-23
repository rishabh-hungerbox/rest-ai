
from django.urls import include, path

urlpatterns = [
    path('api/menu-mapping/', include('menu_mapping.urls')),
    ]
