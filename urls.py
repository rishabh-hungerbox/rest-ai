
from django.urls import include, path

urlpatterns = [
    path('api/ai/menu-mapping/', include('menu_mapping.urls')),
    ]
