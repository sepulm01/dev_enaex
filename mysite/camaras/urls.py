from django.urls import path 
from django.conf.urls import include, url
from camaras.views import alarma_post, alarma_inst, eventos, eventos_list

urlpatterns = [
path('api/alarma/', alarma_post, name='alarma_post'),
path('ajax/alarmas/', alarma_inst, name='alarma_inst'),
path('alarmas/eventos', eventos_list, name='eventos_list'),
path('alarmas/eventos/<int:id>/', eventos, name='eventos'),
]