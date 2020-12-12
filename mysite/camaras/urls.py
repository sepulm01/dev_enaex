from django.urls import path 
from django.conf.urls import include, url
from django.views.generic import TemplateView
from camaras.views import  ( alarma_post,
                            alarma_inst,
                            eventos,
                            eventos_list,
                            upd_envio,
                            cam_disp,
                            camviva_post,
                            cam_down,
                            licencia,
                            )

urlpatterns = [
path('api/alarma/', alarma_post, name='alarma_post'),
path('api/cam_down/', cam_down, name='cam_down'),
path('api/cam_disp/<int:id>/', cam_disp, name='cam_disp'),
path('api/camviva/', camviva_post, name='camviva_post'),
path('ajax/alarmas/', alarma_inst, name='alarma_inst'),
path('alarmas/eventos', eventos_list, name='eventos_list'),
path('alarmas/eventos/<int:id>/', eventos, name='eventos'),
path('ajax/upd_envio/', upd_envio, name='upd_envio'),
#path('sobre/', TemplateView.as_view(template_name="sobre.html")),
path('sobre/', licencia, name='licencia'),
]