"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path 
from django.conf.urls import include, url
from camaras.views import home, camara, lista_camaras, lista_alarmas, video, borra_alarmas
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    #path('accounts/', include('accounts.urls')), # new
    path('accounts/', include('django.contrib.auth.urls')),
    path('', include('camaras.urls')),
    path('accounts/login/', auth_views.LoginView.as_view()),
    path('change-password/', auth_views.PasswordChangeView.as_view()),
    path('admin/', admin.site.urls),
    path('', home, name='home'),
    path('camara/<int:id>', camara, name='camara'),
    path('camara/', lista_camaras, name='lista_camaras'),
    path('alarmas/', lista_alarmas, name='lista_alarmas'),
    path('borra_alarmas/', borra_alarmas, name='borra_alarmas'),
    path('video/<int:id>/', video, name='video'),
]


urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


