from django.http import StreamingHttpResponse
import cv2
import threading
from django.views.decorators.gzip import gzip_page
import imagezmq

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import get_user_model
from django.http import JsonResponse, HttpResponse
from .forms import CamarasForm
from .models import Camara, Alarmas
import pandas as pd
import base64
from django.conf import settings
import pytz
from django.db.models.functions import Trunc

SCL = pytz.timezone(settings.TIME_ZONE)

def fformato(serie):
    #if  serie.dtype.str != '|O':
    serie=serie.dt.strftime('%d/%m/%Y %H:%M:%S')
    return serie

@login_required
def home(request):
    request.session.set_expiry(0) 
    return render(request, "home.html", {} )

@login_required
def camara(request,id):
    instance = get_object_or_404(Camara, pk=id)
    if request.method == 'POST':
        form = CamarasForm(request.POST, instance=instance)
        form.save()
        if form.is_valid():
            print("paso x aquí EnvioForm",)
            # redirect to a new URL:
            return redirect('lista_camaras')
        else:
            print(request.POST, form.is_valid())
    else:
        form = CamarasForm(instance=instance)
    imagen =  '<canvas id="canvas" width=416  height=416 style=" background: url(data:image/png;base64,%s) "></canvas>' % base64.b64encode(instance.image).decode("utf-8")
    campos = {
    'form': form,
    'imagen': imagen,
     }
    return render(request, 'camaras.html', campos)

@login_required
def lista_camaras(request):
    camaras = Camara.objects.all()
    df = pd.DataFrame(camaras.values('id','nombre',  'estado', 'sensib', 'fuente' ))
    df["id"] = df["id"].apply(lambda x: '<a href="/camara/{0}">{0}</a>'.format(x))
    html = df.to_html(index=False,
        escape=False,
        render_links=True,
        justify='center',
        classes=['table table-hover'],
        )
    return render(request, 'camaras_list.html', {'tabla':html})

@login_required
def lista_alarmas(request):
    alarmas = Alarmas.objects.all().order_by('-tiempo')
    df = pd.DataFrame(alarmas.values('id','camara_id__nombre' ,'tiempo','clase','cantidad' ,'video', 'recibido',Fecha=Trunc('tiempo', 'second', tzinfo=SCL)))
    df['Fecha']=fformato(df['Fecha'])
    df.rename(columns={
            'camara_id__nombre':'Cámara',
            'clase': 'Clases Objetos',
            'cantidad': 'Cantidad',
            }, inplace=True)
    html = df.to_html(columns=['Cámara','Fecha','Cantidad','Clases Objetos',],
        index=False,
        escape=False,
        render_links=True,
        justify='center',
        classes=['table table-hover'],
        )
    return render(request, 'alarmas_list.html', {'tabla':html})