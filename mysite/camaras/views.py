from django.http import StreamingHttpResponse
import cv2
import threading
from django.views.decorators.gzip import gzip_page
#import imagezmq

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
import ffmpy
import os
from django.shortcuts import redirect

SCL = pytz.timezone(settings.TIME_ZONE)
path = '/home/martin/Documentos/dev_enaex/mysite/media/alarmas/'

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
    camaras = Camara.objects.all().order_by('id')
    df = pd.DataFrame(camaras.values('id','nombre',  'estado', 'sensib', 'fuente' ))
    df["id"] = df["id"].apply(lambda x: '<a href="/camara/{0}">{0}</a>'.format(x))
    html = df.to_html(index=False,
        escape=False,
        render_links=True,
        justify='center',
        classes=['table table-hover'],
        )
    return render(request, 'camaras_list.html', {'tabla':html})

#@login_required
def lista_alarmas(request):
    alarmas = Alarmas.objects.all().order_by('-tiempo')
    df = pd.DataFrame(alarmas.values('id','camara_id__nombre' ,'tiempo','clase','cantidad' ,'video', 'recibido',Fecha=Trunc('tiempo', 'second', tzinfo=SCL)))
    df['Fecha']=fformato(df['Fecha'])
    df["id"] = df["id"].apply(lambda x: '<a href="/video/{0}">{0}</a>'.format(x))
    df.rename(columns={
            'camara_id__nombre':'Cámara',
            'clase': 'Clases Objetos',
            'cantidad': 'Cantidad',
            }, inplace=True)
    df=df.drop(columns=['video', 'recibido','tiempo'])
    jtabla=df.to_json(orient='split', index=False,)
    col = []
    for i in df.columns:
        fo = {}
        fo["title"] = i
        col.append(fo)

    html = df.to_html(columns=['Cámara','Fecha','Cantidad','Clases Objetos','id'],
        index=False,
        escape=False,
        render_links=True,
        justify='center',
        classes=['table table-hover'],
        table_id='alarmas_id',
        max_rows=100,
        )
    return render(request, 'alarmas_list.html', {'tabla':html, 'jtabla':jtabla, 'col':col})

def borra_alarmas(request):
    q = request.POST
    lista = q.getlist('coment[]')
    l1 = []
    for i in lista:
        l1.append(int(i))
    print(type(l1),l1)
    selec = Alarmas.objects.filter(pk__in=l1)
    for i in selec.values('video'):
        print(i['video'])
        try:
            os.remove(path + i['video'])
        except:
            print("error borrando",i['video'])
            pass
    selec.delete()
    return redirect('/alarmas/')

def video(request, id):
    alarma = Alarmas.objects.get(pk=id)

    file = path + alarma.video
    output = path + 'output.mp4'
    try:
        os.remove(output)
    except:
        pass
    ff = ffmpy.FFmpeg( inputs={file: None}, outputs={output: None} )
    ff.run()
    video = '/media/alarmas/output.mp4'
    content = {
    'video':video,
     'id':alarma.id,
     'tiempo':alarma.tiempo,
     'camara':alarma.camara,
     }
    return render(request, 'video.html', content)
