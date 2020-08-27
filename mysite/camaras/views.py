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
from .models import Camara, Alarmas, Eventos
import pandas as pd
import base64
from django.conf import settings
import pytz
from django.db.models.functions import Trunc
import ffmpy
import os
from os import walk
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt

SCL = pytz.timezone(settings.TIME_ZONE)
path = '/home/martin/Documents/dev_enaex/mysite/media/alarmas/'

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
    imagen =  '<canvas id="canvas" width=608  height=608 style=" background: url(data:image/png;base64,%s) "></canvas>' % base64.b64encode(instance.image).decode("utf-8")
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
    borra_video_zombi()
    return redirect('/alarmas/')

def borra_video_zombi():
    lfiles = []
    lalarm = []
    alarmas = Alarmas.objects.all().order_by('-tiempo')
    for (dirpath, dirnames, filenames) in walk(path):
        lfiles.extend(filenames)
        break
    for a in alarmas:
        lalarm.append(a.video)
    for file in lfiles:
        if file not in lalarm:
            try:
                os.remove(path + file)
            except:
                print("error borrando",file)
                print(file,'no esta')


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

@csrf_exempt
def alarma_post(request):
    ''' vista que recibe las alarmas desde los workers o dockers (detector_multy)'''

    if request.method == 'POST':
        camara = request.POST['camara']
        tiempo = request.POST['tiempo']
        clase = request.POST['clase']
        cantidad = request.POST['cantidad']
        video = request.POST['video']
        secreto  = request.POST['secreto']
        if camara is not None:
            cam=Camara.objects.get(nombre=camara)
            if secreto == cam.secreto:
                print('cam.secreto iguales',video)
                nueva_alarma = Alarmas(
                                camara=cam,
                                tiempo=tiempo,
                                clase=clase,
                                cantidad=cantidad,
                                video=video)
                nueva_alarma.save()
            else:
                print('Erro de validación en cam',cam.nombre)

    data = {
          "data": "data",

        }
    return JsonResponse(data, safe=False)

@login_required
def alarma_inst(request):
    ''' Revisa si la necesidad de apoyo fue solucionada '''
    alarmas = []
    evento = ''
    eve = Eventos.objects.filter(activo=True)
    if len(eve) > 0:
        eve = eve[0]
        evento = eve.pk
        cam = Alarmas.objects.filter(evento=eve)
        for c in cam:
            alarmas.append(str(c.camara))
        alarmas = list(set(alarmas))

    data = {
    'alarmas': alarmas,
    'evento': evento,
    }
    return JsonResponse(data, safe=False)

@login_required
def eventos_list(request): 
    icono_al = '<a class="btn-floating pulse red"><i class="material-icons">transfer_within_a_station</i></a>'
    icono_pr = '<a class="btn-floating blue"><i class="material-icons">star_border</i></a>'
    eve = Eventos.objects.all().order_by('-t_ini')
    df1 = pd.DataFrame(eve.values('pk','t_ini','t_fin','activo','proced','estado'))
    df1["activo"] = df1["activo"].apply(lambda x: icono_al if x else "-")
    df1["proced"] = df1["proced"].apply(lambda x: icono_pr if x else "-")
    df1["t_fin"] = pd.DatetimeIndex(df1["t_fin"]).tz_convert('America/Santiago').strftime('%d/%m/%Y %H:%M')
    df1["t_ini"] = pd.DatetimeIndex(df1["t_ini"]).tz_convert('America/Santiago').strftime('%d/%m/%Y %H:%M')
    df1["t_ini"] = df1['pk'].apply(lambda x: '<a href="admin/camaras/eventos/{0}">{1}</a>'.format(x,df1[df1['pk']==x].t_ini.values[0]))
    df1.rename(columns={
            't_ini':'Fecha hora inicio',
            't_fin': 'Fecha hora cierre',
            'activo': 'Alarma activa',
            'proced':'En prodedimiento',
            'estado':'Nivel escalamiento',
            }, inplace=True)
    html = df1.to_html(
            columns=['Fecha hora inicio','Fecha hora cierre','Alarma activa','En prodedimiento','Nivel escalamiento'],
            classes=["striped, highlight "],
            border=0 , justify='center',
            table_id="tabla_eve",
            index=False,
            escape=False,
            render_links=True)
    return render(request, 'eventos_list.html', { "tabla":html, "eve":eve })

@login_required
def eventos(request, id): 
    eve = Eventos.objects.filter(pk=id).values()[0]
    content = {
    'eventos':eve
    }
    return render(request, 'eventos.html', content)