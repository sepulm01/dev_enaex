# from django.http import StreamingHttpResponse
# import cv2
# import threading
# from django.views.decorators.gzip import gzip_page
#import imagezmq

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import get_user_model
from django.http import JsonResponse, HttpResponse
from .forms import CamarasForm, EventosForm
from .models import *
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
from django.utils.timezone import make_aware , now
from django.forms.models import model_to_dict
from .vivotek import VivotekCamera
import json
from datetime import datetime

SCL = pytz.timezone(settings.TIME_ZONE)
path = 'media/alarmas/'
#path = '/var/www/mysite/media/alarmas/'

coco=['persona', 'bicicleta', 'auto', 'moto', 'avión', 'autobús', 'tren', 'camión', 'barco', 'semáforo', 'boca de incendios', 'señal de parada', 'parquímetro', 'banco', 'pájaro', 'gato', 'perro',
 'caballo', 'oveja', 'vaca', 'elefante', 'oso', 'cebra', 'jirafa', 'mochila', 'paraguas', 'bolso', 'corbata', 'maleta', 'frisbee', 'esquís','bola de nieve','pelota' 'cometa', 'bate',
  'guante de béisbol', 'patineta', 'tabla de surf', 'raqueta', 'botella', 'copa de vino', 'taza', 'tenedor', 'cuchillo', 'cuchara', 'tazón', 'banana', 'sándwich', 'manzana', 'naranja',
   'brócoli', 'zanahoria', 'hot dog', 'pizza', 'donut', 'pastel', 'silla', 'sofá', 'maceta', 'cama', 'maceta', 'comedor', 'inodoro', 'tvmonitor', 'laptop', 'mouse', 'teclado remoto',
    'teléfono', 'celular', 'horno', 'microondas', 'tostador', 'fregadero', 'refrigerador', 'libro', 'reloj', 'florero', 'tijeras', 'peluche', 'secador de pelo', 'cepillo de dientes','NA',"NA"]

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
    icono_true = '<a class="btn-floating green"><i class="material-icons">check_circle</i></a>'
    icono_false = '<a class="btn-floating pulse red"><i class="material-icons">close</i></a>'
    camaras = Camara.objects.all().order_by('id')
    df = pd.DataFrame(camaras.values('id','nombre',  'estado', 'sensib', 'fuente' ))
    df["estado"] = df["estado"].apply(lambda x: icono_true if x else icono_false)
    df["id"] = df["id"].apply(lambda x: '<a href="/camara/{0}">{0}</a>'.format(x))
    df.rename(columns={
        'nombre':'Nombre',
        'estado': 'Estado',
        'sensib': 'Sensibilidad',
        'fuente':'Fuente de video',
        }, inplace=True)
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
    if len(alarmas):
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
        content = {'tabla':html, 'jtabla':jtabla, 'col':col}
    else:
        content = {'info':'Sin información'}

    return render(request, 'alarmas_list.html', content)

def borra_alarmas(request):
    q = request.POST
    lista = q.getlist('coment[]')
    l1 = []
    for i in lista:
        l1.append(int(i))
    #print(type(l1),l1)
    selec = Alarmas.objects.filter(pk__in=l1)
    for i in selec.values('video'):
        #print(i['video'])
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
    error_msg = ''
    try:
        os.remove(output)
    except:
        pass
    if os.path.isfile(file):
        
        try:
            ff = ffmpy.FFmpeg( inputs={file: None}, outputs={output: None} )
            ff.run()
        except:
            pass
            error_msg = 'error convirtiendo ffmpeg'
    else:
        error_msg = 'archivo %s no encontrado' %(file)
    video = '/media/alarmas/output.mp4'
    content = {
    'video':video,
     'id':alarma.id,
     'tiempo':alarma.tiempo,
     'camara':alarma.camara,
     'error_msg':error_msg,
     }
    return render(request, 'video.html', content)

@csrf_exempt
def cam_down(request):
    ''' vista que recibe las alarmas desde los workers o dockers (detector_multy)'''
    if request.method == 'POST':
        camara = request.POST['camara']
        fuente = request.POST['fuente']
        error_msg = request.POST['error_msg']
        secreto  = request.POST['secreto']
        if camara is not None:
            cam=Camara.objects.get(nombre=camara)
            if secreto == cam.secreto:
                txt = 'nodo %s sin señal de video de %s' % (camara,fuente)
                print(txt)
                cam.estado = False
                cam.error_msg = error_msg
                cam.save()
            else:
                print('Erro de validación en cam',cam.nombre)

    data = {
          "data": "data",
        }
    return JsonResponse(data, safe=False)

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
                #print('cam.secreto iguales en alarma_post',video)
                clases=[]
                clase = json.loads(clase)
                for c in clase:
                    #print(c,'c')
                    clases.append(coco[int(c)])
                nueva_alarma = Alarmas(
                                camara=cam,
                                tiempo=tiempo,
                                clase=clases,
                                cantidad=cantidad,
                                video=video)
                nueva_alarma.save()
                evento = Eventos.objects.filter(proced=True)
                #print("evento",evento)
                if len(evento)==0:
                    vivocam = VivotekCamera(host=cam.host,
                             port=cam.port,
                             usr=cam.usr,
                             pwd=cam.pwd,
                             digest_auth=cam.digest_auth,
                             ssl=cam.ssl,
                             verify_ssl=cam.verify_ssl,
                             sec_lvl=cam.sec_lvl)
                    #vivocam.do('do0',1)
            else:
                print('Erro de validación en cam',cam.nombre)

    data = {
          "data": "data",
        }
    return JsonResponse(data, safe=False)

@csrf_exempt
def camviva_post(request):
    ''' vista que recibe las alarmas desde los workers o dockers (detector_multy)'''
    if request.method == 'POST':
        camara = request.POST['camara']
        tiempo = request.POST['tiempo']
        image = request.POST['image']
        image=bytearray(image,'ISO-8859-1') 
        #print('image', type(image))
        secreto  = request.POST['secreto'] 
        lic  = request.POST['lic']
        if camara is not None:
            cam=Camara.objects.get(nombre=camara)
            if secreto == cam.secreto:
                print('cam.secreto iguales en camviva_post')
                cam.actualizado = tiempo
                cam.image =  image
                cam.estado = True
                cam.error_msg = ''
                cam.save()
            else:
                print('Erro de validación en camviva_post, cámara:',cam.nombre)
        if lic is not None: #2020-12-12T07:00:00Z
            d1 = datetime.strptime(lic,"%Y-%m-%dT%H:%M:%SZ") 
            lic=Licencia.objects.all()[0]
            lic.fecha_caducidad = d1
            lic.save()
            #print(lic,type(lic),"esto es Lic"*100)

    data = {
          "data": "data",
        }
    return JsonResponse(data, safe=False)

@csrf_exempt
def cam_disp(request,id):
    ''' vista que recibe las alarmas desde los workers o dockers (detector_multy)'''
    data = {'data':'data'}
    if request.method == 'GET':
        if request.GET['a']=='2485987abr':
            cam=Camara.objects.get(pk=id)
            lic=Licencia.objects.all()[0]
            data = {
                  'pk': cam.pk,
                  'nombre': cam.nombre,
                  'sensib': cam.sensib,
                  'fuente': cam.fuente,
                  'actualizado': cam.actualizado,
                  'areas': cam.areas,
                  'op_ini': cam.op_ini,
                  'op_fin': cam.op_fin,
                  'secreto': cam.secreto,
                  #'url_alarm': cam.url_alarm,
                  'detect_todo': cam.detect_todo,
                  'micw': cam.min_contour_width,
                  'mich': cam.min_contour_height,
                  'macw': cam.max_contour_width,
                  'mach': cam.max_contour_height,
                  'dist_bg': cam.dist_bg,
                  'rep_alar_ni': cam.rep_alar_ni,
                  'min_area_mean': cam.min_area_mean,
                  'time_min' : cam.time_min,
                  'fecha_caducidad' : lic.fecha_caducidad,
                  'clave': lic.clave,
                }

    return JsonResponse(data, safe=False)

@login_required
def alarma_inst(request):
    ''' Revisa si la necesidad de apoyo fue solucionada '''
    alarmas = []
    evento = ''
    eve = Eventos.objects.filter(activo=True, proced=False)
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
    if len(eve):
        df1 = pd.DataFrame(eve.values('pk','t_ini','t_fin','activo','proced','estado'))
        df1["activo"] = df1["activo"].apply(lambda x: icono_al if x else "-")
        df1["proced"] = df1["proced"].apply(lambda x: icono_pr if x else "-")
        if len(eve) > 1:
            df1["t_fin"] = pd.DatetimeIndex(df1["t_fin"]).tz_convert('America/Santiago').strftime('%d/%m/%Y %H:%M')
        df1["t_ini"] = pd.DatetimeIndex(df1["t_ini"]).tz_convert('America/Santiago').strftime('%d/%m/%Y %H:%M')
        df1["t_ini"] = df1['pk'].apply(lambda x: '<a href="eventos/{0}">{1}</a>'.format(x,df1[df1['pk']==x].t_ini.values[0]))
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
        content = {
        "tabla":html,
        }
    else:
        content = {'info':'Sin información'}
    return render(request, 'eventos_list.html',content)

@login_required
def eventos(request, id):
    estado="Inactivo"
    escala = 0
    instance = get_object_or_404(Eventos, pk=id)
    reg = RegAcciones.objects.filter(evento=instance).order_by('-tiempo')
    if len(reg):
        df1 = pd.DataFrame(reg.values('tiempo','accion'))
        df1["tiempo"] = pd.DatetimeIndex(df1["tiempo"]).tz_convert('America/Santiago').strftime('%d/%m/%Y %H:%M')
        acciones_tbl = df1.to_html(
                #columns=['Fecha hora inicio','Fecha hora cierre','Alarma activa','En prodedimiento','Nivel escalamiento'],
                classes=["table table-striped "],
                border=0 , justify='center',
                table_id="tabla_ala",
                index=False,
                escape=False,
                render_links=True)
    else:
        acciones_tbl = '<p>No hay acciones tomadas, verifique que existan responsables asignados\
         a cada nivel de escalamiento para su aviso.<p>'
    ala = Alarmas.objects.filter(evento=instance).order_by('-tiempo')
    df2 = pd.DataFrame(ala.values('tiempo','camara','clase','cantidad','video','pk'))
    df2["video"] = df2["pk"].apply(lambda x: '<a href="/video/{0}">{1}</a>'.format(x,df2[df2['pk']==x].video.values[0]))
    df2["tiempo"] = pd.DatetimeIndex(df2["tiempo"]).tz_convert('America/Santiago').strftime('%d/%m/%Y %H:%M')
    alarmas_tbl = df2.to_html(
            columns=['tiempo','camara','clase','cantidad','video'],
            classes=["table table-striped"],
            border=0 , justify='center',
            table_id="tabla_eve",
            index=False,
            escape=False,
            render_links=True)
    if request.method == 'POST':
        form = EventosForm(request.POST, instance=instance)
        form.save()
        if form.is_valid():
            #print("paso x aquí EventosForm",)
            # redirect to a new URL:
            return redirect('eventos_list')
        else:
            print(request.POST, form.is_valid())
    else:
        form = EventosForm(instance=instance)

    cla = {
    0:'btn btn-danger btn-lg btn-block red',
    1:'btn btn-warning btn-lg btn-block yellow',
    2:'btn btn-secondary btn-lg btn-block grey' ,
    }

    msg = {
    0:'Alarma activada',
    1:'En procedimiento' ,
    2:'Finalizado' ,
    }

    if instance.activo==True and instance.proced==False:
        mensaje=msg[0]
        clase = cla[0]
        val_btn = 0
    elif instance.activo==True and instance.proced==True:
        mensaje=msg[1]
        clase = cla[1]
        val_btn = 1
    elif instance.activo==False and instance.proced==False:
        mensaje=msg[2]
        clase = cla[2]
        val_btn = 2
    else:
        print("alarma",instance.activo,"\nprocedimiento",instance.proced)
        mensaje = "Alm inactiva/proced activo "
        clase = cla[2]
        val_btn = 3
    content = {
    'mensaje':mensaje,
    'clase':clase,
    'val_btn':val_btn,
    'inicio':instance.t_ini,
    'fin':instance.t_fin,
    'eve_id': instance.pk,
    'escala':instance.estado,
    'form':form,
    'acciones':acciones_tbl,
    'alarmas':alarmas_tbl,
    }
    return render(request, 'eventos.html', content)

@login_required        
def upd_envio(request):
    '''rutina AJAX para actualizar ESTADO de la Unidad enviada.'''
    estado = request.GET.get('estado')
    eve_id = request.GET.get('eve_id')
    #print('estado',estado, 'algo')

    eve = get_object_or_404(Eventos, pk=eve_id)
    if estado == "1":
        eve.proced=True
        eve.save()
        apaga_do()
    elif estado == "2":
        eve.proced=False
        eve.activo=False
        eve.t_fin=now()
        eve.save()
    else:
        pass
        #print("NO PESCO ")

    jgps = {"estado":estado,
        'req_id':eve.pk,
    }

    return JsonResponse(jgps)

def apaga_do():
    ''' Rutina para apagar las señal digital output de todas las cámaras'''
    pass
    camaras=Camara.objects.all()
    for cam in camaras:
        vivocam = VivotekCamera(host=cam.host,
         port=cam.port,
         usr=cam.usr,
         pwd=cam.pwd,
         digest_auth=cam.digest_auth,
         ssl=cam.ssl,
         verify_ssl=cam.verify_ssl,
         sec_lvl=cam.sec_lvl)
        vivocam.do('do0',0)

def licencia(request):
    lic=Licencia.objects.all()[0]
    content = {
    'fecha': lic.fecha_caducidad
    }
    return render(request, 'sobre.html', content)
