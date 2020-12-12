import psutil
from django.contrib.auth.models import User, Group
from .models import *
from django.core.mail import EmailMessage, EmailMultiAlternatives
from .models import Ajustes
import datetime
from django.utils.timezone import make_aware , now
import time
from .vivotek import VivotekCamera
import json

def seguimiento():
    pass

def espacio_d():
    disk_usage = psutil.disk_usage("/")
    total = to_gb(disk_usage.total)
    libre = to_gb(disk_usage.free)
    usado = to_gb(disk_usage.used)
    porcentaje= disk_usage.percent

    if porcentaje > 90:
        #print("Espacio total: {:.2f} GB.".format(total))
        #print("Espacio libre: {:.2f} GB.".format(libre))
        #print("Espacio usado: {:.2f} GB.".format(usado))
        #print("Porcentaje de espacio usado: {}%.".format(porcentaje))
        user = User.objects.all()
        for u in user:
            if u.is_superuser and u.email is not None:
                message = "El porcentaje de espacio usado app de cámaras de seguridad es: {}%.\
                            por favor tome las acciones pertinentes para evitar el llenado de los discos.\n \
                            Este mensaje le ha llegado a usted porque figura como administrador en el sistema.".format(porcentaje)
                mail_subject = "Información importante de espacio en disco."
                to_email = u.email
                #envia_mail(to_email,mail_subject,message)
    return (total,libre, usado, porcentaje)


def to_gb(bytes):
    "Convierte bytes a gigabytes."
    return bytes / 1024**3


def envia_mail(to_email,mail_subject,message):
    email=EmailMultiAlternatives(mail_subject, message, to=[to_email])
    email.attach_alternative(message, 'text/html')
    try:
        email.send()
    except Exception as e:
        print("Parece que hay un error en envia_mail:",e)

def check_cam():
    camaras = Camara.objects.all().order_by('id')
    for cam in camaras:
        #print (cam.nombre, cam.estado)
        if cam.actualizado < now()-datetime.timedelta(minutes=2):
            cam.estado= False
            diferencia = now() - cam.actualizado
            cam.error_msg = 'Sin actualización de nodo desde {}'.format(diferencia)
            cam.save()
            #print ('Cambio en',cam.nombre, cam.estado, cam.error_msg)
        else:
            if cam.host is not '':
                vivocam = VivotekCamera(host=cam.host,
                                 port=cam.port,
                                 usr=cam.usr,
                                 pwd=cam.pwd,
                                 digest_auth=cam.digest_auth,
                                 ssl=cam.ssl,
                                 verify_ssl=cam.verify_ssl,
                                 sec_lvl=cam.sec_lvl)
                lar, alt = 608, 608
                factorx = 9999/lar
                factory = 9999/alt
                areas = json.loads(cam.areas)
                for i,a in enumerate(areas):
                    lista = []
                    for aa in a:
                        lista.append(int(aa['x']*factorx))
                        lista.append(int(aa['y']*factory))
                    listToStr = ','.join([str(elem) for elem in lista])
                    if i <=4:
                        vivocam.set_param('motion_c0_win_i'+str(i)+'_enable','1')
                        vivocam.set_param('motion_c0_win_i'+str(i)+'_name',i)
                        vivocam.set_param('motion_c0_win_i'+str(i)+'_polygonstd',listToStr)
                        vivocam.set_param('motion_c0_profile_i0_win_i'+str(i)+'_name',i)
                        vivocam.set_param('motion_c0_profile_i0_win_i'+str(i)+'_polygonstd',listToStr)
                    
