from camaras.models import Alarmas, Eventos, RegAcciones, Responsables
from django.db.models.signals import post_save, post_delete, pre_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from django.utils.timezone import make_aware , now
from twilio.rest import Client
import datetime

from django.core.mail import EmailMessage, EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags


@receiver(post_save, sender=Alarmas)
def alarma_gatillada(sender, instance, created, **kwargs):
    num = '+56949010958'
    eve = Eventos.objects.filter(activo=True)
    resp = Responsables.objects.filter(estado=True)
    niv0 = []
    for r in resp:
        if r.nivel == 0: niv0.append(r)

    if len(eve)==0:
        e = Eventos()
        e.save()
        instance.evento = e
        instance.save()
        print("Nuevo evento creado",e.pk)
        mensaje = 'Nueva alarma gatillada ' + str(e.pk)
        mensajeria(niv0,mensaje,e)
        #wapp(num,'Nueva alarma gatillada\n'+str(instance.tiempo)+' '+str(instance.camara) ,e)
    else:
        if instance.evento is None:
            #and instance.tiempo >= eve[0].t_ini:
            print(eve,eve[0],type(eve[0]),len(eve))
            instance.evento=eve[0]
            instance.save()
            print('viejo evento con alarma agregada')
        else:
            print('no se cumplio')

def escalamiento():
    eve = Eventos.objects.filter(activo=True)
    resp = Responsables.objects.filter(estado=True)
    niv1 = []
    niv2 = []
    niv3 = []
    for r in resp:
        if r.nivel == 1: niv1.append(r)
        elif r.nivel == 2: niv2.append(r)
        elif r.nivel == 3: niv3.append(r)
    if len(eve)==1:
        eve = eve[0]
        if eve.proced==0:
            if eve.t_ini + datetime.timedelta(minutes=5) < now() and eve.estado==0:
                mensaje = 'escalamiento nivel 1'
                mensajeria(niv1,mensaje,eve)
                eve.estado = 1
                eve.save()
            if eve.t_ini + datetime.timedelta(minutes=10) < now() and eve.estado==1:
                mensaje = 'escalamiento nivel 2'
                mensajeria(niv2,mensaje,eve)
                eve.estado = 2
                eve.save()
            if eve.t_ini + datetime.timedelta(minutes=15) < now() and eve.estado==2:
                mensaje = 'escalamiento nivel 3'
                mensajeria(niv3,mensaje,eve)
                eve.estado = 3
                eve.save()
        else:
            print('pasa x eve.proced 1')
            if eve.t_ini + datetime.timedelta(minutes=60) < now():
                eve.activo = False
                eve.t_fin = now()
                eve.save()
                reg=RegAcciones(
                    evento=eve,
                    accion="Evento cerrado automaticamente por encontrarse en procedimiento, más de una hora\
                    desde su inicio.")
                reg.save()
                print("procedimiento cerrado")


def mensajeria(lista,mensaje,evento):
    for n in lista:
        if n.tipo=='SMS':
            num =n.fono
            wapp(num,mensaje, evento)
        elif n.tipo=='EMAIL':
            mail_subject = mensaje
            message = mail_subject
            envia_mail(n.mail,mail_subject, message,evento)
            print(n.mail)

def envia_mail(to_email,mail_subject,message,evento):
    email=EmailMultiAlternatives(mail_subject, message, to=[to_email])
    email.attach_alternative(message, 'text/html')
    try:
        email.send()
        reg=RegAcciones(evento=evento,accion="mail enviado a:"+to_email+"\n"+message)
        reg.save()
    except Exception as e:
        print("Parece que hay un error en envia_mail:",e)

def wapp(num, txt ,evento):
    ''' Función que envía los avisos por Whatsapp '''
    # Your Account Sid and Auth Token from twilio.com/console
    # DANGER! This is insecure. See http://twil.io/secure

    client = Client(account_sid, auth_token)
    #para = 'whatsapp:' + num
    para = str(num)
    print(para, 'numero')
    try:
        message = client.messages.create(
                  from_='+18133286552',
                  body =txt,
                  to='+'+para
                                  )
        print(message.sid)
        reg=RegAcciones(evento=evento,accion="SMS enviado a:"+num+"\n"+txt+",\n SID: "+message.sid)
        reg.save()
    except Exception as e:
        print("Parece que hay un error en wapp:",e)
