import psutil
from django.contrib.auth.models import User, Group
from django.core.mail import EmailMessage, EmailMultiAlternatives
from .models import Ajustes

def seguimiento():
    pass

def espacio_d():
    disk_usage = psutil.disk_usage("/")
    total = to_gb(disk_usage.total)
    libre = to_gb(disk_usage.free)
    usado = to_gb(disk_usage.used)
    porcentaje= disk_usage.percent

    if porcentaje > 900:
        print("Espacio total: {:.2f} GB.".format(total))
        print("Espacio libre: {:.2f} GB.".format(libre))
        print("Espacio usado: {:.2f} GB.".format(usado))
        print("Porcentaje de espacio usado: {}%.".format(porcentaje))
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