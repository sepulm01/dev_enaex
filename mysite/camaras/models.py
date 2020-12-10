from django.db import models
from django.contrib.auth.models import User, Group
import uuid
from django import forms

# Create your models here.
class Camara(models.Model):
    nombre = models.CharField(max_length=10)
    estado = models.BooleanField(default=False)
    sensib = models.IntegerField('Sensibilidad det')
    fuente = models.CharField(max_length=300)
    image = models.BinaryField(blank=True)
    actualizado = models.DateTimeField('Actualizado',blank=True, null=True,auto_now_add=False)
    areas = models.TextField('Areas',blank=True, null=True)
    op_ini = models.TimeField('Inicio turno',blank=True, null=True,auto_now_add=False, editable=True)
    op_fin = models.TimeField('Fin turno',blank=True, null=True,auto_now_add=False, editable=True)
    secreto = models.CharField(max_length=50, default=uuid.uuid4().hex, null=False, blank=False, editable=False, unique=False)
    #url_alarm = models.CharField(max_length=300,)
    detect_todo = models.BooleanField('Detecta toda la pantalla',default=False)
    min_contour_width=models.IntegerField('min_contour_width', default=20,null=False, blank=False)
    min_contour_height=models.IntegerField('min_contour_height', default=20,null=False, blank=False)
    max_contour_width=models.IntegerField('max_contour_width', default=590,null=False, blank=False)
    max_contour_height=models.IntegerField('max_contour_height', default=590,null=False, blank=False)
    dist_bg = models.IntegerField('Distancia recorrida px', default=290)
    rep_alar_ni = models.IntegerField('Rep alarm en fr', default=20)
    error_msg = models.CharField('Msg error',blank=True, null=True,max_length=300)
    min_area_mean = models.IntegerField('Área min prom', blank=True, null=True)
    time_min = models.TimeField('Tiempo mín',auto_now=False, auto_now_add=False, blank=True, null=True  )
    #Parametros vivotek
    cam_model = models.CharField('Modelo',blank=True, null=True,max_length=30)
    host=models.CharField('cam host',blank=True, null=True,max_length=15, default='0.0.0.0')
    port=models.IntegerField('cam port', default=443)
    usr=models.CharField('cam usr',blank=True, null=True,max_length=25, default='worker_st')
    pwd=models.CharField('cam pwd',blank=True, null=True,max_length=35)
    digest_auth=models.BooleanField('cam digest_auth',default=True)
    ssl=models.BooleanField('cam ssl',default=False)
    verify_ssl=models.BooleanField('cam verify_ssl',default=False)
    sec_lvl=models.CharField('cam sec_lvl',blank=True, null=True,max_length=25, default='operator')

    class Meta:
        verbose_name_plural = "Cámaras"

    # def save(self, *args, **kwargs):
    #     self.secreto = uuid.uuid4().hex
    #     super(Camara, self).save(*args, **kwargs)

    def __str__(self):
        return str(self.nombre)

class Alarmas(models.Model):
    camara = models.ForeignKey('Camara', on_delete=models.PROTECT)
    tiempo = models.DateTimeField('Fecha hora',auto_now_add=False)
    clase = models.CharField(max_length=300)
    cantidad = models.IntegerField()
    video = models.CharField(max_length=300)
    recibido = models.DateTimeField('Recibido',blank=True, null=True,auto_now_add=False)
    evento = models.ForeignKey('Eventos',blank=True, null=True, on_delete=models.PROTECT)


    class Meta:
        verbose_name_plural = "Alarmas"

    def __str__(self):
        return str(self.camara)

class Eventos(models.Model):
    """docstring for ClassName"""
    t_ini = models.DateTimeField('Fecha hora inicio',auto_now_add=True)
    t_fin = models.DateTimeField('Fecha hora termino',auto_now_add=False, blank=True, null=True)
    activo = models.BooleanField('Alarma activa',default=True)
    proced = models.BooleanField('En prodedimiento',default=False)
    estado = models.IntegerField('Nivel escalamiento',default=0)
    #responsable = models.ForeignKey(User,blank=True, null=True,on_delete=models.PROTECT)
    responsable = models.TextField('Responsable',blank=True, null=True)
    comentarios = models.TextField('Comentarios',blank=True, null=True)

    class Meta:
        verbose_name_plural = "Eventos"

    def __str__(self):
        return str(self.t_ini)

class RegAcciones(models.Model):
    tiempo = models.DateTimeField('Fecha hora accion',auto_now_add=True)
    accion = models.CharField('Acciones',max_length=300)
    evento = models.ForeignKey('Eventos',blank=True, null=True, on_delete=models.PROTECT)

    class Meta:
        verbose_name_plural = "Acciones"

class Responsables(models.Model):
    TIPO = [
        ('SMS','SMS'),
        ('EMAIL','EMAIL'),
        ('ENDPOINT','ENDPOINT'),
         ]

    LEVEL = [
        (0,'Nivel 0'),
        (1,'Nivel 1'),
        (2,'Nivel 2'),
        (3,'Nivel 3'),
         ]
    nombre = models.CharField('Nombre',max_length=100)
    fono = models.CharField('Fono cell',max_length=20,blank=True, null=True,)
    mail = models.CharField('eMail',max_length=90,blank=True, null=True,)
    nivel = models.IntegerField('Nivel alarma',choices=LEVEL ,default=0)
    tipo = models.CharField("Tipo acciones",choices=TIPO ,max_length=60, unique=False)
    endpoint = models.CharField('uri',max_length=300, blank=True, null=True,)
    estado = models.BooleanField('Estado',default=True)

    class Meta:
        verbose_name_plural = "Responsables"

class Ajustes(models.Model):
    """docstring for ClassName"""
    uso_disco = models.IntegerField('Uso disco video' ,default=90)

    class Meta:
        verbose_name_plural = "Ajustes"

class Licencia(models.Model):
    fecha_caducidad = models.DateTimeField('Fecha de caducidad',auto_now_add=False)
    clave = models.TextField('clave',max_length=256)
        