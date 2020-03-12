from django.db import models

# Create your models here.
class Camara(models.Model):
    nombre = models.CharField(max_length=10)
    estado = models.BooleanField(default=False)
    sensib = models.IntegerField()
    fuente = models.CharField(max_length=300)
    image = models.BinaryField(blank=True)
    actualizado = models.DateTimeField('Actualizado',blank=True, null=True,auto_now_add=False)
    areas = models.TextField('Areas',blank=True, null=True)

    class Meta:
        verbose_name_plural = "Cámaras"

    def __str__(self):
        return str(self.nombre)

class Alarmas(models.Model):
    camara = models.ForeignKey('Camara', on_delete=models.PROTECT)
    tiempo = models.DateTimeField('Fecha hora',auto_now_add=False)
    clase = models.CharField(max_length=300)
    cantidad = models.IntegerField()
    video = models.CharField(max_length=300)
    recibido = models.DateTimeField('Recibido',blank=True, null=True,auto_now_add=False)

    class Meta:
        verbose_name_plural = "Alarmas"

    def __str__(self):
        return str(self.camara)