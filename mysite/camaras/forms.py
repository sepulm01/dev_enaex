from django import forms 
from . import models
from django.forms import ValidationError, Textarea, TextInput, CheckboxInput, Select, DateTimeInput
import datetime
import re
from django.core.exceptions import ValidationError
from django.utils.encoding import force_text
from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.models import User, Group
from django.utils.timezone import make_aware , now

class CamarasForm(forms.ModelForm):
    ''' Formulario para poder re editar un procedimiento cerrado, para ello se habilitó una vista Requerimientos Direcotor,
     donde el usuario que tenga perfil de director pueda volver el estado del código de cierre en null
     '''


    class Meta:
        fields = (
            'nombre',
            'estado',
            'sensib',
            'fuente',
            'actualizado',
            'areas',
            'op_ini',
            'op_fin',
            
            )
        model = models.Camara

        widgets = {
            'areas': forms.HiddenInput(),
            'estado': TextInput(attrs={'class':'form-control', 'readonly':'readonly'}),
            'nombre': TextInput(attrs={'class':'form-control'}),
            'fuente': TextInput(attrs={'class':'form-control'}),
            'sensib': TextInput(attrs={'class':'form-control'}),
            'op_ini': TextInput(attrs={'class':'form-control'}),
            'op_fin': TextInput(attrs={'class':'form-control'}),
            'actualizado': TextInput(attrs={'class':'form-control', 'readOnly':'true'}),
            }
        
class EventosForm(forms.ModelForm):
    class Meta:
        fields = (
              ##'activo', 'proced',
               'responsable', 'comentarios',
            )
        model = models.Eventos

        widgets = {

            #'activo':CheckboxInput(),
            #'proced':CheckboxInput(attrs={'style':"display:none"}),
            #'proced':CheckboxInput(),
            'responsable':TextInput(attrs={'class':'form-control'}),
            'comentarios':Textarea(attrs={'class':'form-control',
                                            'placeholder':'Ingrese aquí comentarios sobre la alarma gatillada si corresponde.',
                                            'rows':"4"}),
        }


