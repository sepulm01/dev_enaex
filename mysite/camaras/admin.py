from django.contrib import admin
from .models import Camara, Alarmas
from django.utils.safestring import mark_safe
import base64
# Register your models here.

#admin.site.register(Camara)

#admin.site.register(Alarmas)

admin.site.site_header = "Seguridad - HRGestión"
admin.site.index_title = "Menú"
admin.site.site_url = '/'


@admin.register(Alarmas)
class AlarmasAdmin(admin.ModelAdmin):
    list_display = ['camara','tiempo','clase','cantidad', 'video'] 

@admin.register(Camara)
class CamarasAdmin(admin.ModelAdmin):
    change_form_template = "admin/camaras/cam_form.html"
    list_display = ['nombre','estado','sensib','fuente'] 
    readonly_fields = ('actualizado', 'imagen' )

    def imagen(self, obj):
        #return mark_safe('<img src="{url}" width="{width}" height={height} />'.format(
        return mark_safe('<br/><canvas id="canvas_f" width={width}  height={height} style="background: url({url})"></canvas><br/><button id="clear">Clear Canvas</button><button id="guardar">Guardar</button>'.format(
            url =   "data:image/png;base64,%s" % base64.b64encode(obj.image).decode("utf-8") ,
            width=412,
            height=412,
            ))

    fieldsets = (
                ('Camaras', {
                            #'fields': ('autor','estado','operador',( 'tipo','clasifi', 'urgencia'),
                            'fields': ('nombre','estado','sensib','fuente','actualizado','imagen'
                                        )}),
                )



    