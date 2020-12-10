from django.contrib import admin
from .models import Camara, Alarmas, Eventos, RegAcciones, Responsables, RegAcciones, Licencia
from django.utils.safestring import mark_safe
import base64
from .views import path, borra_video_zombi
# Register your models here.

#admin.site.register(Camara)

#admin.site.register(Alarmas)

admin.site.site_header = "Seguridad - HRGestión"
admin.site.index_title = "Menú"
admin.site.site_url = '/'
admin.site.disable_action('delete_selected')

def borra_alarmas(modeladmin, request, queryset):
    #print(queryset)
    for i in queryset.values('video'):
        print(i['video'], path)
        try:
            os.remove(path + i['video'])
        except:
            print("error borrando",i['video'])
            pass
    queryset.delete()
    borra_video_zombi()
borra_alarmas.short_description = "Borra alarmas y arch."

class AlarmasInline(admin.TabularInline):
    model = Alarmas
    extra = 0
    list_display = ['camara','tiempo','clase','cantidad', 'video']
    readonly_fields = ('camara','tiempo','clase','cantidad', 'video','recibido',)

class RegAccInline(admin.TabularInline):
    model = RegAcciones
    extra = 0
    list_display = ['tiempo','accion','evento',]
    readonly_fields = ('tiempo','accion','evento',)

@admin.register(RegAcciones)
class RegAcciones(admin.ModelAdmin):
    list_display = ['tiempo','accion','evento',]
    readonly_fields = ('tiempo','accion','evento',)

@admin.register(Licencia)
class Licencia(admin.ModelAdmin):
    list_display = ['fecha_caducidad',]
    readonly_fields = ('fecha_caducidad',)

@admin.register(Eventos)
class EventosAdmin(admin.ModelAdmin):
    list_display = ['t_ini','t_fin','responsable','activo'] 
    inlines = [ RegAccInline,AlarmasInline, ]
    actions = ['delete_selected']

@admin.register(Alarmas)
class AlarmasAdmin(admin.ModelAdmin):
    list_display = ['camara','tiempo','clase','cantidad', 'video'] 
    actions = [borra_alarmas]

@admin.register(Responsables)
class ResponsAdmin(admin.ModelAdmin):
    list_display = ['nombre','estado','fono','mail','nivel', 'tipo', 'endpoint',] 

@admin.register(Camara)
class CamarasAdmin(admin.ModelAdmin):
    change_form_template = "admin/camaras/cam_form.html"
    list_display = ['nombre','estado','sensib','fuente','detect_todo','op_ini','op_fin'] 
    readonly_fields = ('actualizado', 'imagen', 'secreto', 'error_msg', 'estado','cam_model' )


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
                            'fields': (
                                'nombre',
                                ('estado','error_msg'),
                                'detect_todo',
                                'sensib',
                                'fuente',
                                'actualizado',
                                #'url_alarm',
                                'op_ini',
                                'op_fin',
                                'imagen',
                                #'secreto',

                                        )}),
                ('Background Sub',{
                    'classes':('collapse', 'open'),
                    'fields': ( 'min_contour_width',
                                'min_contour_height',
                                'max_contour_width',
                                'max_contour_height',
                                'dist_bg',
                                'rep_alar_ni', 'min_area_mean',  'time_min')
                    }),
                ('Cámara Vivotek',{
                    'classes':('collapse', 'open'),
                    'fields': ( 'cam_model',
                                'host',
                                'port',
                                'usr',
                                'pwd',
                                'digest_auth',
                                'ssl',
                                'verify_ssl',
                                'sec_lvl',
                                )
                    }),
                )



    