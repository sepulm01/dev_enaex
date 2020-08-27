from django.apps import AppConfig
from django.utils.translation import ugettext_lazy as _

class CamarasConfig(AppConfig):
    name = 'camaras'
    verbose_name = _('Camaras')

    def ready(self):
        import camaras.signals  # noqa
        from camaras import update
        update.start()
