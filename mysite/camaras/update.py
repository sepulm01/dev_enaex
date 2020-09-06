from apscheduler.schedulers.background import BackgroundScheduler
from camaras.utils import espacio_d, check_cam
from camaras.signals import escalamiento


def start():
    scheduler = BackgroundScheduler()
    scheduler.add_job(escalamiento,'interval',minutes=1)
    scheduler.add_job(espacio_d,'interval',minutes=30)
    scheduler.add_job(check_cam,'interval',minutes=1)
    scheduler.start()