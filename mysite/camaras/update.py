from apscheduler.schedulers.background import BackgroundScheduler
from camaras.utils import seguimiento
from camaras.signals import escalamiento


def start():
    scheduler = BackgroundScheduler()
    scheduler.add_job(escalamiento,'interval',minutes=1)
    scheduler.start()