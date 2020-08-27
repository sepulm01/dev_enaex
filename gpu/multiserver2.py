#!/usr/bin/env python

'''
Recdata SPA.
Feb - 2019
Programa de servidor de inferencias StreetFlow.
Version 2.

'''

# import socket programming library 
import socket
import cv2 
import numpy as np
import random
import datetime 
from datetime import timedelta
from _thread import *
import threading 
from drk import Drk
import queue
from struct import *
import logging
from timeit import default_timer as timer
  
# Diccionario de colas a manejar.
frame_q = {}
isort_q = {}
sensib_q = {}

borra_q = queue.Queue(maxsize=1500)
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename="multiserver2.log",
    level = logging.DEBUG,
    format = LOG_FORMAT,
    filemode = "w")
logger = logging.getLogger()
logger.info("Inicio logger")


class Detector:
    '''

    '''
    def __init__(self, isort):
        self.isort = isort
        self.deteccion = Drk() # inicializa darknet 
        self.started = False
        self.ahora= datetime.datetime.now()
        #self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            print('funcion start en class Detector')
            return None
        self.started = True
        self.thread = threading.Thread(name='DETECTOR',target=self.read, args=())
        self.thread.start()
        return self

    def read(self):
        print("paso x read hu hu")
        logger.debug("Detector.read:self.read_lock ")
        while self.started:
        #with self.read_lock:
            for i in list(frame_q):
                # if frame_q[i].empty() != True and sensib_q[i].empty() != True :
                #     frame = frame_q[i].get()
                #     threshold = sensib_q[i].get() 
                #     objetos=self.deteccion.dark(frame, threshold)
                #     isort_q[i].put(objetos)
                if frame_q[i].empty() != True:
                    frame = frame_q[i].get()
                    #largo = frame_q[i].qsize()
                    #print(largo)
                    if sensib_q[i].empty() != True :
                        threshold = sensib_q[i].get()
                    else:
                        threshold = 0.75
                    objetos=self.deteccion.dark(frame, threshold)
                    isort_q[i].put(objetos)

            if datetime.datetime.now()-self.ahora>timedelta(seconds=2):
                print(threading.current_thread(), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                for i in list(frame_q):
                    print(frame_q[i].qsize(),'frame_q',i)
                self.ahora = datetime.datetime.now()

            if borra_q.empty() != True:
                borraid = borra_q.get()
                self.clear(borraid)
                print(borraid,"borraid")
        print("read, sali de while")

    def clear(self, id_):
        isort_q.pop(id_)
        frame_q.pop(id_)
        sensib_q.pop(id_)
        #fr_num_q.pop(id_)
        logger.debug("Detector.clear id:%s",id_)

# inicializa el detector antes del main.
detec_demon=Detector([])

class Conexion(object):

    def __init__(self, connections, output, cola, sensib, id_):
        self.client_socket = connections
        self.cola = cola
        self.output = output
        self.id_ = id_
        self.comienzo = False
        self.sensib = sensib
        #self.fr_num = fr_num

    def start(self):
        if self.comienzo:
            print('Comenzo')
            return None
        self.comienzo = True
        self.thread = threading.Thread(target=self.threaded,name=self.id_,daemon=False)
        self.thread.start()

    def threaded(self):
        print('inicia hilo',get_ident() )

        self.client_socket.setblocking(True)

        while self.comienzo:

            print("se")
            data = b''
            while 1:
                inicio2=timer()
                try:
                    r = self.client_socket.recv(90456)
                    if len(r) == 0:
                        borra_q.put(self.id_)
                        print("no datos desde %2d, bye!"%(self.id_))
                        self.comienzo=False
                        break
                    a = r.find(b'END!')
                    if a != -1:
                        data += r[:a]
                        sen = r[-4:]
                        o,p =unpack('hh',sen)
                        if o==2019:
                            self.sensib.queue.clear()
                            self.sensib.put(p/100)
                            break
                        else:
                            self.sensib.queue.clear()
                            self.sensib.put(0.75)
                            break
                    data += r
                except Exception as e:
                    print(e)
                    break
                    continue
                #print("tiempo de Recepcion %.3f" %(timer()-inicio2))
            #print(data,"data")
            nparr = np.fromstring(data, np.uint8)
            if nparr.shape[0]>0:
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if type(frame) is type(None):
                pass
            else:
                try:
                    #frame = cv2.resize(frame,(416,416))
                    if self.cola.empty()!= True: self.cola.queue.clear()
                    self.cola.put(frame)
                    #if self.fr_num.empty()!= True: self.fr_num.queue.clear()
                    #self.fr_num.put(frn)

                except Exception as e:
                    print(e)
                    continue

            isort = []
            try:
                if self.output.empty() != True: 
                    isort=self.output.get()
                    #print(type(isort))
                    #if isort != []:
                        #bfr=np.zeros((1,isort.shape[1]),dtype=int)
                        #bfr=np.int32(bfr)
                        #bfr[0][0]=self.fr_num.get()
                        #isort=np.append(isort,bfr,axis=0)
                        #print(isort)
                    self.client_socket.send(isort)
                    self.client_socket.send(b"FIN!")
                    #print(isort)
            except socket.error:
                print("isort no enviado")
                pass

        print(threading.enumerate())
        #borra_q.put(self.id_)
        print("cerrando conexion")
        # connection closed 
        self.client_socket.close() 
        
        print(isort_q.keys(), frame_q.keys(), sensib_q.keys(),"borra",borra_q.qsize() )
        exit()

    def stop(self):
        self.comienzo = False
        self.thread.join()
  
def Main(): 
    infiere = []
    host = "" 
    port = 12347
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #server_socket.setblocking(0)
    server_socket.bind((host, port))
    #server_socket.settimeout(1)
    print("El socket esta atado al puerto", port)
    server_socket.listen(5)
    print("El socket esta escuchando...") 
    detec_demon.start()

    while True: 
        # Establece coneccion con el cliente
        client_socket, addr = server_socket.accept() 
        client_socket.sendall(b'Conexion ')
        # adquiere el look con el cliente 
        frame_q[addr[1]]=queue.LifoQueue(maxsize=1500)
        isort_q[addr[1]]=queue.LifoQueue(maxsize=1500)
        sensib_q[addr[1]] = queue.LifoQueue(maxsize=2000)
        print('Conectado a :', addr[0], ':', addr[1],len(frame_q),"frame_q",len(isort_q),"isort_q") 
        client = Conexion(client_socket,
            isort_q[addr[1]],
            frame_q[addr[1]],
            sensib_q[addr[1]],
            addr[1] )
        client.start()

    server_socket.close()
    detec_demon.stop() 
  
if __name__ == '__main__': 
    Main() 