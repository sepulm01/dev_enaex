#!/usr/bin/env python

'''
Recdata SPA.
Nov - 2019
Programa de tracking de objetos StreetFlow.
Version de prueba para la Enaex para camaras fijas.

'''
from __future__ import print_function
from __future__ import division
import numpy as np
import cv2 as cv
from timeit import default_timer as timer
from common import anorm2, draw_str
import scipy
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.path as mpltPath
import datetime
from datetime import timedelta
import time
import math
import sys
from sklearn.cluster import KMeans
import pandas as pd
import threading
#import shapely.geometry as geometry
#from descartes import PolygonPatch
from sql.basedatos import basedatos
from sql.bdremota import bdremota
import csv
#import darknet
#from drk import Drk
import os
import queue
from centroidtracker import CentroidTracker
import socket , errno
import random
#from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment
from struct import *
import json
import ast
import logging

LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename="dev_uoct_drk_bgsub_v1.5.log",
    level = logging.DEBUG,
    format = LOG_FORMAT,
    filemode = "w")
logger = logging.getLogger()
logger.info("Inicio logger")

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 19, #19 o 95
                       qualityLevel = 0.3,
                       minDistance = 4, #7
                       blockSize = 7 )

gamma = 0.20
gamma_max = 200
input_q = queue.Queue(maxsize=100) # cola para el retraso
drk_q = queue.LifoQueue(maxsize=150) # cola para almacenar el frame
sensib_q = queue.Queue(maxsize=200) # cola para la sensibilidad dia / noche de la prediccion
isort_q  = queue.LifoQueue(maxsize=250)

######################### Darknet ###################################################
# netMain = None
# metaMain = None
# configPath = "./cfg/yolov3.cfg"
# #configPath = "./cfg/yolov3-8clases.cfg"
# weightPath = "./data/yolov3.weights" 
# #weightPath = "./data/yolov3-uoct_final.weights"
# #weightPath = "./data/yolov3-uoct_last.weights" 
# #weightPath = "./data/yolov3-8clases_16000.weights" 
# metaPath = "./cfg/coco.data"
# if netMain is None:
#     netMain = darknet.load_net_custom(configPath.encode(
#         "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
# if metaMain is None:
#     metaMain = darknet.load_meta(metaPath.encode("ascii"))
# darknet_image = darknet.make_image(darknet.network_width(netMain),
#                                 darknet.network_height(netMain),3)

def dark(frame, thresh=0.75):
    ''' Funcion que llama al modelo de deteccion, recive una imagen y el nivel de confianza y devuelve una matriz.
    '''
    antes=timer()
    clases = [0] #clases a seguir

    def convertBack(x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def Boxes(detections):
        det = []
        for detection in detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
            prob = round(float(detection[1]),2)*100
            if int(detection[0].decode()) in clases: 
                det.append([ xmin, ymin, xmax, ymax ,detection[0].decode(), prob])
        return det

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_resized = cv.resize(frame_rgb,
                               (darknet.network_width(netMain),
                                darknet.network_height(netMain)),
                               interpolation=cv.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
    detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh)
    #print(frame.shape, frame_resized.shape, "frame.shape, frame_resized" )
    det = Boxes(detections)
    isort=np.int32(det)
    #print("drk detect",timer()-antes)
    return isort
######################### Fin Darknet ###################################################

class Detector:
    ''' Demonio que corre automaticamente reciviendo las imagenes (frames) del flujo de video principal, mediante la colas drk_q
    y devuelve una objeto numpy (matriz). La clase llama a la función dark, a quien le entrega el frame y la sensibilidad 
    requerida para esa prediccion.
    '''
    def __init__(self):
        self.isort = []
        self.started = False

        self.read_lock = threading.Lock()
        print('deteccion ha empezado')

    def start(self):
        if self.started:
            print('deteccion ha empezado')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, name='Detector-hilo',args=(),daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            isort = self.read()
            with self.read_lock:
                self.isort = isort
                #q.task_done()

    def read(self):
        with self.read_lock:
            frame= drk_q.get()
            sensib = sensib_q.get()
            self.isort = dark(frame,sensib)
            drk_q.task_done()
            drk_q.queue.clear()
            sensib_q.task_done()
            sensib_q.queue.clear()
        return self.isort

    def stop(self):
        self.started = False
        print("stop hilo prejoin")
        self.thread.join(timeout=2)
        print("stop hilo postjoin")

class App:
    def __init__(self):
        self.track_len = 10 #10
        self.detect_interval =  10#10
        self.tracks = []
        self.frame_idx = 0
        self.anchos_mx = np.zeros((4,4),dtype=int)

    def run(self):
        ######SOCKETS###############
        def signal_handler(signal=None, frame=None):
            exit()

        def envia(client_socket):
            nombre_v = str(random.random())
            sigue=threading.Event()
            hilo20 = threading.Thread(name='recibe', 
                target=recibe,
                args=(client_socket,sigue),
                daemon=True)
            hilo20.start()
            sigue.set()
            sensib=75
            while 1:
                ret = sckt_ret
                #try:
                #if drk_q.empty() != True:
                frame = drk_q.get()
                drk_q.task_done()
                drk_q.queue.clear()
                data = cv.imencode('.jpg', frame)[1].tostring()
                    # if sensib_q.empty() != True:
                sensib=sensib_q.get()
                sensib_q.task_done()
                sensib_q.queue.clear()
                #except:
                #    print("fallo en colas get")
                 
                if ret:
                  
                    try:
                        client_socket.send(data)
                        #frm_num=int(cap.get(1))
                        zen = pack('hh',2019, sensib )
                        client_socket.send(b"END!"+zen) # envia parametro para parar loop en el server
                    except socket.error:
                        print("error")
                        break
                else:
                    print("ret 0")
                    sigue.clear()
                    hilo20.join(timeout=1)
                    break
     
        def recibe(client_socket, sigue):
            while sigue:
                data = b''
                while 1:
                    inicio2=timer()
                    try:
                        r = client_socket.recv(90456)
                        if len(r) == 0:
                            print("no datos desde el servidor de inferencias, bye!")
                            logger.warning("Sokcet: comunicacion interrumpida con serv inferencias ")
                            evento.set()
                            exit()
                        a = r.find(b'FIN!')
                        if a != -1:
                            data += r[:a]
                            break
                        data += r
                    except Exception as e:
                        print(e)
                        continue
                    #print("tiempo de Recepción %.3f" %(timer()-inicio2))
                isort = []
                isort = np.fromstring(data, np.int32)
                isort= np.reshape(isort, (-1, 6))
                if isort.sum() > 1:
                    isort=isort.astype('int')
                    #print(isort)
                    isort_q.put(isort)
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
        ##### FIN SOCKETS###############

        def export_db():
            ''' exporta bases de datos '''
            print("exporta tablas de base de datos")
            basedatos.export_csv(conn,'base.csv')
            basedatos.export_cont_csv(conn,'base_cont.csv')
            print("termine \n", datetime.datetime.now())

        #####################Backgroud detector ######################################
        def filter_mask( img, a=None):
            '''
                Filtro para tratar el background substration
            '''
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)) #(2, 2)
            # llena los agujeros pequeños
            closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
            # Remueve el ruido
            opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
            # Dilata los blobs adjacentes
            dilation = cv.dilate(opening, kernel, iterations=1  )
            return dilation

        def detect_vehicles(fg_mask):
            min_contour_width=10
            min_contour_height=50
            max_contour_width=90
            max_contour_height=150
            matches = []
            centroide = []
            # Encuentra los contornos externos
            contours, hierarchy = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)

            for (i, contour) in enumerate(contours):
                #cv.polylines(frame_gray,[contour],True,(0, 0, 10), 1) # background sub
                cv.drawContours(frame_gray, [contour], 0, (0,0,0), 1)
                M = cv.moments(contour)
                # calculate x,y coordinate of center
                if M["m00"] <=0:
                    M["m00"] = 1
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                if cX <= dim_x/2 and cY <= dim_y/2: n=0 # verdadero q1
                elif cX <= dim_x/2 and cY >= dim_y/2: n=1 # verdadero q2
                elif cX >= dim_x/2 and cY >= dim_y/2: n=2 # verdadero q3
                elif cX >= dim_x/2 and cY <= dim_y/2: n=3 

                min_contour_width=self.anchos_mx[n][2]
                min_contour_height=self.anchos_mx[n][3]
                max_contour_width=self.anchos_mx[n][0]
                max_contour_height=self.anchos_mx[n][1]
                
                (x, y, w, h) = cv.boundingRect(contour)

                contour_valid = (w >= min_contour_width) and (w <= max_contour_width) and (
                    h <= max_contour_height) and (h >= min_contour_height) and (h/w > 2.7)

                if contour_valid:
                    matches.append((x, y, x+w, y+h, 79, 0))
                    centroide.append((cX, cY))
            centroide=np.int32(centroide)
            matches=np.int32(matches)
            return matches, centroide
        
        def get_centroid(x, y, w, h):
            x1 = int(w / 2)
            y1 = int(h / 2)
            cx = x + x1
            cy = y + y1
            return (cx, cy)

        #########################Fin Backgroud detector ######################################

        ########################## Correccion GAMMA ##########################################
        def gammaCorrection(frame):
            '''
            Función de corrección de gamma aplicado a la imagen
            '''
            img_original=frame
            lookUpTable = np.empty((1,256), np.uint8)
            for i in range(256):
                lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            res = cv.LUT(img_original, lookUpTable)
            return res

        def on_gamma_correction_trackbar(valor):
            global gamma
            gamma = valor / 100
            #print(gamma,'gamma')

        ##########################FIN Correccion GAMMA ##########################################

        def velocidad(id_):
            ptsoutX, phiXX, magXXX, claseXX, frm_num=basedatos.cons_obj_dir(conn,int(u))
            dist=0
            vel=0
            if len(ptsoutX)>0 and len(frm_num)>0:
                dist=int(euclidean_distances(ptsoutX[0],ptsoutX[len(ptsoutX)-1])[0][0])
                tiempo_frm=frm_num[len(frm_num)-1]-frm_num[0]
                if tiempo_frm>0:
                    vel= dist / tiempo_frm
            return vel
        
        def update_bd_remota(row,id_flujo):
            print('Hilo:',threading.current_thread().getName(),'con identificador:',threading.current_thread().ident)
            #bdremota.ins_5min(row) # a Mysql
            bdremot.ins_postgres2(row) # a Postgres
            bdremot.get_ini(2,id_flujo) # a Postgres, infoma que el flujo de video esta activo.

        def update_bd_alarma(row,id_flujo, foto_file):
            print('Hilo:',threading.current_thread().getName(),'con identificador:',threading.current_thread().ident)
            bdremot.alarma_post(row, foto_file) # a Postgres
            bdremot.get_ini(2,id_flujo) # a Postgres, infoma que el flujo de video esta activo.

        def comprime_vid(video_nombre):
            print('Hilo:',threading.current_thread().getName(),'con identificador:',threading.current_thread().ident)
            os.system("ffmpeg -i {0} {0}.mp4".format(video_nombre))
            os.system("rm {0}".format(video_nombre))

        def nuevos_ptos(i_sort,frame_gray,frame, tracks, id0, indice,bg_detect ):
            '''
            Asocia las detecciones con los id0 ya existentes o genera nuevos id0s.
            '''
            bg_=bg_detect
            #i_sort = capo.isort

            if bg_.shape[0]>0:
                for bg in bg_:
                    agrega = 0
                    for so in i_sort:
                        int_iou = bb_iou((so[0],so[1],so[2],so[3]),
                            (bg[0],bg[1],bg[2],bg[3]))
                        if int_iou>0:
                            agrega +=1
                    if agrega == 0 and len(i_sort)>0:
                        i_sort=np.append(i_sort,[bg], axis=0 )
            if len(i_sort)<1:
                i_sort = bg_
   
            lista_maches = np.zeros((0,4))

            def anchos(x1,y1,x2,y2,anchos_mx):
                if x2 <= dim_x/2 and y2 <= dim_y/2: n=0 # verdadero q1
                elif x2 <= dim_x/2 and y2 >= dim_y/2: n=1 # verdadero q2
                elif x2 >= dim_x/2 and y2 >= dim_y/2: n=2 # verdadero q3
                elif x2 >= dim_x/2 and y2 <= dim_y/2: n=3 # verdadero q4
                ancho_max,alto_max,ancho_min,alto_min=anchos_mx[n][0],anchos_mx[n][1],anchos_mx[n][2],anchos_mx[n][3]
                if (x2-x1)>ancho_max: anchos_mx[n][0]=(x2-x1)
                elif (y2-y1)>alto_max: anchos_mx[n][1]=(y2-y1)
                elif (y2-y1)<alto_min: anchos_mx[n][3]=(y2-y1)
                elif (y2-y1)<ancho_min:anchos_mx[n][2]=(x2-x1)
                return anchos_mx

            for i in range(i_sort.shape[0]):
                x1,y1,x2,y2, clase, score=i_sort[i][0],i_sort[i][1],i_sort[i][2],i_sort[i][3],i_sort[i][4],i_sort[i][5]
                ancho = x2-x1
                alto = y2-y1
                if ancho*alto > 200 and clase in clases: #TODO probar en bici

                    ############ extrae objetos no identificados para re entrenamiento
                    if clase == 79 and re_entrenar: #Originalmente 400
                        archivo_ = outputdir+"revisar/"+str(self.frame_idx)+".jpg"
                        cv.imwrite(archivo_,frame)
                        file1 = open(archivo_+".txt","w") 
                        yolo_fmt ="%2d %2f %2f %2f %2f" % (0,x1 / frame.shape[0], y1 / frame.shape[1], ancho/frame.shape[0], alto / frame.shape[1])
                        file1.writelines(str(yolo_fmt))
                        file1.close()
                    self.anchos_mx=anchos(x1,y1,x2,y2,self.anchos_mx)

                    cv.rectangle(vis,(x1,y1),(x2,y2),(0,0,255),1)# rectangulo al momento de la deteccin
                    #cv.imwrite("output/"+str(self.frame_idx)+".1.jpg",vis)
                    #etiqueta="x %2d y %2d" % (x1,y1)
                    #cv.putText(vis, etiqueta, (x1-10, y1-5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
                    mapa=mpltPath.Path([(x1,y1),(x1,y2),(x2,y2),(x2,y1),(x1,y1)])
                    iou_ptos__isort = {i:(x1,y1,x2,y2)}
                    iou_ptos_=[]
                    cx0,cy0 = x1+(ancho/2),y1+(alto/2)
                    identificador = []
                    puntaje = 0
                    
                    ## recorre indice de obj activos en seguimiento para asignar puntaje de mayo IOU con el objeto (i sort) detectado.
                    ## tambien borra los puntos del obj del indice con mayor tiempo sin detectar.
                    #print("entrando al loop frame ", self.frame_idx, np.unique(indice))
                    for u in np.unique(indice):
                        #print("dentro del loop frame ", self.frame_idx)
                        ptos_ = []
                        ptos_ = obtiene_ptos(int(u), self.tracks, indice)
                        contiene = mapa.contains_points(ptos_) # Evalua si el punto est en el mapa
                        contiene = np.ones(contiene.shape)*contiene
                        porcentaje = contiene.sum()/contiene.shape[0] + 1
                        xboun,yboun,wboun,hboun = cv.boundingRect(np.array(ptos_))
                        cxb,cyb = xboun+(wboun/2),yboun+(hboun/2)
                        if len(dimen[int(u)])>0:
                            ancho_, largo_ = dimen[int(u)]
                            anc = ancho_/2
                            lar = largo_/2
                            iou_ptos_=(int(cxb-anc),int(cyb-lar),
                                int(cxb+anc),int(cyb+lar))
                            #cv.rectangle(vis,(int(cxb-anc),int(cyb-lar)),(int(cxb+anc),int(cyb+lar)),(0,250,0),1) 
                        else:    
                            iou_ptos_=(xboun,yboun,xboun+wboun,yboun+hboun)
                            #cv.rectangle(vis,(xboun,yboun),(xboun+wboun,yboun+hboun),(0,250,0),1) 
                        iou = bb_iou(iou_ptos__isort[i],iou_ptos_)
                        porcentaje = porcentaje * iou
                        if porcentaje > puntaje:
                            identificador = int(u)
                            puntaje = porcentaje #deja el puntaje mas alto
                    
                    if identificador==[]:
                        if len(hermano)>0:
                            mayor=0
                            objectID_ = 0
                            for (objectID, ce) in objects.items():
                                contiene = mapa.contains_points([[ce[0], ce[1]],[ce[0]+10,ce[1]+10],
                                    [ce[0]-10, ce[1]-10],[ce[0]+10, ce[1]-10],[ce[0]-10, ce[1]+10]])
                                if (np.ones((5))*contiene).sum() > mayor:
                                    mayor = (np.ones((5))*contiene).sum()
                                    objectID_ = objectID                            
                            if objectID_ > 0:
                                if objectID_ in hermano.keys():
                                    
                                    ident_tmp=hermano[objectID_]
                                    puntaje =1.1
                                else:
                                    ident_tmp=0
                            else:
                                ident_tmp=0
                        else:        
                            ident_tmp=0
                    else:
                        ident_tmp=identificador
                    lista_maches=np.append(lista_maches,[[i,ident_tmp,puntaje,clase]],axis=0)
            a = lista_maches
            nueva = np.zeros((0,4))
            nueva2 = np.zeros((0,4))
            z=np.unique(a[:,1], return_counts=True)
            for i in range(0,len(z[0][z[1]>1])):
                u=z[0][z[1]>1][i]
                b=a[a[:,1]==u,...]
                v=b[:,2].max()
                d=(b[b[:,2]==v,...])
                c=b[b[:,2]!=v,...]
                c[:,2]=0
                d = np.append(d,c,axis=0)
                nueva = np.append(nueva,d,axis=0)
            for i in range(0,len(z[0][z[1]==1])):
                u=z[0][z[1]==1][i]
                b=a[a[:,1]==u,...]
                nueva2 = np.append(nueva2,b,axis=0)
            final = np.append(nueva,nueva2,axis=0)
            final=np.sort(final.view('i8,i8,i8,i8'), order=['f0'], axis=0).view(np.float)
  
            for i in range(final.shape[0]):
                puntaje = final[i][2] 
                if int(final[i][1]) is not 0 and puntaje > ptje_min:
                    ciclos[int(final[i][1])]= datetime.datetime.now()
                    self.tracks, indice = borra_ptos(int(final[i][1]), self.tracks, indice) #refresca los puntos a seguir
                mask = np.zeros(frame_gray.shape, dtype = "uint8")
                x1,y1,x2,y2= i_sort[i][0],i_sort[i][1],i_sort[i][2],i_sort[i][3]
                cv.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)             
                p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)

                if p is not None:
                    id_=np.zeros((p.shape[0], 1))
                    if puntaje > ptje_min:
                        id_.fill(int(final[i][1]))
                        dclase[int(final[i][1])]=int(final[i][3]) 
                        dimen[int(final[i][1])]=((x2-x1),(y2-y1))
                        mapa=mpltPath.Path([(x1,y1),(x1,y2),(x2,y2),(x2,y1),(x1,y1)])
                        ############link con bgsub
                        if b_ground_sub:
                            mayor=0
                            objectID_ = 0
                            for (objectID, ce) in objects.items():
                                contiene = mapa.contains_points([[ce[0], ce[1]],[ce[0]+10,ce[1]+10],
                                    [ce[0]-10, ce[1]-10],[ce[0]+10, ce[1]-10],[ce[0]-10, ce[1]+10]])
                                if (np.ones((5))*contiene).sum() > mayor:
                                    mayor = (np.ones((5))*contiene).sum()
                                    objectID_ = objectID                            
                            if objectID_ > 0:
                                hermano[objectID_]=int(final[i][1])
                        ############ fin link con bgsub
                    else:
                        id0 += 1
                        id_.fill(id0)
                        dclase[id0]=int(final[i][3])  
                        ciclos[id0]= datetime.datetime.now()
                        ############link con bgsub
                        if b_ground_sub:
                            mayor=0
                            objectID_ = 0
                            for (objectID, ce) in objects.items():
                                contiene = mapa.contains_points([[ce[0], ce[1]],[ce[0]+10,ce[1]+10],
                                    [ce[0]-10, ce[1]-10],[ce[0]+10, ce[1]-10],[ce[0]-10, ce[1]+10]])
                                if (np.ones((5))*contiene).sum() > mayor:
                                    mayor = (np.ones((5))*contiene).sum()
                                    objectID_ = objectID                            
                            if objectID_ > 0:
                                hermano[objectID_]=id0
                        ############ fin link con bgsub
                        dimen[id0]= ((x2-x1),(y2-y1))
                    indice = np.append(indice,id_, axis=0)
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
            #self.tracks, indice=buscadup(self.tracks, indice)
            return self.tracks, id0, indice, dclase
        
        def pol2cart(rho, phi):
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return(x, y)

        def obtiene_ptos(id0, tracks = [], indice= []):
            ptos = []
            for pto, indx in zip(tracks, indice.reshape(-1,1)):
                if indx[0]==id0:
                    x,y = int(pto[len(pto)-1][0]),int(pto[len(pto)-1][1])
                    ptos.append((x,y))
            return ptos
        
        def index_todo( tracks = [], indice= [],calle=0):
            '''
            Revisa la coherencia de los puntos del optical flow (tracks) con sus indices(ids).
            elimina los puntos que quedan atrapados o rezagados.
            Mantiene la etiqueta que se dibuja en los objetos cuadro a cuadro junto a la flecha de la dirección
            y actualiza la base sqlite.
            '''
            aborrar =[]
            registro = []
            
            trazo = 0
            for pto, indx in zip(tracks, indice.reshape(-1,1)):
                x2,y2 = int(pto[len(pto)-1][0]),int(pto[len(pto)-1][1])
                x1,y1 = int(pto[0][0]),int(pto[0][1])
                #phi=math.atan2((x2-x1),(y2-y1))
                mag= math.hypot((x2-x1),(y2-y1))
                #pa, pb = (x1,y1),(x2,y2)
                #registro.append([trazo,indx[0],phi, mag, x1,y1,x2,y2])  # registro: index, id0, phi, mag, x1,y1,x2,y2
                registro.append([trazo,indx[0],0, mag, x1,y1,x2,y2])
                trazo +=1
            registro=np.array(registro) 
            
            for u in np.unique(indice):
                data = registro[registro[:,1]==u,...]
                #pi=data[:,2].mean()
                mag=data[:,3].mean()
                x1=data[:,4].mean()
                y1=data[:,5].mean()
                x2=data[:,6].mean()
                y2=data[:,7].mean()
                x1,y1,x2,y2 = int(x1), int(y1),int(x2), int(y2)
                mag_std=data[:,3].std()
                ptos = data[:,[6,7]]
                ptos = np.int32(ptos)
                #print(ptos, ptos.shape, "ptos dentro de sub")
                x_,y_,w,h = cv.boundingRect(ptos) # reodea con un rectangulo una nuve de puntos
                #basedatos.inserta(conn, int(u), str(dclase[u]),x_,y_,x_+w,y_+h,datetime.datetime.now(), round(pi,2), round(mag,2), calle,cap.get(1) )
                basedatos.inserta(conn, int(u), str(dclase[u]),x_,y_,x_+w,y_+h,datetime.datetime.now(), 0, round(mag,2), calle,cap.get(1) )
                etiqueta = '%2d %2d' % ( dclase[u], int(u) )
                #etiqueta = '%s' % ( "Persona1")
                #cv.rectangle(vis, (x2-5,y2-10),(x2+40,y2+2),(255,255,255),-1)
                cv.putText(vis, etiqueta, (x2, y2), cv.FONT_HERSHEY_DUPLEX, 0.4, (255,0,0), 1)
                cv.arrowedLine(vis,(x1,y1),(x2,y2),(25,250,250),2)
                #cv.polylines(vis,[ptsoutX],False,(0, 0, 10), 1) # dibuja el trayecto
                if mag_std > 7: #14 ,7
                    aborrar=data[data[:,3]<2,...][:,0].tolist()
            if aborrar !=[]:
                for i in reversed(aborrar):
                    tracks.pop(int(i))
                    indice = np.delete(indice, [int(i)],axis=0)
                aborrar=[]
            return tracks, indice
      

        def borra_ptos(idbr, tracks = [], indice= []):
            new_tracks = []
            for pto, indx in zip(tracks, indice.reshape(-1,1)):
                if indx[0]!=idbr:
                    x,y = int(pto[len(pto)-1][0]),int(pto[len(pto)-1][1])
                    new_tracks.append([(x, y)])
            indic_ = indice[indice!=idbr]
            indice = indic_.reshape(-1,1)
            return new_tracks, indice

        def buscadup(tracks = [], indice= []):
            #print(iou_matrix,"iou_matrix", np.unique(indice))
            borrado = []
            a = []
            for d in np.unique(indice):
                ptos_ = obtiene_ptos(int(d), tracks, indice)
                x,y,w,h = cv.boundingRect(np.array(ptos_))
                det=(x,y,x+w,y+h)
                for t in np.unique(indice):
                    ptos_ = obtiene_ptos(int(t), self.tracks, indice)
                    x,y,w,h = cv.boundingRect(np.array(ptos_))
                    trk=(x,y,x+w,y+h)
                    iou=bb_iou(det,trk)
                    if iou > 0.5 and int(d)!=int(t):
                        print(int(d),int(t), bb_iou(det,trk),"iou")
                        a.append([int(d),int(t)])
            for i in range(0,len(a)):
                a[i].sort()
            for i in range(0,len(a)):
                for ii in range(0,len(a)):
                    if a[i]==a[ii]:
                        print(max(a[i]))
                        self.tracks, indice=borra_ptos(max(a[i]), tracks, indice)
            return self.tracks, indice
        
        def bb_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            iou = interArea / float(boxAArea + boxBArea - interArea)
            return iou

        def totuple(a):
            try:
                return tuple(totuple(i) for i in a)
            except TypeError:
                return a

        def mapa_areas():
            '''Carga areas desde el archivo areas.csv, entrega un objeto tipo mpltPath
            y una lista con N np array (1x2) con los puntos x,y del perimetro del area.
            Donde el N es el id del area o detector.
            '''
            areas = []
            lista = []
            your_list = []
            with open('areas.csv', 'r') as f:
                reader = csv.reader(f,delimiter=',')
                your_list = list(reader)
            for z in range(0,len(your_list)):
                par = []
                for i in range(0,len(your_list[z]),2):
                    par.append([your_list[z][i],your_list[z][i+1]])
                lista.append(par)
            for w in range(0,len(lista)):
                asd = np.array(lista[w], np.int32)
                areas.append(asd)
            #hace mapa de las areas
            mapa = []
            dibujo = []
            for z in range(0,len(areas)): 
                mapa.append(mpltPath.Path(areas[z]))
                dibujo.append(areas[z])
            print(len(mapa), len(dibujo),"mapa, dibujo")
            return mapa, dibujo

        def mapa_areas_bd(det_):
            '''Carga areas desde base datos, entrega un objeto tipo mpltPath
            y una lista con N np array (1x2) con los puntos x,y del perimetro del area.
            Donde el N es el id del area o detector.
            '''
            areas = []
            lista=[]
            for z in det_:
                par = []
                for i in range(0,len(det_[z]),2):
                    par.append([det_[z][i],det_[z][i+1]])
                lista.append(par)
            for w in range(0,len(lista)):
                asd = np.array(lista[w], np.int32)
                areas.append(asd)
            mapa = []
            dibujo = []
            for z in range(0,len(areas)): 
                mapa.append(mpltPath.Path(areas[z]))
                dibujo.append(areas[z])
            return mapa, dibujo

        #################### Parametros a inicializar  #######################################
        ######################################################################################
        parametros = []
        bdremot=bdremota() # inicializa coneccion a base de datos remota.
        while len(parametros) == 0:
            time.sleep(2)
            print("En espera de parametros de inicializacion.")
            parametros, id_flujo = bdremot.get_ini(0,0)

        bdremot.get_ini(1,id_flujo) # update el estado del flujo de disponible(0), a tomado(1)

        #fuente = 'rtsp://172.26.1.143:654/00000001-0000-babe-0000-accc8e0039da/live' #flujo video UOCT
        #fuente = 'rtsp://172.26.1.143:654/00000001-0000-babe-0000-accc8e221565/live' #locurro

        fuente = parametros['fuente']
        clases = parametros['clases'] # clases a identificar
        sensib_min = parametros['sensib_min']
        sensib_max = parametros['sensib_max']
        sensib_lim = parametros['sensib_lim']
        bgshow = parametros['bgshow']
        gammashow = parametros['gammashow']
        b_ground_sub = parametros['b_ground_sub']
        area_mask = parametros['area_mask']
        contadores = parametros['contadores']
        buf_conges = parametros['buf_conges']
        l = []
        for q in buf_conges:
            c=int(q[1]),int(q[3])
            l.append(c)
        buf_conges = l
        #print(type(buf_conges[0]),buf_conges)
        host = parametros['gpu_server'] # host sockets
        port = parametros['gpu_port'] # port sockets
        retardo = parametros['retardo'] #4 Var retardo del flujo de video para que la GPU tenga tiempo de procesar la imagen actual
        detec = parametros['detec'] # areas de deteccion
        detec_tipo = parametros['detec_tipo'] # tipo de detector V, C, P
        junction = parametros['junction']
        detec_arco = parametros['detec_arco']
        detec_sen = parametros['detec_sen'] # direccion combinaciones de O,P,N,S
        detec_vir = parametros['detec_vir'] # Viraje o no

        fuente = sys.argv[1]
        print("Fuente",fuente)
        cap = cv.VideoCapture(fuente)
        _ret, frame = cap.read()
        drk_q.put(frame)
        #print("frames totales",cap.get(cv.CAP_PROP_FRAME_COUNT))
        indice = np.array([])
        indice = indice.reshape(-1,1)
        ciclos = {"id":datetime.datetime.now()} #Dic mantiene los puntos trackeados x un tiempo hasta su eliminación 
        crono = datetime.datetime.now()
        id_cont = 0
        dclase = {"id":"clase"}
        p0 = [] # optical flow
        id0 = 0 # id correlativo de detección
        idprocesados = [] # buffer de id contabilizados
        ptje_min = 0.00001
        dim_y, dim_x = 416,416 # ajuste de imagen para matriz de detección        
        winbGround = 'background subtraction'
        bg_subtractor = cv.createBackgroundSubtractorMOG2(history=400, varThreshold = 20, detectShadows=True) #detecta mov de la camara
        outputdir = '/tmp/'+ junction +'/'+str(id_flujo)+'/'# directorio de salida
        try:  
            os.mkdir('/tmp/'+ junction)
            os.mkdir(outputdir)
        except OSError:  
            print ("Creation of the directory %s failed" % outputdir)
        else:  
            print ("Successfully created the directory %s " % outputdir)        
        ## set up registro video
        fourcc = cv.VideoWriter_fourcc(*'XVID') #XVID X264 
        crono_vid = datetime.datetime.now() # variable de tiempo para gatillar hilos
        fecha = datetime.datetime.today().strftime('%Y-%m-%d')
        ayer = fecha # Var para la rutina de mantención 
        hora = datetime.datetime.now().strftime("%H:%M:%S")
        vidfile=outputdir+str(fecha)+'_'+str(hora)+'_'+str(id0)+'.avi'
        vid_writer = cv.VideoWriter(vidfile,fourcc, 25, (dim_x,dim_y))
        graba_video = False # flag graba video
        ## fin set up registro video
        conn = basedatos.create_connection() #Inicia conector de coneccion con SLQLite 

        dimen = {} # Diccionario de altos y anchos para la asignación de puntos antiguos
        hermano={} # Diccionario de id de la paca de backgroud substraction para la asignación de puntos antiguos
        hist_mtx = np.zeros((2,3,3))
        hist_mtx[1].fill(9999999)
        self.anchos_mx[:,(2,3)]=dim_x,dim_y
        histograma = []
        luz_dia = 0 # inicializa la variable con valor noche       
        cv.namedWindow('Streetflow.cl',cv.WINDOW_AUTOSIZE)
        gamma_init = int(gamma * 100)
        cv.createTrackbar('Ajuste Gamma', 'Streetflow.cl', gamma_init, gamma_max, on_gamma_correction_trackbar)
        mapa, dibujo = mapa_areas_bd(detec)
        puntaje =np.zeros((len(dibujo))) #matriz de conteo
        con = np.zeros((1,len(dibujo)+1)) # Matriz de congestion
        con[0][0]=1 # Variable de congestion
        vect=np.zeros(con.shape) # Variable de congestion
        ct = CentroidTracker() # inicializa el track del los id hermanos de la capa de background subs
        alarma_bl = False
        crono_alarm = datetime.datetime.now()
        crono_alarm_s = datetime.datetime.now()
        alarma = pd.DataFrame(columns=["detector", "tiempo", "tipo"])
        re_entrenar = False # Flag para extraer imagenes para re entrenar en dir /output/revisar
        ###### SOCKETS ############################
        #host = '0.0.0.0'
        #port = 12347
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        noConectado = True
        tiempofuera = 0
        while noConectado:
            try:
                client_socket.connect((host, port))
                info = client_socket.recv(1024)
                print('Recibido', repr(info))
                info = repr(info)
                noConectado = False

            except socket.error as e:
                logger.warning("Sokcet: fallo conectar serv inferencias ")
                if e.errno == errno.ECONNREFUSED:
                    # Handle the exception...
                    print(e,"\nMientras trataba de conectar con servidor de inferencias.\nHost %s y puerto %s" % (host, port))
                    print("\nReintenando en 10s, reintento #",tiempofuera)
                else:
                    print(e)
                    raise
                time.sleep(10)
                tiempofuera +=1
                bdremot.get_ini(5,0) #update tabla flujos a estado 5 
                if tiempofuera > 3:
                    bdremot.get_ini(4,id_flujo) # Deja el flujo de video en estado 0, disponible.
                    exit()


        sckt_ret = threading.Event()
        sckt_ret.set()
     
        hilo10 = threading.Thread(name='envia', 
            target=envia,
            args=([client_socket]),
            daemon=True)
        hilo10.start()
        
        ######FIN SOCKETS###########################
        frame = cv.resize(frame,(int(dim_x),int(dim_y))) # disminuye la imagen del 1er frame.
        mask_fg = np.ones(frame.shape, dtype = "uint8") # background sub
        cv.rectangle(mask_fg,(0,0),(dim_y, dim_x),(255, 255, 255),-1) 
        for i in range(0,len(dibujo)):
                cv.fillPoly(mask_fg, [dibujo[i]], (1, 1, 1))
        intermitente = False # flag intermitencia en color de las areas.
        ######################################################################################
        ####################Loop de deteccion en video #######################################
        ######################################################################################
        evento = threading.Event()
        while(cap.isOpened()):

            inicio = timer()
            _ret, frame = cap.read()
            if _ret == False:
                print("termino flujo de video (ret==0). espera 5s")
                break #TODO sacar en produccion
                
                while _ret == False:
                    logger.warning("Loop video: _ret = False. ie No hay señal del flujo de video desde la fuente. ")
                    time.sleep(5)
                    bdremot.get_ini(3,id_flujo) # actualiza estado 3 en tabla de flujos de bd
                    cap = cv.VideoCapture(fuente)
                    _ret, frame = cap.read()
            frame = cv.resize(frame,(int(dim_x),int(dim_y))) # disminuye la imagen
            if area_mask: #Coloca una mascara al area de detección para limitar el n° de ids seguidor.
                frame_msk = cv.add(mask_fg,frame) 
            else:
                frame_msk = frame
            #if _ret:
            input_q.put(frame) # cola de buffer para el retardo
            drk_q.put(frame_msk) # alimenta la cola del detector
            if luz_dia==0:
                if sensib_q.qsize() == 0:
                    sensib_q.put(sensib_min)
            else:
                if sensib_q.qsize() == 0:
                    sensib_q.put(sensib_max)
            if self.frame_idx > retardo:
                frame = input_q.get()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Visiual en capa vis: textos info.
            vis = frame.copy()
            draw_str(vis, (10, 15), 'Streetflow.cl')
            fps = cap.get(5) # Calcula los Frame x senconds del input.
            if area_mask:
                draw_str(vis, (350, 40), 'Area')
            if gammashow:
                frame2bg=gammaCorrection(frame) # Gamma Correction
                draw_str(vis, (350, 12), 'Gamma')
            else:
                frame2bg=frame
            if alarma_bl:
                draw_str(vis, (340, 55), 'alarma_blq')
            if graba_video:
                draw_str(vis, (340, 67), 'Video Rec')
            if re_entrenar:
                draw_str(vis, (340, 80), 'Re-entren')
            draw_str(vis, (340, 95), junction)
            draw_str(vis, (340, 110), str(id_flujo))

            th = np.sum(np.sum(frame))/(frame.shape[0]*frame.shape[1]*frame.shape[2])
            histograma.append(th)
            draw_str(vis, (10, 45), 'fps: %d                   luz %2d' % (fps,th))
            #puntaje_str =  np.array2string(puntaje, formatter={'float_kind':lambda x: "%.1d" % x})
            puntaje_str =  np.array2string(con, formatter={'float_kind':lambda x: "%.1d" % x}) # con = es la matriz de tiempo de los obj
            draw_str(vis, (10,30), puntaje_str)
            draw_str(vis, (10,100), str(cap.get(1)))
            #draw_str(vis, (10,115), str(frm_num_proc))

            for i in range(0,len(dibujo)):
                    color = (0,255,0)
                    zona = con[:,i+1].tolist()
                    for zi in zona:
                        #if zi > 125 and i>1: color = (0,0,255) discriminador de áreas i > X
                        if zi > 10 : #150
                            color = (0,255,255)
                        if zi > 30: #200
                            color = (0,0,255)
                            #intermitente = True
                    if intermitente:
                        color = (0,0,255)
                        if self.frame_idx % 10 == 0:
                            color = (0,255,255)
                    cv.polylines(vis,[dibujo[i]],True, color,2)
                    cv.fillPoly(mask_fg, [dibujo[i]], (1, 1, 1))
                    #cv.polylines(mask_fg,[dibujo[i]],True,(255, 255, 255), 1) # background sub
                    cv.putText(vis, str(i), totuple((dibujo[i])[0:1][0]), cv.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0),2)
            ############background sub #######################################
            if b_ground_sub:
                draw_str(vis, (350, 27), 'BG Sub')
                dst = cv.add(mask_fg,frame2bg) #frame2bg
                fg_mask = bg_subtractor.apply(dst, None, 0.001) # back ground detector
                fg_mask = filter_mask(fg_mask) # back ground detector
                bg_detect,centroide = detect_vehicles(fg_mask)  # back ground detector
                rects = []
                for e in bg_detect:
                    rects.append(e[0:4].astype("int"))
                    #cv.rectangle(fg_mask, (x, y), (a, b) , (255, 255, 255), 1)
                objects= ct.update(rects,centroide)
                for (objectID, centroid) in objects.items():
                    text = "ID {}".format(objectID)
                    cv.putText(vis, text, (centroid[0] - 10, centroid[1] - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    #cv.circle(vis, (centroid[0], centroid[1]), 4, (2, 255, 255), -1)
            else:
                bg_detect = np.array([])
            if bgshow and b_ground_sub:
                cv.imshow(winbGround,fg_mask)
            ############fin background sub #######################################

            ################ Optical Flow ########################################
            if len(self.tracks) > 0:
                t03 = timer()
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                new_indice = []
                for tr, (x, y), good_flag, good_indice in zip(self.tracks, p1.reshape(-1, 2), good, indice):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len: # largo del trazo de mediciones de los tracks
                        del tr[0]
                    new_tracks.append(tr)
                    new_indice.append(good_indice)
                    #cv.circle(vis, (x, y), 2, (0, 255, 0), -1) #visializa los puntos a seguir
                self.tracks = new_tracks
                indice =  np.float32([ind[-1] for ind in new_indice]).reshape(-1, 1)
                #cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0)) #estela verde
                #draw_str(vis, (10, 30), 'Objetos identificados: %d' % id0)
                self.tracks, indice =index_todo( self.tracks, indice,0) # Actualiza BD y lleva el control de casi todo
            ################ Optical Flow ########################################

            if self.frame_idx % 900 == 0:
                sum = 0
                for num in histograma:
                    sum = sum +num
                promedio  = sum / len(histograma)
                #print(promedio,"promedio")
                if promedio > sensib_lim :
                    if luz_dia==0:
                        bg_subtractor = cv.bgsegm.createBackgroundSubtractorGMG()
                        luz_dia = 1
                        print("Cambio a luz dia", datetime.datetime.now())
                else:
                    if luz_dia==1:
                        bg_subtractor = cv.createBackgroundSubtractorMOG2(history=200, varThreshold = 20, detectShadows=True)
                        luz_dia=0
                        print("Cambio a luz noche", datetime.datetime.now())
                histograma = []

            if self.frame_idx % 5 == 0:    
                #Rutina que cuenta el paso de los vehiculos por las areas 
                #print(len(np.unique(indice).tolist()))  
                for obj in np.unique(indice).tolist():
                    if obj not in idprocesados:
                        ptsout, clase= basedatos.consulta_obj_clase(conn,obj)
                        ptsout = ptsout.reshape(-1,2)
                        obj_dist = 0
                        if ptsout.shape[0] > 0:
                            inicio_ = ptsout[0]
                            fin_ = ptsout[ptsout.shape[0]-1]
                            obj_dist= euclidean_distances([inicio_], [fin_])[0][0]
                        if obj_dist > 30:
                            #for e in range(0,len(mapa)):
                            for e in contadores:
                                if any(mapa[e].contains_points(ptsout)):
                                    idprocesados.append(obj)
                                    puntaje[e]+=1
                                    if sys.argv[2] is not None: #TODO cambiar en produccion y dejar id_flujo fijo.
                                        camara = sys.argv[2]
                                    else:
                                        camara = id_flujo
                                    detector = e
                                    phi=math.atan2(([fin_][0][0]- [inicio_][0][0]),([fin_][0][1]-[inicio_][0][1]))
                                    sentido=np.digitize(phi,np.linspace(-1*np.pi,np.pi, 9))

                                    basedatos.inserta_contador2(conn,
                                        camara,detector,obj,clase,fecha,hora,1,int(sentido),
                                        detec_tipo[str(e)] ,junction, detec_arco[str(e)],
                                        detec_sen[str(e)], detec_vir[str(e)])

                                    #print(puntaje,obj,clase,sentido)
                                    if datetime.datetime.now()-crono>timedelta(seconds=60*3): # 300 seg = 5 min
                                        row, id_cont=basedatos.consulta_contador2(conn,id_cont)
                                        hilo1 = threading.Thread(name='update_bd_remota', 
                                            target=update_bd_remota,
                                            args=([row, id_flujo]),
                                            daemon=False)
                                        hilo1.start()
                                        crono = datetime.datetime.now()
                                        print(id_cont,'id_cont update base remota')

            if self.frame_idx % self.detect_interval == 0:
                if evento.isSet(): # Revisa si hay que salir del pipeline pq se ha perdido coneccion
                    break          # con el servidor de inferencias (GPU)
                fecha = datetime.datetime.today().strftime('%Y-%m-%d')
                hora = datetime.datetime.now().strftime("%H:%M:%S")
                #minuto  = datetime.datetime.now().strftime("%M")
                #Rutina q actualiza puntos si hay detección
                #print(input_q.qsize(),drk_q.qsize(),sensib_q.qsize(),isort_q.qsize())
                if isort_q.empty() != True:
                    i_sort =isort_q.get()
                    isort_q.task_done()
                    isort_q.queue.clear()
                else:
                    i_sort = np.array([])
                
                if len(i_sort)>0 or len(bg_detect)>0:
                    #print(i_sort.shape,"i_sort.shape")
                    #zen = sensib_q.get()
                    #sensib_q.queue.clear()
                    self.tracks, id0, indice, dclase=nuevos_ptos(i_sort,frame_gray,frame, self.tracks, id0, indice, bg_detect )
                else:
                    if bg_detect.shape[0]>0:
                        self.tracks, id0, indice, dclase=nuevos_ptos(
                            frame_gray,bg_detect, self.tracks, id0, indice, bg_detect )

                #rutina que elimina puntos zombi
                dimen_tmp = {}
                for u in np.unique(indice):
                    dimen_tmp[int(u)]=dimen[int(u)]
                    if int(u) in ciclos:
                        if datetime.datetime.now()-ciclos[int(u)]>timedelta(seconds=5): # antes 8seg
                            self.tracks, indice = borra_ptos(int(u), self.tracks, indice)
                            del ciclos[int(u)]
                dimen = dimen_tmp

                if b_ground_sub:
                    hermano_tmp={}
                    for (objectID, centroid) in objects.items():
                        if objectID in hermano.keys():
                            hermano_tmp[objectID] = hermano[objectID]
                    hermano=hermano_tmp
                
                #Mantención diaria de la base de datos e indices, borra tablas y comprime.
                if len(self.tracks) < 1:
                    if fecha > ayer:
                        id0 = 0
                        idprocesados = []
                        self.frame_idx = retardo + 10
                        puntaje =np.zeros((len(dibujo)))
                        hilo3 = threading.Thread(name='export_db', 
                                        target=export_db,
                                        args=(),
                                        daemon=False)
                        basedatos.borra(conn)
                        self.anchos_mx = np.zeros((4,4),dtype=int)
                        self.anchos_mx[:,(2,3)]=416,416
                        ayer = fecha
                
                if self.frame_idx % self.detect_interval*2 == 0:                  
                # Rutina contador Congestión #
                # Crea matriz 'CON' con datos de id y n° de frames que un id(col=0) 'vive' en un area (columna>0)
                    tmp_con = np.zeros(vect.shape)
                    for obj in np.unique(indice).tolist():
                        if obj in con[:,0]:
                            ptsout= basedatos.consulta_obj(conn,obj)
                            ptsout = ptsout.reshape(-1,2)
                            for e in range(0,len(mapa)):
                                if any(mapa[e].contains_points(ptsout)):
                                    #congestion[obj]=e
                                    con[np.nonzero(con[:,0]==obj)[0][0]][e+1]+=10
                                    #print(con, obj)
                                else:
                                    con[np.nonzero(con[:,0]==obj)[0][0]][e+1]=0
                                #if segundero > (25*10) and e > 1:
                                #    print('obj:%s seg:%s en area:%s ' %(obj, segundero/25,e))
                        else:
                            vect[0][0]=obj
                            con = np.append(con,vect,0)
                        tmp_con=np.append(tmp_con,con[con[:,0]==obj,...],0)
                    con = tmp_con
                    print(con)
                    if datetime.datetime.now()-crono_alarm_s>timedelta(seconds=1):
                        for par, buf_ in enumerate(buf_conges):
                            buff, bloq = buf_
                            if (con[:,buff+1]>100).sum() >= 1 and (con[:,bloq+1]>=100).sum() >=1: #10 y 200
                                alarma_bl = True
                                alarma = alarma.append({
                                             "detector": bloq,
                                             "tiempo": datetime.datetime.now(),
                                             "tipo": 0
                                              }, ignore_index=True)
                                print(bloq,datetime.datetime.now())
                                cv.imwrite(outputdir+"alarma.0.jpg",frame)
                            if (con[:,bloq+1]>=500).sum() >=1:
                                alarma_bl = True
                                alarma = alarma.append({
                                             "detector": bloq,
                                             "tiempo": datetime.datetime.now(),
                                             "tipo": 1
                                              }, ignore_index=True)
                                print(bloq,datetime.datetime.now())
                                cv.imwrite(outputdir+"alarma.1.jpg",frame)
                            else:
                                alarma_bl = False
                        crono_alarm_s = datetime.datetime.now()

                    if datetime.datetime.now()-crono_alarm>timedelta(seconds=60) and alarma.count().tolist()[0] >0: 
                        alarma_list = []
                        for par, buf_ in enumerate(buf_conges):
                            buff, bloq = buf_
                            for tipo in alarma.tipo.unique():
                                consulta = 'detector == %s & tipo==%s' % (bloq,tipo)
                                fin=alarma.query(consulta).iloc[[-1]]['tiempo']
                                ini=alarma.query(consulta).iloc[[0]]['tiempo']
                                alarmaseg = int((fin.tolist()[0]-ini.tolist()[0]).total_seconds())
                                cuenta=alarma.query(consulta).count().tolist()[0]
                                print("alarma #",bloq, cuenta, alarmaseg, hora, "tipo",tipo)
                                alarma_list = [id_flujo,bloq,fecha,hora,
                                        detec_tipo[str(bloq)] ,junction, detec_arco[str(bloq)],
                                        detec_sen[str(bloq)], detec_vir[str(bloq)],tipo,cuenta, alarmaseg]
                                if tipo == 0:
                                    foto_file = outputdir+'alarma.0.jpg'
                                elif tipo ==1:
                                    foto_file = outputdir+'alarma.1.jpg'
                                hilo_alarm = threading.Thread(name='update_bd_remota_alarma', 
                                            target=update_bd_alarma,
                                            args=([alarma_list, id_flujo, foto_file]),
                                            daemon=False).start()
                        
                        alarma = pd.DataFrame(columns=["detector", "tiempo", "tipo"])
                        crono_alarm = datetime.datetime.now()

                        #camara,detector,fecha,hora, detec_tipo[str(e)],junction, detec_arco[str(e)],detec_sen[str(e)], detec_vir[str(e)]
                


            ## registro video
            if graba_video:
                if datetime.datetime.now()-crono_vid>timedelta(seconds=900): # 300 seg = 5 min
                    vidfileold = vidfile
                    vid_writer.release()
                    vidfile=outputdir+str(fecha)+'_'+str(hora)+'_'+str(id0)+'.avi'
                    vid_writer = cv.VideoWriter(vidfile,fourcc, 25, (dim_x,dim_y))
                    crono_vid = datetime.datetime.now()
                    ### comprime video
                    hilo2 = threading.Thread(name='comprime_vid', 
                                            target=comprime_vid,
                                            args=([vidfileold]),
                                            daemon=False)
                    hilo2.start()
                vid_writer.write(vis)
            ## fin registro video

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('Streetflow.cl', vis)

            ch = cv.waitKey(1)
            if ch == 27 :
                print(threading.active_count(),"threading.active_count()",threading.get_ident(),"current_thread\n", threading.enumerate())
                break
            if ch == ord('n'): 
                if bgshow:
                    bgshow = False
                    cv.destroyWindow(winbGround)
                else:
                    bgshow = True
            if ch == ord('b'): 
                if b_ground_sub:
                    b_ground_sub = False
                else:
                    b_ground_sub = True
            if ch == ord('g'):
                if gammashow:
                    gammashow = False
                else:
                    gammashow = True
            if ch == ord('a'):
                if area_mask:
                    area_mask = False
                else:
                    area_mask = True
            if ch == ord('v'):
                if graba_video:
                    graba_video = False
                else:
                    graba_video = True
            if ch == ord('r'):
                if graba_video:
                    re_entrenar = False
                else:
                    graba_video = True


            #print( " %0.4f resto y sum total" %(timer()-inicio))

        #capo.stop()
        sckt_ret.clear()
        hilo10.join(timeout=1)
        print(sckt_ret.is_set(),"sckt_ret.is_set()")
        cap.release()
        vid_writer.release()
        #### update  tabla contador
        row, id_cont=basedatos.consulta_contador(conn,id_cont)
        bdremot.ins_postgres(row)
        crono = datetime.datetime.now()
        #### update  tabla contador
        export_db() #exporta base sqlite
        #np.savetxt("anchos_mx.csv", self.anchos_mx, delimiter=",")
        bdremot.get_ini(4,id_flujo) # Deja el flujo de video en estado 0, disponible.
        
def main():
    print(__doc__)
    App().run()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()