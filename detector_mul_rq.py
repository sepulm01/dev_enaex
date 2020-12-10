#!/usr/bin/env python

'''
Recdata SPA.
Ene - 2020
Detector de movimiento y reconocmiento de objeto.

'''

from __future__ import print_function
from __future__ import division
import numpy as np
import cv2 as cv
from centroidtracker import CentroidTracker
import os
import queue
import threading
import time
from timeit import default_timer as timer
from datetime import datetime, timedelta
from common import anorm2, draw_str
import multiprocessing as mul
import socket
import random
from struct import *
import requests
import json
import matplotlib.path as mpltPath
import sys
import pytz
from tzlocal import get_localzone
import logging
from scipy.spatial import distance
import Crypto
from Crypto.PublicKey import RSA
import binascii 
from Crypto.Cipher import PKCS1_OAEP

no_resetear = True
input_q = queue.Queue(maxsize=1500)
drk_q = queue.LifoQueue(maxsize=1500)
sensib_q = queue.Queue(maxsize=1500)
isort_q  = queue.LifoQueue(maxsize=250)
tz=get_localzone()

cam_worker = os.environ.get('CAM')
if cam_worker == None:
    print("Falta variable de ambiente del worker de la camara")
    exit()

LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename=str(cam_worker)+"detector_mul_rq.log",
    level = logging.DEBUG,
    format = LOG_FORMAT,
    filemode = "w")
logger = logging.getLogger()
logger.info("Inicio logger")

private_key="3082025c02010002818100bf32f1f92407337ca27ce93724c858f79e2deea0149ed9d7d1ee3c9599b0d06dc55818f2a89efb3cc2d9021d07e9659bc98bc6a115dfd23ea28b83bd94388bba5af9161ed9e91055fec8a8de2cac5a0fb9e9e8d23d376969a8a2b62071a979faf3c6825dcbcb90f04467d6678674eade19d1f797e1d563fb162b8212b4c5a631020301000102818012b12154d0ffdf39b50cef23d3f5be34df02f08c37d7dbc62ca0d4cd6f4c08e462619d76c3a35f3e6e7216b1cddf346ec9825fb5c9d4aad232c3deea3ebe5472853b9e563569ff78f07660dc2db144b84f954f31a25e62154a9b57d53c32746b6bb797b6d81e5ec20518208fae162710be9b7487e1a5df02c7fec5eebc6acdad024100d3c54a671894041d9f511f1ed7b106cc37a3f5afd6425b63007e2932f7bfe3d0235ddcc36177d5f5736fd3a6572886e6d3cc1f14ab934d498b7be5c156ceffb3024100e721c0b25efdd3e263bac58a2b02803b39cec4de03b481df4fef124756343347c6e60d6ea1422790bcfc199e098bdffe8f03d7a8f6a859b6d0d3bbe72633f08b024100aa33265935a7c0a70e24649ea53be1fabfbd46f8cb7b0977c82d9d6f192f6029284387ea7fab908a74fcab5e452e8d3d777bd67f06669cf73ee395048e804f8102402fc5f0386e1df4efb44164973c7095e4a7fc2f00dcaf30b0e1aabe927424f1fc820606fcb8e41d9d7312809103d41f8654352d1c456f62abc0da22da9230e6250240584b05f73ed781ea083b736eff4eea11c8732ed96ad7e78cc0b02d8319d6cc718592d5ca50b4b7330a700214e3e2e6e653bdcb31b59fe4055d6068e9ce060e60"
private_key = RSA.importKey(binascii.unhexlify(private_key))

class App:
    def __init__(self):
        #self.track_len = 10 #10
        self.detect_interval =  5#10
        #self.tracks = []
        self.frame_idx = 0
        #self.anchos_mx = np.zeros((4,4),dtype=int)


    def run(self):
        def totuple(a):
            try:
                return tuple(totuple(i) for i in a)
            except TypeError:
                return a

        ##################### Areas #####################################################

        def mapa_areas_bd(det_): 
            '''Carga areas desde base datos, entrega un objeto tipo mpltPath
            y una lista con N np array (1x2) con los puntos x,y del perimetro del area.
            Donde el N es el id del area o detector.
            '''
            if det_ == '' or det_ == None:
                mapas, dibujo = '', ''
            else:
                jareas=json.loads(det_)
                
                areas = []
                for i in jareas:
                    area = []
                    for ii in i:
                        area.append(list((ii['x'],ii['y'])))
                    areas.append(area)
                mapas = []
                dibujo = []
                for a in areas:
                    mapas.append(mpltPath.Path(a))
                    dibujo.append(a)
            return mapas, dibujo

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
                #print("drk_q ",drk_q.qsize())
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
                        print("error env")
                        logger.warning("Sokcet: error en env")
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
                            logger.warning("Sokcet: comunicacion interrumpida con serv GPU")
                            #evento.set()
                            exit()
                        a = r.find(b'FIN!')
                        if a != -1:
                            data += r[:a]
                            break
                        data += r
                    except Exception as e:
                        print(e)
                        continue
                    #print("tiempo de Recepcion %.3f" %(timer()-inicio2))
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

#####################Backgroud detector ######################################
        def filter_mask( img, a=None):
            '''
            Filtro para tratar el background substration
            '''
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)) #(2, 2)
            # llena los agujeros pequenos
            closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
            # Remueve el ruido
            opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
            # Dilata los blobs adjacentes
            dilation = cv.dilate(opening, kernel, iterations=3  )
            return dilation

        def detect_mov(fg_mask, micw=20,mich=20,macw=590,mach=590):
            min_contour_width=micw
            min_contour_height=mich
            max_contour_width=macw
            max_contour_height=mach
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
                (x, y, w, h) = cv.boundingRect(contour)
                contour_valid = (w >= min_contour_width) and (w <= max_contour_width) and (
                    h <= max_contour_height) and (h >= min_contour_height) #and (h/w > 2.7)

                if contour_valid:
                    matches.append((x, y, x+w, y+h, 81, 0))
                    centroide.append((cX, cY))
            centroide=np.int32(centroide)
            matches=np.int32(matches)
            return matches, centroide

        ##################### turno ###########################
        def en_turno(ini,fin):
            format = '%H:%M:%S'
            fin=time.strptime(fin, format)
            ini=time.strptime(ini, format)
            ahora = str(time.localtime().tm_hour) +':'+ str(time.localtime().tm_min)+':00'
            ahora = time.strptime(ahora, format)
            cero_horas = time.strptime('00:00:00', format)
            media_noche = time.strptime('23:59:59', format)
            if ini == None or fin == None:
                return True
            else:
                if ini < fin: # operacion normal
                    if ini <= ahora and ahora < fin:
                        return True
                    else:
                        return False
                elif ini > fin: # de un dia para el otro
                    if ini <= ahora and ahora < media_noche:
                        return True
                    elif cero_horas <= ahora and ahora < fin:
                        return True
                    else:
                        return False

        ################# alarma ############################
        def gatilla_alarma(i_sort,secreto):
            fecha = datetime.today().strftime('%Y-%m-%d')
            hora = datetime.now(tz).strftime("%H:%M:%S")
            vidfile=str(nombre_cam)+'_'+str(fecha)+'_'+str(hora)+'_'+str(id0)+'.avi'
            v_path = output_dir+vidfile
            logger.info("pipeline: path video %s" %(v_path))
            dim_y_, dim_x_, _ = frame.shape
            vid_writer = cv.VideoWriter(v_path,fourcc, 20.0, (dim_x_,dim_y_))
            alarma = True
            #tiempo ,clase,cantidad = datetime.now(tz), 2, 2, vidfile
            upd_base = timer()
            payload = {
                'camara': nombre_cam,
                'tiempo': datetime.now(tz),
                'clase': str(list(i_sort[:,4])),
                'cantidad': list(i_sort.shape)[0],
                'video':vidfile,
                'secreto':secreto
                }
            envia_rq(url_alarm, payload)
            logger.info("requests:comm base time %s" % (timer()-upd_base)) 
            return vid_writer, alarma
        ################# fin alarma ############################

        ###################### rutinas requests ####################
        def envia_rq(dir_url, info):
            while 1:
                try:
                    rq = requests.post(dir_url, data=info)
                    if rq.ok:
                        break
                except requests.exceptions.HTTPError as errh:
                    logger.error("requests: Http Error:",errh)
                    break
                except requests.exceptions.ConnectionError as errc:
                    logger.error("requests: Error Connecting:",errc)
                    break
                except requests.exceptions.Timeout as errt:
                    logger.error("requests: Timeout Error:",errt)
                    break
                except requests.exceptions.RequestException as err:
                    logger.error("requests: OOps: Something Else",err)
                    break
                print("Reintentando en 0.2 segundos..., ")
                time.sleep(0.2)
                logger.info("requests: Reintentando en 0.2 segundos...")

        def recibe_rq(url_recibe):
            rq = []
            while 1:
                try:
                    rq = requests.get(url_get)
                    rq.raise_for_status()
                    if rq.ok:
                        print('comunicacion establecida')
                        logger.info("requests: comunicacion establecida con serv web")
                        break
                except requests.exceptions.HTTPError as errh:
                    print("Http Error:",errh)
                except requests.exceptions.ConnectionError as errc:
                    print("Error Connecting:",errc)
                except requests.exceptions.Timeout as errt:
                    print("Timeout Error:",errt)
                except requests.exceptions.RequestException as err:
                    print("OOps: Something Else",err)
                print("Reintentando en 5 segundos...")
                time.sleep(5)
                logger.info("requests: Reintentando en 5 segundos serv web")
            return rq

        ###################### fin rutinas requests ####################

        #################### settings #######################

        with open('conf.file') as json_file:
            datafile = json.load(json_file)
        urlweb=str(datafile['website'])
        output_dir=str(datafile['output_dir'])
        urlcamviva =urlweb+'/api/camviva/'
        url_get=urlweb+'/api/cam_disp/'+ str(cam_worker) +'/?a=2485987abr'
        url_alarm=urlweb+'/api/alarma/'

        area_mask = True # variable de mascara para la deteccion
        actualizar = datetime.now(tz) # Variable de update de estado de camara y envio de foto a base y website
        alarma = False
        fecha = datetime.today().strftime('%Y-%m-%d')
        hora = datetime.now(tz).strftime("%H:%M:%S")
        record=datetime.now(tz)
        retardo = 0 # retardo para la grabacion
        rq = recibe_rq(url_get)
        ajts = rq.json()
        sensibilidad = ajts['sensib'] 
        secreto = ajts['secreto']
        fuente = ajts['fuente']
        if fuente == '0':
            fuente = 0
        nombre_cam = ajts['nombre']
        cam_id = ajts['pk']
        fin = ajts['op_fin']
        ini = ajts['op_ini']
        turno=en_turno(ini,fin)
        detect_todo = ajts['detect_todo']
        dist_bg = ajts['dist_bg']
        rep_alar_ni = ajts['rep_alar_ni']
        mapa, dibujo=mapa_areas_bd(ajts['areas'])
        micw,mich,macw,mach = ajts['micw'],ajts['mich'],ajts['macw'],ajts['mach']
        con = np.zeros((1,len(dibujo)+1)) # Matriz de congestion
        con[0][0]=1 # Variable de congestion
        tiempo_min_serv = ajts['time_min']
        area_min_serv = ajts['min_area_mean']
        fecha_caducidad = ajts['fecha_caducidad']
        ######################  Validación Licenciamiento ############################
        clave = ajts['clave']
        clave = binascii.unhexlify(clave)
        cipher = PKCS1_OAEP.new(private_key)
        message = cipher.decrypt(clave)
        message = message.decode('utf-8')
        d1 = datetime.strptime(message,"%Y-%m-%dT%H:%M:%SZ") #"2013-07-12T07:00:00Z"
        if d1 < datetime.now():
            logger.warning("Licenciamiento: Fecha de fecha caducidad alcanzada")
            print("Licenciamiento: Fecha de fecha caducidad alcanzada",fecha_caducidad)
            time.sleep(30)
            exit()
        cap = cv.VideoCapture(fuente)
        _ret, frame = cap.read()
        if _ret == False:
            print("No hay imagen desde la fuente:", fuente)
            urlcamdown =urlweb+'/api/cam_down/'
            camdonw_info = {
                        'camara': nombre_cam,
                        'tiempo': datetime.now(tz),
                        'error_msg': "No hay imagen desde la fuente %s" %(fuente),
                        'fuente': fuente,
                        'secreto':secreto
                        }
            envia_rq(urlcamdown,camdonw_info)
            logger.warning("cap.read(): No hay imagen desde la fuente %s" %(fuente))
            time.sleep(5)
            exit()
            while _ret == False:
                time.sleep(5)
                cap = cv.VideoCapture(fuente)
                _ret, frame = cap.read()
        dim_y, dim_x, _ = frame.shape
        dim_y, dim_x = int(dim_y/2), int(dim_x/2)
        dim_y, dim_x = 608,608
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_rate = fps
        prev = 0
        id0 = 0
        #output_dir = 'mnt/alarmas/alarmas/' # Docker dentro de /lab
        #output_dir = 'mysite/media/alarmas/'
        if os.path.isdir(output_dir)==False:
            print('ERROR: Directorio destino de los videos, no existe:',output_dir)
            exit()
        vidfile=str(nombre_cam)+'_'+str(fecha)+'_'+str(hora)+'_'+str(id0)+'.avi'
        v_path = output_dir+vidfile
        #fourcc = cv.VideoWriter_fourcc(*'XVID') #XVID X264 
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        vid_writer = cv.VideoWriter(v_path,fourcc, 20.0, (dim_x,dim_y))
        
        b_ground_sub = True
        bgshow = False
        bg_subtractor = cv.createBackgroundSubtractorMOG2(history=400, varThreshold = 120, detectShadows=False) #detecta mov de la camara
        winbGround = 'background subtraction'
        
        ct = CentroidTracker()

        mask_fg = np.ones((dim_x,dim_y), dtype = "uint8") # background sub
        cv.rectangle(mask_fg,(0,0),(dim_y, dim_x),(255, 255, 255),-1) 
        for i in range(0,len(dibujo)):
            pts = np.array(dibujo[i], np.int32)
            cv.fillPoly(mask_fg, [pts], (1, 1, 1))

        targetSize = 608
        x_scale = dim_x / targetSize
        y_scale = dim_y / targetSize
        intermitente = False # flag intermitencia en color de las areas.
        objeto_ni = False #flat de gatilla alarma de objeto No Identificado
        contador_obj_det =0 #contador num veces obj detec antes de gatillar la alarma de los NI        
        mtx_bgdet = np.zeros((1,2))
        ###### SOCKETS ############################
        conecto = True
        host =  '0.0.0.0'
        #host = '10.0.2.4'
        port = 12347
        try: 
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            print ("Socket creado exitosamente")
        except socket.error as err: 
            print ("fallo la creacion del socket con error %s" %(err))
            logger.error("client_socket: fallo la creacion del socket con error %s" %(err))

        try: 
            host_ip = socket.gethostbyname(host) 
        except socket.gaierror: 
            print ("there was an error resolving the host")
            logger.error("client_socket: error resolviendo el host ")
        while conecto:
            try:
                client_socket.connect((host, port))
                info = client_socket.recv(1024)
                print('Recibido', repr(info))
                info = repr(info)
                conecto = False
            except:
                print("error")
            print("esperando servidor GPU en %s:%s" %(host,port) )
            logger.info("esperando servidor GPU en %s:%s" %(host,port))
            time.sleep(2)

        sckt_ret = threading.Event()
        sckt_ret.set()
     
        hilo10 = threading.Thread(name='envia', 
            target=envia,
            args=([client_socket]),
            daemon=True)
        hilo10.start()
        ######FIN SOCKETS###########################

        blobs = {}
        tracks = []
        while(cap.isOpened()):
            inicio = timer()
            timestamp = datetime.timestamp(datetime.now(tz))
            time_elapsed = time.time() - prev
            if time_elapsed > 1./frame_rate:
                _ret, frame_ori = cap.read()
                prev = time.time()
                #frame = cv.resize(frame_ori, (dim_x,dim_y), interpolation = cv.INTER_AREA)
                if _ret == False:
                    print("termino flujo de video (ret==0).", nombre_cam )
                    logger.info("pipeline: termino flujo de video (ret==0).")
                    no_resetear=True
                    break
                frame = cv.resize(frame_ori, (dim_x,dim_y), interpolation = cv.INTER_AREA)

                #Graba con retardo
                output_rgb = frame
                if _ret:
                    input_q.put(frame)
                    if self.frame_idx > retardo:
                         #output_rgb = cv.cvtColor(input_q.get(), cv.COLOR_RGB2BGR)
                         output_rgb = input_q.get()
                         input_q.task_done()

                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                if area_mask: #Coloca una mascara al area de deteccion para limitar el n de ids seguidor.
                    frame_msk = cv.add(mask_fg,frame_gray)
                    #draw_str(frame, (350, 40), 'Area')
                else:
                    frame_msk = frame

                ############background sub #######################################
                if b_ground_sub:
                    #draw_str(frame, (350, 27), 'BG Sub')
                    #dst = cv.add(mask_fg,frame_msk) #frame2bg
                    fg_mask = bg_subtractor.apply(frame_msk, None, 0.001) # back ground detector
                    fg_mask = filter_mask(fg_mask) # back ground detector
                    bg_detect,centroide = detect_mov(fg_mask,micw,mich,macw,mach)  # back ground detector
                    rects = []
                    for e in bg_detect:
                        rects.append(e[0:4].astype("int"))
                        x,y,a,b = e[0],e[1],e[2],e[3]
                        areapx = (a-x)*(b-y)
                        if alarma==False:
                            mtx_bgdet=np.append(mtx_bgdet,[[areapx,timestamp]],axis=0)
                        #print(mtx_bgdet, [areapx,timestamp])
                        cv.rectangle(fg_mask, (x, y), (a, b) , (255, 255, 255), 1)
                    #print(len(bg_detect))

                    objects= ct.update(rects,centroide)
                    objeto_ni = False
                    for (objectID, centroid) in objects.items():
                        text = "ID {}".format(objectID)
                        cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        cv.circle(fg_mask, (centroid[0], centroid[1]), 4, (2, 255, 255), -1)

                        if objectID in blobs.keys(): 
                            dst = distance.euclidean(blobs[objectID], centroid)
                            if dst > dist_bg:
                                objeto_ni = True
                                
                        else:
                            blobs[objectID]=[centroid] 
                else:
                    bg_detect = np.array([])
                if len(objects.items())==0 or len(blobs)>50 :
                    blobs ={}
                if objeto_ni:
                    contador_obj_det +=1

                if len(bg_detect) > 0 and self.frame_idx % self.detect_interval == 0 and en_turno(ini,fin):
                    #print('frame_msk',frame_msk.shape,'bg_detect', bg_detect)
                    if detect_todo:
                        drk_q.put(frame)
                    else:
                        drk_q.put(frame_msk) # alimenta la cola del detector
                    
                    sensib_q.put(sensibilidad) # Sensibilidad de deteccion o nivel de confianza 
                    #draw_str(frame, (10, 30), str(drk_q.qsize()))
  
                if isort_q.empty() != True:
                    logger.info("pipeline: deteccion hecha")
                    i_sort =isort_q.get()
                    isort_q.task_done()
                    isort_q.queue.clear()
                    #print('i_sort',i_sort)
                    objeto_ni = False # desactiva alarma No Identificado
                    contador_obj_det = 0 # desactiva alarma No Identificado
                    record=datetime.now(tz)+timedelta(seconds=3)
                    
                    if record > datetime.now(tz) and alarma == False and en_turno(ini,fin):
                        vid_writer, alarma = gatilla_alarma(i_sort,secreto)

                    for e in i_sort:
                        x,y,a,b = e[0],e[1],e[2],e[3]
                        x = int(np.round(x * x_scale))
                        y = int(np.round(y * y_scale))
                        a = int(np.round(a * x_scale))
                        b = int(np.round(b * y_scale))
                        cv.rectangle(output_rgb, (x, y), (a, b) , (255, 0, 255), 2) 
                else:
                    i_sort = np.array([])
                
                ########################## alarma bg sub ########################################################
                if alarma == False and en_turno(ini,fin) and objeto_ni==True and contador_obj_det>rep_alar_ni:
                    if mtx_bgdet.shape[0]>1:
                        ini_=datetime.fromtimestamp(mtx_bgdet[1][1])
                        fin_=datetime.fromtimestamp(mtx_bgdet[-1][1])
                        tiempo_min = fin_ - ini_
                        area_min = int(mtx_bgdet[:,0].mean())
                        tiempo_min_serv = timedelta(seconds=3)
                        print('tiempo_min_serv',tiempo_min_serv)
                        if tiempo_min > tiempo_min_serv and area_min > area_min_serv:
                            print(area_min , mtx_bgdet.shape, tiempo_min) 
                            mtx_bgdet = np.zeros((1,2))
                            # gatilla alarma de objeto No Indentificados.
                            print("pre pasa x alarma NI")
                            record=datetime.now(tz)+timedelta(seconds=3)
                            if record > datetime.now(tz):
                                print('contador_obj_det',contador_obj_det)
                                contador_obj_det =0
                                print('alarma',self.frame_idx )
                                alarma = True
                                ni = np.array([0, 0, 0, 0, 81, 0])
                                ni = ni.reshape((1,6))
                                vid_writer, alarma = gatilla_alarma(ni,secreto)

                if bgshow and b_ground_sub:
                    cv.imshow(winbGround,fg_mask)
                ############fin background sub #######################################

                if record > datetime.now(tz):
                    vid_writer.write(output_rgb)
                else:
                    alarma = False

                if actualizar < datetime.now(tz):
                    # 60s actualiza estado y envia una foto a base de datos
                    #, est, pk, img, actualizado):
                    data = cv.imencode('.jpg', output_rgb)[1].tostring()
                    data = data.decode("ISO-8859-1")
                    #print(secreto)
                    camvload = {
                        'camara': nombre_cam,
                        'tiempo': datetime.now(tz),
                        'image': data,
                        'secreto':secreto,
                        'lic': message,
                        }
                    print('actializa')
                    envia_rq(urlcamviva, camvload)
                    #rq = requests.post(urlcamviva, data=camvload)
                   
                    actualizar = datetime.now(tz) + timedelta(seconds=60)
                    rq =recibe_rq(url_get)
                    #rq = requests.get(url_get)
                    ajts = rq.json()
                    fin = ajts['op_fin']
                    ini = ajts['op_ini']
                    turno=en_turno(ini,fin)
                    #mapa, dibujo=mapa_areas_bd(disp[0][7]) ajts['areas']
                    mapa, dibujo=mapa_areas_bd(ajts['areas'])
                    
                    mask_fg = np.ones((dim_x,dim_y), dtype = "uint8") # background sub
                    cv.rectangle(mask_fg,(0,0),(dim_y, dim_x),(255, 255, 255),-1) 
                    for i in range(0,len(dibujo)):
                        pts = np.array(dibujo[i], np.int32)
                        cv.fillPoly(mask_fg, [pts], (1, 1, 1))

                    print("Actualizado estado en ",self.frame_idx, 'y esta en el turno?',ini,fin,turno) 
                    logger.info("pipeline: actualizado -en turno? %s" %(turno))

                for i in range(0,len(dibujo)):
                    color = (0,255,0)
                    zona = con[:,i+1].tolist()
                    for zi in zona:
                        #if zi > 125 and i>1: color = (0,0,255) discriminador de areas i > X
                        if zi > 10 : #150
                            color = (0,255,255)
                        if zi > 30: #200
                            color = (0,0,255)
                            #intermitente = True
                    if intermitente:
                        color = (0,255,255)
                        if self.frame_idx % 10 == 0:
                            color = (0,255,255)
                    pts = np.array(dibujo[i], np.int32)
                    cv.polylines(output_rgb,[pts],True, color,2)
                    #cv.fillPoly(mask_fg, pts, (1, 1, 1))
                    #cv.polylines(mask_fg,[dibujo[i]],True,(255, 255, 255), 1) # background sub
                    cv.putText(output_rgb, str(i), totuple((dibujo[i])[0:1][0]), cv.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0),2)

                draw_str(output_rgb, (10, 15), nombre_cam)
                draw_str(output_rgb, (10, 30), "alarma  " + str(alarma))
                #draw_str(output_rgb, (10, 45), "Turno    " + str(en_turno(ini,fin)))
                #draw_str(output_rgb, (10, 60), "objeto_ni " + str(objeto_ni))
                #draw_str(output_rgb, (10, 75), "pts  " + str(pts))

                if hilo10.is_alive()==False: #bota la app si se cae la coneccion a la GPU.
                    break
                if self.frame_idx > 1728000: # aprox 24 horas, reset lipieza
                    break

                cv.imshow("Frame", output_rgb)
                #sender.send_image(nombre_cam, frame)

                ch = cv.waitKey(1)
                if ch == 27 :
                    no_resetear = False
                    print('no_resetear',no_resetear)
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

                if ch == ord('a'):
                    if area_mask:
                        area_mask = False
                    else:
                        area_mask = True
                self.frame_idx += 1
                #print(timer()-inicio, "Tiempo")
        logger.warning("pipeline: brake ejecutado")
        cv.destroyAllWindows()
        #bdremota.cam_ocupada(conn, False,disp[0][0])
        print(self.frame_idx, 'self.frame_idx') 
        client_socket.shutdown(socket.SHUT_RDWR)
        client_socket.close()

def main():
    while no_resetear:
        
        
        print(__doc__)

        App().run()
        cv.destroyAllWindows()
        print('no_resetear',no_resetear)
        disp=None
        with input_q.mutex:
            input_q.queue.clear()
            drk_q.queue.clear() 
            sensib_q.queue.clear()
            isort_q.queue.clear() 

if __name__ == '__main__':
    main()