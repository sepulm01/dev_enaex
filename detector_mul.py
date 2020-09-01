#!/usr/bin/env python

'''
Recdata SPA.
Ene - 2020
Detector de movimiento

'''

from __future__ import print_function
from __future__ import division
import numpy as np
import cv2 as cv
from centroidtracker import CentroidTracker
#import darknet
#from drk import Drk
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
#import imagezmq
#from sql.bd_django import basedatos
from sql.basedatospg import bdremota
import requests
import json
import matplotlib.path as mpltPath
import sys
import pytz
from tzlocal import get_localzone
from django.db import models

no_resetear = True
input_q = queue.Queue(maxsize=1500)
drk_q = queue.LifoQueue(maxsize=1500)
sensib_q = queue.Queue(maxsize=1500)
isort_q  = queue.LifoQueue(maxsize=250)
tz=get_localzone()

class App:
    def __init__(self):
        self.track_len = 10 #10
        self.detect_interval =  10#10
        self.tracks = []
        self.frame_idx = 0
        self.anchos_mx = np.zeros((4,4),dtype=int)


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
            print(det_)
            if det_ == '' or det_ == None:
                mapas, dibujo = '', ''
            else:
                jareas=json.loads(det_)
                areas = []
                for i in jareas:
                    area = []
                    for ii in i:
                        area.append(list(ii.values()))
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
                            #logger.warning("Sokcet: comunicacion interrumpida con serv inferencias ")
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

#####################Backgroud detector ######################################
        def filter_mask( img, a=None):
            '''
            Filtro para tratar el background substration
            '''
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)) #(2, 2)
            # llena los agujeros pequeños
            closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
            # Remueve el ruido
            opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
            # Dilata los blobs adjacentes
            dilation = cv.dilate(opening, kernel, iterations=3  )
            return dilation

        def detect_mov(fg_mask):
            min_contour_width=10
            min_contour_height=10
            max_contour_width=590
            max_contour_height=590
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
            ahora=datetime.now().time() # ahorita mismo.
            cero_horas = datetime.strptime('00:00:00.0', '%H:%M:%S.%f')
            media_noche = datetime.strptime('23:59:59.0', '%H:%M:%S.%f')
            if ini == None or fin == None:
                return True
            else:
                if ini < fin: # operación normal
                    if ini < ahora and ahora < fin:
                        return True
                    else:
                        return False
                elif ini > fin: # de un dia para el otro
                    if ini < ahora and ahora < media_noche.time():
                        return True
                    elif cero_horas.time() < ahora and ahora < fin:
                        return True
                    else:
                        return False

        #################### settings #######################

        area_mask = True # variable de mascara para la detección
        actualizar = datetime.now(tz) # Variable de update de estado de camara y envio de foto a base y website
        alarma = False
        fecha = datetime.today().strftime('%Y-%m-%d')
        hora = datetime.now(tz).strftime("%H:%M:%S")
        record=datetime.now(tz)
        retardo = 0 # retardo para la grabación
        #conn = basedatos.create_connection()
        #disp=basedatos.cam_disp(conn)
        conn = bdremota()
        disp=bdremota.cam_disp(conn)
        while 1:
            if disp != []:
                bdremota.cam_ocupada(conn, True,disp[0][0])
                break
            else:
                #disp=basedatos.cam_disp(conn)
                disp=bdremota.cam_disp(conn)
                print("esperando camara disponible")
                time.sleep(5)

        print(disp)
        sensibilidad = disp[0][3]
        secreto = disp[0][10]
        url_alarm=disp[0][11]
        fuente = disp[0][4]
        if fuente == '0':
            fuente = 0

        nombre_cam = disp[0][1]
        cam_id = disp[0][0]
        fin = disp[0][8] # inicio de turno operación
        ini = disp[0][9] # fin turno operacion
        turno=en_turno(ini,fin)
        print('Turno:',turno)
        mapa, dibujo=mapa_areas_bd(disp[0][7])
        
        con = np.zeros((1,len(dibujo)+1)) # Matriz de congestion
        con[0][0]=1 # Variable de congestion
        
        cap = cv.VideoCapture(fuente)
        _ret, frame = cap.read()
        dim_y, dim_x, _ = frame.shape
        dim_y, dim_x = int(dim_y/2), int(dim_x/2)
        dim_y, dim_x = 608,608
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_rate = fps
        prev = 0

        id0 = 0
        output_dir = 'mysite/media/alarmas/'
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
            print ("fallo la creación del socket con error %s" %(err))

        try: 
            host_ip = socket.gethostbyname(host) 
        except socket.gaierror: 
            print ("there was an error resolving the host")
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
            time.sleep(2)

        sckt_ret = threading.Event()
        sckt_ret.set()
     
        hilo10 = threading.Thread(name='envia', 
            target=envia,
            args=([client_socket]),
            daemon=True)
        hilo10.start()
        

        ######FIN SOCKETS###########################
        
        while(cap.isOpened()):
            inicio = timer()
            time_elapsed = time.time() - prev
            if time_elapsed > 1./frame_rate:
                _ret, frame_ori = cap.read()
                prev = time.time()
                #frame = cv.resize(frame_ori, (dim_x,dim_y), interpolation = cv.INTER_AREA)
                if _ret == False:
                    print("termino flujo de video (ret==0).", nombre_cam )
                    no_resetear=True
                    break #TODO sacar en produccion
                    while _ret == False:
                        time.sleep(20)
                        cap = cv.VideoCapture(fuente)
                        _ret, frame = cap.read()
                frame = cv.resize(frame_ori, (dim_x,dim_y), interpolation = cv.INTER_AREA)
                

                #Graba con retardo
                output_rgb = frame
                if _ret:
                    input_q.put(frame)
                    if self.frame_idx > retardo:
                         output_rgb = cv.cvtColor(input_q.get(), cv.COLOR_RGB2BGR)
                         input_q.task_done()


                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                if area_mask: #Coloca una mascara al area de detección para limitar el n° de ids seguidor.
                    frame_msk = cv.add(mask_fg,frame_gray)
                    draw_str(frame, (350, 40), 'Area')
                else:
                    frame_msk = frame

                ############background sub #######################################
                if b_ground_sub:
                    draw_str(frame, (350, 27), 'BG Sub')
                    #dst = cv.add(mask_fg,frame_msk) #frame2bg
                    fg_mask = bg_subtractor.apply(frame_msk, None, 0.001) # back ground detector
                    fg_mask = filter_mask(fg_mask) # back ground detector
                    bg_detect,centroide = detect_mov(fg_mask)  # back ground detector
                    rects = []
                    for e in bg_detect:
                        rects.append(e[0:4].astype("int"))
                        x,y,a,b = e[0],e[1],e[2],e[3]
                        cv.rectangle(fg_mask, (x, y), (a, b) , (255, 255, 255), 1)
                    #print(len(bg_detect))

                    objects= ct.update(rects,centroide)
                    for (objectID, centroid) in objects.items():
                        text = "ID {}".format(objectID)
                        cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        cv.circle(fg_mask, (centroid[0], centroid[1]), 4, (2, 255, 255), -1)
                else:
                    bg_detect = np.array([])

                if len(bg_detect) > 0 and self.frame_idx % self.detect_interval == 0:
                    #print('frame_msk',frame_msk.shape,'bg_detect', bg_detect)
                    #drk_q.put(frame_msk) # alimenta la cola del detector
                    drk_q.put(frame)
                    sensib_q.put(sensibilidad) # Sensibilidad de detección o nivel de confianza 
                    draw_str(frame, (10, 30), str(drk_q.qsize()))

                if isort_q.empty() != True:
                    i_sort =isort_q.get()
                    isort_q.task_done()
                    isort_q.queue.clear()
                    #print('i_sort',i_sort)
                    record=datetime.now(tz)+timedelta(seconds=2)
                    
                    if record > datetime.now(tz) and alarma == False and turno:
                        fecha = datetime.today().strftime('%Y-%m-%d')
                        hora = datetime.now(tz).strftime("%H:%M:%S")
                        vidfile=str(nombre_cam)+'_'+str(fecha)+'_'+str(hora)+'_'+str(id0)+'.avi'
                        v_path = output_dir+vidfile
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
                        rq = requests.post(url_alarm, data=payload)
                        #bdremota.alarma(conn, cam_id,datetime.now(tz) ,str(list(i_sort[:,4])),list(i_sort.shape)[0],vidfile) 
                        print("comm base time", timer()-upd_base)
                        #bdremota.alarma(conn, 1,tiempo ,1,3,vidfile) 
                    for e in i_sort:
                        x,y,a,b = e[0],e[1],e[2],e[3]
                        x = int(np.round(x * x_scale))
                        y = int(np.round(y * y_scale))
                        a = int(np.round(a * x_scale))
                        b = int(np.round(b * y_scale))
                        cv.rectangle(output_rgb, (x, y), (a, b) , (255, 0, 255), 2) 
                else:
                    i_sort = np.array([])
                

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
                    #basedatos.cam_viva(conn, 1, cam_id, data, datetime.now(tz))
                    #bdremota.cam_viva(conn, True, cam_id, data, datetime.now(tz))
                    disp = bdremota.cam_viva(conn, True, cam_id, data, datetime.now(tz))
                    
                    actualizar = datetime.now(tz) + timedelta(seconds=60)
                    print(disp,"disp\n", len(disp))
                    fin = disp[0][8] # inicio de turno operación
                    ini = disp[0][9] # fin turno operacion
                    turno=en_turno(ini,fin)

                    print("Actualizado estado en ",self.frame_idx, 'y esta en el turno?',turno) 

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
                        color = (0,255,255)
                        if self.frame_idx % 10 == 0:
                            color = (0,255,255)
                    pts = np.array(dibujo[i], np.int32)
                    cv.polylines(output_rgb,[pts],True, color,2)
                    #cv.fillPoly(mask_fg, pts, (1, 1, 1))
                    #cv.polylines(mask_fg,[dibujo[i]],True,(255, 255, 255), 1) # background sub
                    cv.putText(output_rgb, str(i), totuple((dibujo[i])[0:1][0]), cv.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0),2)

                draw_str(output_rgb, (10, 15), nombre_cam)
                draw_str(output_rgb, (10, 30), "input_q  " + str(input_q.qsize()))
                draw_str(output_rgb, (10, 45), "drk_q    " + str(drk_q.qsize()))
                draw_str(output_rgb, (10, 60), "sensib_q " + str(sensib_q.qsize()))
                draw_str(output_rgb, (10, 75), "isort_q  " + str(isort_q.qsize()))
 

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

        cv.destroyAllWindows()
        bdremota.cam_ocupada(conn, False,disp[0][0])
        print(self.frame_idx, 'self.frame_idx') 
        client_socket.shutdown(socket.SHUT_RDWR)
        client_socket.close()

def main():
    while no_resetear:
        
        print('no_resetear',no_resetear)
        print(__doc__)

        App().run()
        cv.destroyAllWindows()

        disp=None
        with input_q.mutex:
            input_q.queue.clear()


if __name__ == '__main__':
    main()