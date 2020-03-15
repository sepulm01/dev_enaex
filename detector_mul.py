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
import imagezmq
from sql.bd_django import basedatos
import json
import matplotlib.path as mpltPath


#q = queue.LifoQueue(maxsize=1000)
input_q = queue.Queue(maxsize=1500)
drk_q = queue.LifoQueue(maxsize=1500)
sensib_q = queue.Queue(maxsize=1500)
isort_q  = queue.LifoQueue(maxsize=250)

class App:
    def __init__(self):
        self.track_len = 10 #10
        self.detect_interval =  10#10
        self.tracks = []
        self.frame_idx = 0
        self.anchos_mx = np.zeros((4,4),dtype=int)
        self.detect_interval = 10

    def run(self):
        def totuple(a):
            try:
                return tuple(totuple(i) for i in a)
            except TypeError:
                return a

        def update_alarma(camara,tiempo ,clase,cantidad,video):
            print('Hilo:',threading.current_thread().getName(),'con identificador:',threading.current_thread().ident)
            basedatos.alarma(camara,tiempo ,clase,cantidad,video) 
        ##################### Areas #####################################################

        def mapa_areas_bd(det_): 
            '''Carga areas desde base datos, entrega un objeto tipo mpltPath
            y una lista con N np array (1x2) con los puntos x,y del perimetro del area.
            Donde el N es el id del area o detector.
            '''
            if det_ == '':
                mapa, dibujo = '', ''
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
            return mapa, dibujo

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
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)) #(2, 2)
            # llena los agujeros pequeños
            closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
            # Remueve el ruido
            opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
            # Dilata los blobs adjacentes
            dilation = cv.dilate(opening, kernel, iterations=1  )
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

        #################### settings #######################
        area_mask = False # variable de mascara para la detección
        actualizar = datetime.now() # Variable de update de estado de camara y envio de foto a base y website
        alarma = False
        fecha = datetime.today().strftime('%Y-%m-%d')
        hora = datetime.now().strftime("%H:%M:%S")
        record=datetime.now()
        retardo = 20 
        conn = basedatos.create_connection()
        disp=basedatos.cam_disp(conn)
        print(disp)
        sensibilidad = disp[0][2]
        fuente = disp[0][3]
        if fuente == '0':
            fuente = 0
        basedatos.cam_ocupada(conn, 1,disp[0][0])
        nombre_cam = disp[0][1]
        cam_id = disp[0][0]
        mapa, dibujo=mapa_areas_bd(disp[0][4])
        con = np.zeros((1,len(dibujo)+1)) # Matriz de congestion
        con[0][0]=1 # Variable de congestion
        intermitente = False # flag intermitencia en color de las areas.
        cap = cv.VideoCapture(fuente)
        _ret, frame = cap.read()
        dim_y, dim_x, _ = frame.shape
        id0 = 0
        vidfile='output/'+str(nombre_cam)+'_'+str(fecha)+'_'+str(hora)+'_'+str(id0)+'.avi'
        fourcc = cv.VideoWriter_fourcc(*'XVID') #XVID X264 
        vid_writer = cv.VideoWriter(vidfile,fourcc, 20.0, (dim_x,dim_y))
        b_ground_sub = True
        bgshow = False
        bg_subtractor = cv.createBackgroundSubtractorMOG2(history=400, varThreshold = 20, detectShadows=True) #detecta mov de la camara
        winbGround = 'background subtraction'
        ct = CentroidTracker()
        mask_fg = np.ones(frame.shape, dtype = "uint8") # background sub
        cv.rectangle(mask_fg,(0,0),(dim_y, dim_x),(255, 255, 255),-1) 
        for i in range(0,len(dibujo)):
            pts = np.array(dibujo[i], np.int32)
            cv.fillPoly(mask_fg, [pts], (1, 1, 1))

        targetSize = 416
        x_scale = dim_x / targetSize
        y_scale = dim_y / targetSize

                ###### SOCKETS ############################
        host =  '0.0.0.0'
        #host = 'multiserver'
        port = 12347
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        client_socket.connect((host, port))

        info = client_socket.recv(1024)
        print('Recibido', repr(info))
        info = repr(info)

        sckt_ret = threading.Event()
        sckt_ret.set()
     
        hilo10 = threading.Thread(name='envia', 
            target=envia,
            args=([client_socket]),
            daemon=True)
        hilo10.start()
        
        ######FIN SOCKETS###########################

        #sender = imagezmq.ImageSender(connect_to='tcp://*:5566', REQ_REP=False)
        #sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')

        while(cap.isOpened()):
            _ret, frame = cap.read()

            if _ret == False:
                print("termino flujo de video (ret==0).", nombre_cam )
                
                break #TODO sacar en produccion
                while _ret == False:
                    time.sleep(20)
                    cap = cv.VideoCapture(fuente)
                    _ret, frame = cap.read()
            draw_str(frame, (10, 15), nombre_cam)

            #Graba con retardo
            output_rgb = frame
            if _ret:
                input_q.put(frame)
                if self.frame_idx > retardo:
                     output_rgb = cv.cvtColor(input_q.get(), cv.COLOR_RGB2BGR)
                     input_q.task_done()

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if area_mask: #Coloca una mascara al area de detección para limitar el n° de ids seguidor.
                frame_msk = cv.add(mask_fg,frame)
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
                drk_q.put(frame_msk) # alimenta la cola del detector
                sensib_q.put(sensibilidad) # Sensibilidad de detección o nivel de confianza 
                draw_str(frame, (10, 30), str(drk_q.qsize()))

            if isort_q.empty() != True:
                i_sort =isort_q.get()
                isort_q.task_done()
                isort_q.queue.clear()
                print(i_sort)
                record=datetime.now()+timedelta(seconds=2)
                
                if record > datetime.now() and alarma == False:
                    fecha = datetime.today().strftime('%Y-%m-%d')
                    hora = datetime.now().strftime("%H:%M:%S")
                    vidfile='output/'+str(nombre_cam)+'_'+str(fecha)+'_'+str(hora)+'_'+str(id0)+'.avi'
                    vid_writer = cv.VideoWriter(vidfile,fourcc, 20.0, (dim_x,dim_y))
                    alarma = True
                    tiempo ,clase,cantidad,video = datetime.now(), 2, 2, vidfile
                    basedatos.alarma(conn, cam_id,tiempo ,str(list(i_sort[:,4])),list(i_sort.shape)[0],video) 
                    # hilo1 = threading.Thread(name='update_alarma', 
                    #     target=update_alarma,
                    #     args=(camara,tiempo ,clase,cantidad,video),
                    #     daemon=False)
                    # hilo1.start()
                for e in i_sort:
                    x,y,a,b = e[0],e[1],e[2],e[3]
                    x = int(np.round(x * x_scale))
                    y = int(np.round(y * y_scale))
                    a = int(np.round(a * x_scale))
                    b = int(np.round(b * y_scale))
                    cv.rectangle(frame, (x, y), (a, b) , (255, 0, 255), 2) # TODO recomponer para la pintana
            else:
                i_sort = np.array([])


            if bgshow and b_ground_sub:
                cv.imshow(winbGround,fg_mask)
            ############fin background sub #######################################
            # print(capo.isort)
            # for e in capo.isort:
            #     x,y,a,b = e[0],e[1],e[2],e[3]
            #     cv.rectangle(frame, (x, y), (a, b) , (255, 255, 255), 1)
            if record > datetime.now():
                vid_writer.write(output_rgb)
            else:
                alarma = False


            if actualizar < datetime.now():
                # cada 1200 frames actualiza estado y envia una foto a base de datos
                #, est, pk, img, actualizado):
                data = cv.imencode('.jpg', frame)[1].tostring()
                basedatos.cam_viva(conn, 1, cam_id, data, datetime.now())
                print("Actualizado estado en ",self.frame_idx) 
                actualizar = datetime.now() + timedelta(seconds=60)

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
                cv.polylines(frame,[pts],True, color,2)
                #cv.fillPoly(mask_fg, pts, (1, 1, 1))
                #cv.polylines(mask_fg,[dibujo[i]],True,(255, 255, 255), 1) # background sub
                cv.putText(frame, str(i), totuple((dibujo[i])[0:1][0]), cv.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0),2)

            cv.imshow("Frame", frame)
            #sender.send_image(nombre_cam, frame)

            ch = cv.waitKey(1)
            if ch == 27 :
#                capo.stop()
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

            self.frame_idx += 1

        cv.destroyAllWindows()
        basedatos.cam_ocupada(conn, 0,disp[0][0])

def main():
    print(__doc__)
    App().run()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()