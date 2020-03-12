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


#q = queue.LifoQueue(maxsize=1000)
input_q = queue.Queue(maxsize=1500)
drk_q = queue.LifoQueue(maxsize=1500)
sensib_q = queue.Queue(maxsize=1500)
isort_q  = queue.LifoQueue(maxsize=250)


######################### Darknet ###################################################
# netMain = None
# metaMain = None
# configPath = "./cfg/yolov3.cfg"
# #configPath = "./cfg/yolov3-8clases.cfg"
# weightPath = "./cfg/yolov3.weights" 
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
    clases = [0,1,2,3,4,5,6,7,8,9] #clases a seguir

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
    print("drk detect",timer()-antes)
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

#    def start(self):
        # if self.started:
        #     print('deteccion ha empezado')
        #     return None
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
            #sensib = sensib_q.get()
            self.isort = dark(frame,0.2)
            drk_q.task_done()
            drk_q.queue.clear()
            #print(drk_q.qsize())
            #sensib_q.task_done()
            #sensib_q.queue.clear()
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
        self.detect_interval = 10

    def run(self):

        def update_alarma(camara,tiempo ,clase,cantidad,video):
            print('Hilo:',threading.current_thread().getName(),'con identificador:',threading.current_thread().ident)
            basedatos.alarma(camara,tiempo ,clase,cantidad,video) 
        ##################### Areas #####################################################
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
        alarma = False
        fecha = datetime.today().strftime('%Y-%m-%d')
        hora = datetime.now().strftime("%H:%M:%S")
        record=datetime.now()
        retardo = 20 
        conn = basedatos.create_connection()
        disp=basedatos.cam_disp(conn)
        print(disp)
        fuente = disp[0][3]
        basedatos.cam_ocupada(conn, 1,disp[0][0])
        nombre_cam = disp[0][1]
        cam_id = disp[0][0]
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
        #capo = mul.Process(target=dark, args[frame, 0,5]) # inicialicia detector
        #capo.start() # parte detector de objetos

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
        sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')

        while(cap.isOpened()):
            _ret, frame = cap.read()

            if _ret == False:
                print("termino flujo de video (ret==0).")
                basedatos.cam_ocupada(conn, 0,disp[0][0])
                break #TODO sacar en produccion
                while _ret == False:
                    time.sleep(20)
                    cap = cv.VideoCapture(fuente)
                    _ret, frame = cap.read()
            draw_str(frame, (10, 15), nombre_cam)
            output_rgb = frame
            if _ret:
                input_q.put(frame)
                #inp_q= input_q.qsize()
                #drk_q.put(frame) # alimenta la cola del detector
                #wqu= drk_q.qsize()
                #print(inp_q)
                if self.frame_idx > retardo:
                     output_rgb = cv.cvtColor(input_q.get(), cv.COLOR_RGB2BGR)
                     input_q.task_done()
                #     frame = input_q.get()
                #     w= input_q.qsize()
                #     dr= drk_q.qsize()

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            ############background sub #######################################
            if b_ground_sub:
                #draw_str(vis, (350, 27), 'BG Sub')
                #dst = cv.add(mask_fg,frame2bg) #frame2bg
                fg_mask = bg_subtractor.apply(frame, None, 0.001) # back ground detector
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
                drk_q.put(frame) # alimenta la cola del detector
                sensib_q.put(5)
                draw_str(frame, (10, 30), str(drk_q.qsize()))
            #print("bg_detect", len(bg_detect) )

            if isort_q.empty() != True:
                i_sort =isort_q.get()
                isort_q.task_done()
                isort_q.queue.clear()
                print(i_sort, type(i_sort))
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
                    cv.rectangle(frame, (x, y), (a, b) , (255, 255, 255), 1) # TODO recomponer para la pintana
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
            cv.imshow("Frame", frame)

            if self.frame_idx % 1200 == 0:
                # cada 1200 frames actualiza estado y envia una foto a base de datos
                data = cv.imencode('.jpg', frame)[1].tostring()
                basedatos.cam_viva(conn, 1, 1, data, datetime.now())


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

def main():
    print(__doc__)
    App().run()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()