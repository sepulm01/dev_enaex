###########################################
# Recdata SPA 2 Ago 2020
# Vista camaras pruebas
# programa para camaras fijas
###########################################

import cv2
import pafy
import numpy as np
import socket
import threading
import time
from struct import *
import random
from timeit import default_timer as timer
import queue
from common import anorm2, draw_str, bb_iou
from centroidtracker import CentroidTracker
import math
import scipy.stats
from scipy.spatial import distance as dist
from scipy import stats
import matplotlib.path as mpltPath

drk_q = queue.LifoQueue(maxsize=1500)
sensib_q = queue.Queue(maxsize=1500)
isort_q  = queue.LifoQueue(maxsize=250)
input_q = queue.Queue(maxsize=100) # cola para el retraso

######SOCKETS DEF###############
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
        data = cv2.imencode('.jpg', frame)[1].tostring()
            # if sensib_q.empty() != True:
        sensib=sensib_q.get()
        sensib_q.task_done()
        sensib_q.queue.clear()
        #except:
         
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
                #r = client_socket.recv(90456) 
                r = client_socket.recv(1024)
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
            #print("tiempo de Recepcion %.3f" %(timer()-inicio2))
        isort = []
        isort = np.fromstring(data, np.int32)
        isort = np.reshape(isort, (-1, 6))
        if isort.sum() > 1:
            isort=isort.astype('int')
            isort_q.put(isort)
    client_socket.shutdown(socket.SHUT_RDWR)
    client_socket.close()
##### FIN SOCKETS DEF###############

#####################Backgroud detector ######################################
def filter_mask( img, a=None):
    '''
    Filtro para tratar el background substration
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)) #(2, 2)
    # llena los agujeros pequeos
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Remueve el ruido
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # Dilata los blobs adjacentes
    dilation = cv2.dilate(opening, kernel, iterations=1  )
    return dilation

def detect_mov(fg_mask):
    min_contour_width=30
    min_contour_height=30
    max_contour_width=250
    max_contour_height=200
    matches = []
    centroide = []
    # Encuentra los contornos externos
    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    for (i, contour) in enumerate(contours):
        #cv.polylines(frame_gray,[contour],True,(0, 0, 10), 1) # background sub
        #cv2.drawContours(frame_gray, [contour], 0, (0,0,0), 1)
        M = cv2.moments(contour)
        # calculate x,y coordinate of center
        if M["m00"] <=0:
            M["m00"] = 1
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= min_contour_width) and (w <= max_contour_width) and (
            h <= max_contour_height) and (h >= min_contour_height) #and (h/w > 2.7)

        if contour_valid:
            matches.append((x, y, x+w, y+h, 81, 0))
            centroide.append((cX, cY))
    centroide=np.int32(centroide)
    matches=np.int32(matches)
    return matches, centroide

################## Backgroud detector #################################

def centroides(x,y,a,b):
    cx = x + int((x-a)/2)
    cy = y + int((y-b)/2)
    return cx,cy 


############ SETTINGS ################################################
youtube = False
if youtube:
    #url = "https://www.youtube.com/watch?v=nPA_J80hBOg"
    #url = "https://www.youtube.com/watch?v=Fz1uPoAH-4E"
    #url = "https://www.youtube.com/watch?v=cdwr8hh58pk"
    #url = "https://www.youtube.com/watch?v=xqpUV69323Q&list=TLPQMjUwNzIwMjAh3GwG76uo9A&index=2"
    url = "https://www.youtube.com/watch?v=zR1Y8MjdbYI"
    video = pafy.new(url)
    best = video.getbest(preftype="mp4") 
    capture = cv2.VideoCapture(best.url)
    xcr, ycr, hcr, wcr = 170, 145, 416, 416
else:
    fuente = 'videos/otros/bellavista9am.mp4'
    #fuente = 'videos/rot.mp4'
    fuente = 'videos/bici4pm.mp4'
    #fuente = 'videos/2019-05-07_07:50:41_0.avi.mp4'
    capture = cv2.VideoCapture(fuente)
    xcr, ycr, hcr, wcr = 0, 0, 416, 416




ret, img = capture.read()
scale_percent = 100 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

frame_idx = 0
detect_interval =  4 #7 ideal rotonda
retardo = 7 #7 Var retardo del flujo de video para que la GPU tenga tiempo de procesar la imagen actual
x_scale, y_scale = 1, 1
sens = 60
fps = capture.get(cv2.CAP_PROP_FPS)
frame_rate = fps
prev = 0

b_ground_sub = True
bgshow = False
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold = 20, detectShadows=True) #detecta mov de la camara
winbGround = 'background subtraction'
ct = CentroidTracker()
ct_yolo = CentroidTracker()
objects_yolo =None
abajo = 0
arriba = 0
contados = []
id0 = 0

##################### OPTICAL FLOW ####################################
color = {1:(255,255,255),2:(255,0,0),3:(0,255,0),0:(0,0,255),5:(255,0,255),6:(255,255,0),7:(0,255,255),8:(100,200,0),4:(0,0,0),9:(20,255,100)}
matriz = np.array([0,0,0,0,id0,0]).reshape(1,6)
tracks = []
track_len = 10
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict( maxCorners = 190, #19
                       qualityLevel = 0.3,
                       minDistance = 4, #7
                       blockSize = 7 )

index = np.array([])
index = index.reshape(-1,1)
tiempo = np.array([])
tiempo = tiempo.reshape(-1,1)
def optical_flow(prev_gray, frame_gray, lk_params, tracks, vis, index, tiempo):
    img0, img1 = prev_gray, frame_gray
    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    good = d < 1
    new_tracks = []
    new_index = []
    reg = []
    new_tiempo = []
    for tr, (x, y), good_flag, good_index, gtime in zip(tracks, p1.reshape(-1, 2), good, index, tiempo):
        if not good_flag or timer()-gtime > 2:
            continue
        tr.append((x, y))
        if len(tr) > track_len:
            del tr[0]
        new_tracks.append(tr)
        new_index.append(good_index)
        new_tiempo.append(gtime)
        lastdigit = int(repr(int(good_index[0]))[-1])
        cv2.circle(vis, (x, y), 2, color.get(lastdigit), -1)
        txt = int(good_index[0])
        reg.append([x,y,txt])
        #if txt:
        #    cv2.putText(vis, str(txt), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1) 
    tracks = new_tracks
    index =  np.float32([ind[-1] for ind in new_index]).reshape(-1, 1)
    tiempo = np.float32([ind[-1] for ind in new_tiempo]).reshape(-1, 1)
    reg = np.array(reg).astype(int)
    #cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
    draw_str(vis, (20, 20), 'track count: %d' % len(tracks))
    sup = .93 # nivel de confianza
    inf = 1 - sup
    for u in np.unique(index):
        data = reg[reg[:,2]==u,...]
        x1=int(data[:,0].mean())
        y1=int(data[:,1].mean())
        #if int(u):
            #cv2.putText(vis, str(int(u)), (x1, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        ptos = data[:,[0,1]]
        ptos = np.int32(ptos)
        prom_x = ptos[:,0].mean()
        sd_x = ptos[:,0].std()
        sup_x=scipy.stats.norm.ppf(sup,prom_x,sd_x)
        inf_x=scipy.stats.norm.ppf(inf,prom_x,sd_x)
        prom_y = ptos[:,1].mean()
        sd_y = ptos[:,1].std()
        sup_y=scipy.stats.norm.ppf(sup,prom_y,sd_y)
        inf_y=scipy.stats.norm.ppf(inf,prom_y,sd_y)
        if inf_x >0 and sup_x>0 and inf_y>0 and sup_y>0:
            ptos = ptos[ptos[:,0] < sup_x]
            ptos = ptos[ptos[:,0] > inf_x]
            ptos = ptos[ptos[:,1] < sup_y]
            ptos = ptos[ptos[:,1] > inf_y]
        x_,y_,w_,h_ = cv2.boundingRect(ptos) # reodea con un rectangulo una nuve de puntos
        text = str(int(u))
        if int(u):
            cv2.putText(vis, text, (x_+int(w_/2), y_+int(h_/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        #print(tiempo.shape, tiempo)
    return  vis, tracks, index, tiempo


###################### Optical Flow #####################################

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
    print ("fallo la creacin del socket con error %s" %(err))

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

while(ret):
    time_elapsed = time.time() - prev
    # Capture frame-by-frame
    
    if time_elapsed > 1./frame_rate:
        ret, frame = capture.read()

        #frame= cv2.resize(frame, (516,416) )
        input_q.put(frame) # cola de buffer para el retardo


        crop_img = frame[ycr:ycr+hcr, xcr:xcr+wcr]
#         frame_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
#         vis = crop_img
# ################ Optical Flow ########################################
#         if len(tracks) > 0:
#             pass
#             vis, tracks,index, tiempo = optical_flow(prev_gray, frame_gray, lk_params, tracks, vis,index, tiempo)
# ################ Optical Flow ########################################

        if frame_idx % detect_interval == 0:
            drk_q.put(crop_img) # alimenta la cola del detector
            #crop_img = vis
            sensib_q.put(sens) # Sensibilidad de deteccin o nivel de confianza 
            # mask = np.zeros_like(frame_gray)
            # mask[:] = 255
            # for x, y in [np.int32(tr[-1]) for tr in tracks]:
            #     cv2.circle(mask, (x, y), 5, 0, -1)
            # p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
            # if p is not None:
            #     id_=np.zeros((p.shape[0], 1))
            #     id_.fill(0)
            #     ti_=np.zeros((p.shape[0], 1))
            #     for x, y in np.float32(p).reshape(-1, 2):
            #         tracks.append([(x, y)])
            #         #index.append(0)
            #     index = np.append(index,id_, axis=0)
            #     ti_.fill(timer())
            #     tiempo = np.append(tiempo,ti_, axis=0)


        if frame_idx > retardo:
            frame = input_q.get()
        crop_img = frame[ycr:ycr+hcr, xcr:xcr+wcr]

        frame_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        vis = crop_img
################ Optical Flow ########################################
        if len(tracks) > 0:
            pass
            vis, tracks,index, tiempo = optical_flow(prev_gray, frame_gray, lk_params, tracks, vis,index, tiempo)
################ Optical Flow ########################################
        if frame_idx % detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
            if p is not None:
                id_=np.zeros((p.shape[0], 1))
                id_.fill(0)
                ti_=np.zeros((p.shape[0], 1))
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])
                    #index.append(0)
                index = np.append(index,id_, axis=0)
                ti_.fill(timer())
                tiempo = np.append(tiempo,ti_, axis=0)

        prev = time.time()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

        ############background sub #######################################
        #frame_msk = crop_img
        frame_msk = frame_gray
        if b_ground_sub:
            draw_str(frame, (350, 27), 'BG Sub')
            fg_mask = bg_subtractor.apply(frame_msk, None, 0.001) # back ground detector
            fg_mask = filter_mask(fg_mask) # back ground detector
            bg_detect,centroide = detect_mov(fg_mask)  # back ground detector
            rects = []
            for e in bg_detect:
                rects.append(e[0:4].astype("int"))
                xb,yb,ab,bb = e[0],e[1],e[2],e[3]
                cv2.rectangle(fg_mask, (xb, yb), (ab, bb) , (255, 255, 255), 1)
            objects= ct.update(rects,centroide)
            for (objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)
                #cv2.putText(crop_img, text, (centroid[0] - 10, centroid[1] - 10),
                #    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.circle(fg_mask, (centroid[0], centroid[1]), 4, (2, 255, 255), -1)
        else:
            bg_detect = np.array([])
        ############End background sub #######################################

        if isort_q.empty() != True:
            i_sort =isort_q.get()
            isort_q.task_done()
            isort_q.queue.clear()
            centroid_yolo = []
            rects_yolo = []
            for e in i_sort:
                xs,ys,a,b = e[0],e[1],e[2],e[3]
                cv2.rectangle(crop_img, (xs, ys), (a, b) , (255, 0, 255), 1) # TODO recomponer para la pintana
                xsc, ysc = xs, ys
                centroid_yolo.append((xsc,ysc))
                rects_yolo.append(e[0:4].astype("int"))
            objects_yolo= ct_yolo.update(rects_yolo, centroid_yolo)

            puntos = np.int32([tr[-1] for tr in tracks]).reshape(-1,  2)
            for e in i_sort:
                xs,ys,a,b,clas = e[0],e[1],e[2],e[3],e[4]
                roi=mpltPath.Path([(xs-1,ys-1),(xs-1,b+1),(a+1,b+1),(a+1,ys-1)])
                pts = np.array([[xs-1,ys-1],[xs-1,b+1],[a+1,b+1],[a+1,ys-1],[xs-1,ys-1]])
                pts = pts.reshape((-1, 1, 2))
                image = cv2.polylines(crop_img, [pts], True, (255,255,255), 1)
                contiene = roi.contains_points(puntos)
                a = index[contiene]
                b = a[a[:,0]>0]
                moda,cont=stats.mode(a)
                modab,contb=stats.mode(b)
                print('modab', type(modab),modab.shape,modab)
                if moda.shape[0]>0:
                    moda,cont=int(moda[0][0]),int(cont[0][0])
                    print("SI")
                    if modab.shape[0]:
                        modab=int(modab[0][0])
                        print('pre',a,modab)
                        if modab>0:
                            moda = modab
                    m0 = moda
                    print('moda',moda)
                    if moda == 0:
                        id0 +=1
                        moda = id0
                    for i,c in enumerate(contiene):
                        #print(i,c)
                        if c:
                            index[i,0]=moda #si el punto está en contiene "True" le asigna la moda
                            tiempo[i,0]=timer()
                        else:
                            val = int(index[i][0])
                            if val == m0 and val !=0: 
                                print("falso",i,val, m0)
                                index[i,0]=0
                else:
                    print("NO")
                #print('contiene',a,moda ) 
                a = index[contiene]
                print('post',a)


        else:
            i_sort = np.array([])
        if objects_yolo !=None:
            lista_id = []
            par_cent = []
            for (objectID_y, centroid_y) in objects_yolo.items():
                text = "ID {}".format(objectID_y)
                #cv2.putText(crop_img, text, (centroid_y[0], centroid_y[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1) 
                lista_id.append(objectID_y)
                par_cent.append(centroid_y)
            par_cent = np.array(par_cent)
            #print(lista_id,par_cent)
            if i_sort.shape[0]:
                xs=i_sort[:,[0,2]].mean(axis=1)
                ys=i_sort[:,[1,3]].mean(axis=1)
                xs=xs.reshape(xs.shape[0],1)
                ys=ys.reshape(ys.shape[0],1)
                cent_sort=np.append(xs,ys,axis=1).astype(int)
                D = dist.cdist(par_cent, cent_sort)
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                for (row, col) in zip(rows, cols):
                    pass
                    #print(row,col,'\n', matriz.astype(int))
        cv2.rectangle(crop_img, (0, 0), (hcr-1, wcr-1) , (255, 0, 255), 1) 
        backtorgb[ycr:ycr+crop_img.shape[0], xcr:xcr+crop_img.shape[1]] = crop_img
        resized = cv2.resize(backtorgb, dim, interpolation = cv2.INTER_AREA)
        # Display the resulting frame
        draw_str(resized, (10, 50), "time_elapsed: "+str(time_elapsed))
        cv2.imshow('frame',resized)

        if bgshow and b_ground_sub:
            cv2.imshow(winbGround,fg_mask)

        ch = cv2.waitKey(1)
        if ch & 0xFF == ord('q'):
            break
        if ch == ord('n'): 
            if bgshow:
                bgshow = False
                cv2.destroyWindow(winbGround)
            else:
                bgshow = True
        if ch == ord('b'): 
            if b_ground_sub:
                b_ground_sub = False
            else:
                b_ground_sub = True

        frame_idx += 1
        prev_gray = frame_gray

# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()