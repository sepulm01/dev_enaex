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
#from centroidtracker import CentroidTracker
import math
import scipy.stats
from scipy.spatial import distance as dist
#from scipy import stats
import matplotlib.path as mpltPath
import os
from shapely.geometry import box
from shapely.geometry import MultiPoint
from shapely.geometry import asMultiPoint

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
    sensib=55
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

########################## Correccion GAMMA ##########################################
def gammaCorrection(frame):
    '''
    Función de corrección de gamma aplicado a la imagen
    '''
    img_original=frame
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img_original, lookUpTable)
    return res

def on_gamma_correction_trackbar(valor):
    global gamma
    gamma = valor / 100
    #print(gamma,'gamma')

##########################FIN Correccion GAMMA ##########################################

#####################Backgroud detector ######################################
def filter_mask( img, a=None):
    '''
    Filtro para tratar el background substration
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) #(2, 2)
    # llena los agujeros pequeos
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Remueve el ruido
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # Dilata los blobs adjacentes
    dilation = cv2.dilate(opening, kernel, iterations=4  )
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
        #cv2.polylines(frame_gray,[contour],True,(0, 0, 10), 1) # background sub
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

def centros(matriz):
    ''' calcula los centroides de la matriz que ingresa'''
    xs=matriz[:,[0,2]].mean(axis=1)
    ys=matriz[:,[1,3]].mean(axis=1)
    xs=xs.reshape(xs.shape[0],1)
    ys=ys.reshape(ys.shape[0],1)
    return np.append(xs,ys,axis=1).astype(int)

def area_mtx(a):
    wh=np.subtract(a[:,[2,3]],a[:,[0,1]])
    return wh[:,[0]]*wh[:,[1]]
    
def i_sort_check(opt,ord=1):
    '''Toma la matriz de la moda y num de los resultados de contiene y la ordena por cantidad de puntos que contiene el id solicitado'''
    
    a=list(opt[:,1])
    ide = 0
    indice = 0
    index_sets = [np.argwhere(i==a) for i in np.unique(a)]
    nueva = np.zeros((1,3))
    prima = np.zeros((1,3))
    lista = []
    for i in index_sets:  
        if i.shape[0]<2:
            nueva = np.append(nueva, opt[i,[0,1,2]],axis=0)
            continue
        if ord==-1:
            mayor = opt.max()
        else:
            mayor = 0
        for ii in i:
            prima = np.append(prima, [[ii[0],0,0]],axis=0)
            if ord==-1:
                if mayor > opt[:,2][ii]:
                    mayor = opt[:,2][ii]
                    ide =  opt[:,1][ii]
                    indice = ii 
            else:
                if mayor < opt[:,2][ii]:
                    mayor = opt[:,2][ii]
                    ide =  opt[:,1][ii]
                    indice = ii
        lista.append((indice, ide))
    prima=np.delete(prima,0,0)
    nueva=np.delete(nueva,0,0)
    nueva = np.append(nueva,prima, axis=0).astype('int')
    nueva=np.sort(nueva.view('i8,i8,i8'), order=['f0'], axis=0).view(np.int)
    for i in lista:
        nueva[i[0],1]=i[1]
        nueva[i[0],2]=opt[i[0],2]
    return nueva

def hist_corr(src_base,src_test1):
    ''' devuelve la correlación entre los histogramas de las dos imágenes, 1 es, perfecto 0 es malo'''
    hsv_base = cv2.cvtColor(src_base, cv2.COLOR_BGR2HSV)
    hsv_test1 = cv2.cvtColor(src_test1, cv2.COLOR_BGR2HSV)
    hsv_half_down = hsv_base[hsv_base.shape[0]//2:,:]
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges # concat lists
    # Use the 0-th and 1-st channels
    channels = [0, 1]
    hist_base = cv2.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    hist_half_down = cv.calcHist([hsv_half_down], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_half_down, hist_half_down, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    hist_test1 = cv.calcHist([hsv_test1], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    base_half = cv.compareHist(hist_base, hist_half_down, 0)
    base_test1 = cv.compareHist(hist_base, hist_test1, 0)
    if base_half<base_test1:
        return True
    else:
        return False

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


############ SETTINGS ################################################
youtube = False
if youtube:
    #url = "https://www.youtube.com/watch?v=nPA_J80hBOg"
    #url = "https://www.youtube.com/watch?v=Fz1uPoAH-4E"
    #url = "https://www.youtube.com/watch?v=cdwr8hh58pk"
    #url = "https://www.youtube.com/watch?v=xqpUV69323Q&list=TLPQMjUwNzIwMjAh3GwG76uo9A&index=2"
    url = "https://youtu.be/XpYdDF8U4N8"
    video = pafy.new(url)
    best = video.getbest(preftype="mp4") 
    capture = cv2.VideoCapture(best.url)
    xcr, ycr, hcr, wcr = 1280, 580, 416, 416
else:
    fuente = 'videos/otros/bellavista9am.mp4'
    fuente = 'videos/rot.mp4'
    #fuente = 'videos/rot_bus.mp4'
    #fuente = 'videos/bici4pm.mp4' 
    #fuente = 'videos/2019-04-28_15:15:37_0.avi.mp4'
    #fuente = 'videos/bici8am.mp4'
    #fuente = 'videos/2019-05-07_07:50:41_0.avi.mp4'
    #fuente = 'videos/2019-05-06_18:50:40_0.avi.mp4'
    #fuente = 'videos/oclusion.mp4'
    fuente = 'videos/pza_noche.mp4'
    capture = cv2.VideoCapture(fuente)
    xcr, ycr, hcr, wcr = 300,300, 416, 416 # rot
    #xcr, ycr, hcr, wcr = 150, 50, 320, 320
#capture.set(1,2800)
#capture.set(1,1600)

cv2.namedWindow('Streetflow.cl',cv2.WINDOW_AUTOSIZE)
gamma = 0.20
gamma_max = 200
gammashow = False
gamma_init = int(gamma * 100)
cv2.createTrackbar('Ajuste Gamma', 'Streetflow.cl', gamma_init, gamma_max, on_gamma_correction_trackbar)


ret, img = capture.read()
print(ret,'ret', img.shape)
scale_percent = 80 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

frame_idx = 0
detect_interval =  7 #7 ideal rotonda
retardo = 8 #9 Var retardo del flujo de video para que la GPU tenga tiempo de procesar la imagen actual
x_scale, y_scale = 1, 1
sens = 65
fps = capture.get(cv2.CAP_PROP_FPS)
frame_rate = fps
prev = 0

b_ground_sub = True
bgshow = False
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold = 155, detectShadows=False) #detecta mov de la camara
winbGround = 'background subtraction'
#ct = CentroidTracker()
#ct_yolo = CentroidTracker()
objects_yolo =None
abajo = 0
arriba = 0
contados = []
id0 = 0
mtrx_log = np.zeros((1,8))
mtrx_id = 0
obj_img = {}

dic_ar_ant = {}




##################### OPTICAL FLOW ####################################
color = {1:(255,255,255),2:(255,0,0),3:(0,255,0),0:(0,0,255),5:(255,0,255),6:(255,255,0),7:(0,255,255),8:(100,200,0),4:(0,0,0),9:(20,255,100)}
matriz = np.array([0,0,0,0,id0,0]).reshape(1,6)
tracks = []
track_len = 10
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict( maxCorners = 90, #19
                       qualityLevel = 0.3,
                       minDistance = 4, #7
                       blockSize = 4 ) #importante a menos valor más puntos. 4 probado ok

index = np.array([])
index = index.reshape(-1,1)
tiempo = np.array([])
tiempo = tiempo.reshape(-1,1)
def optical_flow(prev_gray, frame_gray, lk_params, tracks, vis, index, tiempo):
    # for u in np.unique(index):
    #     print('previos',u)
    img0, img1 = prev_gray, frame_gray
    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
    t_calcOpticalFlowPyrLK = timer()
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    print(timer()-t_calcOpticalFlowPyrLK,'t_calcOpticalFlowPyrLK')
    t_asig = timer()
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    good = d < 1
    new_tracks = []
    new_index = []
    reg = []
    new_tiempo = []
    ww = 0
    for tr, (x, y), good_flag, good_index, gtime in zip(tracks, p1.reshape(-1, 2), good, index, tiempo):
        if not good_flag or timer()-gtime > 2: # 2 seg 
        #if not good_flag:
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
        reg.append([x,y,txt,ww])
        ww +=1
        #if txt:
        #    cv2.putText(vis, str(txt), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1) 
    tracks = new_tracks
    index =  np.float32([ind[-1] for ind in new_index]).reshape(-1, 1)
    tiempo = np.float32([ind[-1] for ind in new_tiempo]).reshape(-1, 1)
    reg = np.array(reg).astype(int)
    #cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
    #draw_str(vis, (20, 20), 'track count: %d' % len(tracks))
    #print(timer()-t_asig,'asignación tracks')
    t_index = timer()
    sup = .99 # nivel de confianza
    inf = 1 - sup
    filtros = []
    
    lhulk = []
    for u in np.unique(index):
        #print('Posteriores',u)
        if u:
            data = reg[reg[:,2]==u,...]
            ptos = data[:,[0,1]]
            ptos = np.int32(ptos)
            # prom_x = ptos[:,0].mean()
            # sd_x = ptos[:,0].std()
            # sup_x=scipy.stats.norm.ppf(sup,prom_x,sd_x)
            # inf_x=scipy.stats.norm.ppf(inf,prom_x,sd_x)
            # prom_y = ptos[:,1].mean()
            # sd_y = ptos[:,1].std()
            # sup_y=scipy.stats.norm.ppf(sup,prom_y,sd_y)
            # inf_y=scipy.stats.norm.ppf(inf,prom_y,sd_y)
            # if inf_x >0 and sup_x>0 and inf_y>0 and sup_y>0:
            #     xa,ya,xb,yb = int(inf_x), int(inf_y), int(sup_x), int(sup_y)
            #     roi = mpltPath.Path([(xa,ya),(xa,yb),(xb,yb),(xb,ya)])
            #     contiene = roi.contains_points(reg[:,[0,1]])
            #     filtros.append(contiene)
            text = str(int(u))
            hulk = asMultiPoint(ptos).convex_hull
            hulk.id = int(u)
            lhulk.append(hulk)
            x1, y1 = hulk.centroid.xy
            if int(u):
                cv2.putText(vis, text, (int(x1[0]), int(y1[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (25,255,2), 1)
            #hull = cv2.convexHull(ptos)
            #cv2.polylines(vis, [hull[::,0]], True, (255, 255, 0),1)
    # if frame_idx % 3 == 0:
    #     filtro = sum(filtros)
    #     if isinstance(filtro, int):
    #         pass
    #     else:
    #         for f,j in enumerate(filtro):
    #             if j!=False:
    #                 continue
    #             else:
    #                 index[f]=0
    #print(timer()-t_index,'index todo (varianzas y eso)')
    return  vis, tracks, index, tiempo, lhulk


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
timepo_max = 0
while(capture.isOpened()):
    inicio = timer()
    time_elapsed = time.time() - prev
    # Capture frame-by-frame
    if time_elapsed > 1./frame_rate:
        ret, frame = capture.read()

        if ret == False:
            print("termino flujo de video (ret==0). espera 5s")
            break #TODO sacar en produccion
            
        input_q.put(frame) # cola de buffer para el retardo
        prev = time.time()
        crop_img = frame[ycr:ycr+hcr, xcr:xcr+wcr]

        if frame_idx % detect_interval == 0:
            drk_q.put(crop_img) # alimenta la cola del detector
            sensib_q.put(sens) # Sensibilidad de deteccin o nivel de confianza 

        if frame_idx > retardo:
            frame = input_q.get()
        crop_img = frame[ycr:ycr+hcr, xcr:xcr+wcr]

        if gammashow:
            crop_img=gammaCorrection(crop_img) # Gamma Correction
            draw_str(crop_img, (350, 12), 'Gamma')

        frame_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        ## correccion Histograma y CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        frame_gray=cv2.equalizeHist(frame_gray)
        frame_gray = clahe.apply(frame_gray)
        #print(timer()-inicio,'pre opttime')
################ Optical Flow ########################################
        vis = crop_img
        if len(tracks) > 0:
            opttime = timer()
            vis, tracks,index, tiempo, lhulk = optical_flow(prev_gray, frame_gray, lk_params, tracks, vis,index, tiempo)
            #print(timer()-opttime,'opttime')
################ Optical Flow ########################################
        t_interval = timer()
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
                    if x > frame_gray.shape[1] or y > frame_gray.shape[0]:
                        print(x,y,'FALTA'*30)
                        print('frame_gray.shape',frame_gray.shape)
                index = np.append(index,id_, axis=0)
                ti_.fill(timer())
                tiempo = np.append(tiempo,ti_, axis=0)
            t_interval = timer() - t_interval
            print(t_interval,'t_interval')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

        ############background sub #######################################
        t_background = timer()
        frame_msk = frame_gray
        if b_ground_sub:
            draw_str(frame, (350, 27), 'BG Sub')
            fg_mask = bg_subtractor.apply(frame_msk, None, 0.001) # back ground detector
            fg_mask = filter_mask(fg_mask) # back ground detector
        else:
            bg_detect = np.array([])
        #print(timer()-t_background,'t_background')
        ############End background sub #######################################

        t_isort=timer()
        if isort_q.empty() != True:
            mtrx_id +=1
            i_sort =isort_q.get()
            isort_q.task_done()
            isort_q.queue.clear()
            # centroid_yolo = []
            # rects_yolo = []
            i_sort=np.sort(i_sort.view('i8,i8,i8,i8,i8,i8'), order=['f1'], axis=0).view(np.int) # ordena matriz (lejos ->cerca)
            
            #Recorre la mtrx de detect y en lista sort almacena los vertices sin oclusión (cajas)
            #para que luego revisar los puntos (index/tracks) contenidos en ellos
            cajas = {}
            i_sort_obj_img = {}
            for b,i in enumerate(i_sort):
                x1,y1,x2,y2, clase=i[:5]
                caja = box(x1,y1,x2,y2)
                caja.clase = clase
                cajas[b]= caja
                crop_det = crop_img[y1:y2, x1:x2]
                i_sort_obj_img[b]=crop_det

            lista_sort = []
            lista_f = []
            for i in cajas:
                if i:
                    f = cajas[i-1].difference(cajas[i])
                    if f.type == 'Polygon':
                        cont = list(f.exterior.coords)
                        if cont == []:
                            #print(f, cajas[i-1], cajas[i],'cajas[i-1], cajas[i]', i_sort)
                            lista_sort.append(list(cajas[i].exterior.coords))
                            lista_f.append(f)
                        else:
                            lista_sort.append(list(f.exterior.coords))
                            lista_f.append(f)

            lista_sort.append(list(cajas[list(cajas)[-1]].exterior.coords))
            lista_f.append(cajas[list(cajas)[-1]])
            
            liston = []
            desocupados = []
            for ni,cuad in enumerate(lista_f):
                grande = 0
                for hu in lhulk:
                    if cuad.intersects(hu):
                        aaaa = cuad.intersection(hu).area
                        if aaaa > grande:
                            grande = aaaa
                            element = (ni, hu.id)
                if grande:
                    liston.append(element)
                else:
                    desocupados.append(ni)
                    liston.append((ni,0))
            liston.sort()
            defi = []
            for l in liston:
                defi.append(l[1])

            for ni,cuad in enumerate(lista_f):
                if ni in desocupados:
                    grande = 100000
                    element = 0
                    for hu in lhulk:
                        if hu.id in defi:
                            continue
                        aaaa = cuad.distance(hu)
                        if aaaa < grande:
                            grande = aaaa
                            element = hu.id
                    if grande < 10:
                        defi[ni]=element
            #print('Defi:\n',defi)

            mtrx_ant = []
            for obj in np.unique(index).tolist():
                mt = mtrx_log[mtrx_log[:,6]==obj]
                mtrx_ant.append(mt[-1])
            mtrx_ant = np.array(mtrx_ant)

            #hist_corr(src_base,src_test1)

            #Identifica los centroides mas cercanos del frame anterior
            mtrx_tmp = np.append(i_sort,np.zeros((i_sort.shape[0],2)),axis=1)

            #Transforma los tracks en matriz de puntos
            puntos = np.int32([tr[-1] for tr in tracks]).reshape(-1,  2)

            # Actualiza el tiempo de los tracks si es que estan en fg_mask es decir que tienen movimiento
            for ti,(py,px) in enumerate(puntos):
                if px <= fg_mask.shape[1] and py <= fg_mask.shape[0]:
                    if fg_mask[py-1,px-1]: 
                        tiempo[ti,0]=timer()

            #Identifica que el indice (num) de los puntos de tracks que está en las areas de los objetos reconocidos (isort)
            # evalua la moda y se la asigna como infice a todos los puntos que estén dentro del roi
            d_contiene = {}
            #l_contiene = []
            for num,lis in enumerate(lista_sort):
                roi = mpltPath.Path(lis)
                contiene = roi.contains_points(puntos)
                d_contiene[num]=contiene
                xy = np.array(lis).astype('int')
                #cv2.polylines(crop_img,[xy],True,(255,255,255), 1)
                

            for i_num,ii in enumerate(defi):
                contiene=d_contiene[i_num]
                moda = int(ii)
                if ii == 0:
                    id0 +=1
                    moda = id0
                mtrx_tmp[i_num,6]=moda
                mtrx_tmp[i_num,7]= mtrx_id
                #obj_img[moda]=i_sort_obj_img[i_num] #almacena la ultima imagen asociada al id0

                for i,c in enumerate(contiene):
                    if c:
                        if index[i,0] !=0 and index[i,0] !=moda:
                            continue
                        else:
                            index[i,0]=moda #si el punto está en contiene "True" le asigna la moda
                            tiempo[i,0]=timer()
                    else:
                        val = int(index[i][0])
                        if val == moda and val !=0: #Expulsa a los puntos q no estan en roi
                            #print("falso",i,val)
                            index[i,0]=0

            mtrx_log = np.append(mtrx_log,mtrx_tmp,axis=0).astype("int")
            t_isort = timer() - t_isort 
            #print(t_isort,'t_isort')
        else:
            i_sort = np.array([])


        if bgshow and b_ground_sub:
            cv2.imshow(winbGround,fg_mask)

        # mascara que mezcla bg substration con la imagen
        # src2 = cv2.resize(fg_mask,crop_img.shape[1::-1])
        # src2 = cv2.cvtColor(src2,cv2.COLOR_GRAY2RGB)
        # dst = cv2.bitwise_and(src2,crop_img)
        # cv2.imshow("mask",dst)

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
        if ch == ord('g'):
            if gammashow:
                gammashow = False
            else:
                gammashow = True

        frame_idx += 1
        prev_gray = frame_gray

        cv2.rectangle(crop_img, (0, 0), (hcr-1, wcr-1) , (255, 0, 255), 1) 
        backtorgb[ycr:ycr+crop_img.shape[0], xcr:xcr+crop_img.shape[1]] = crop_img
        resized = cv2.resize(backtorgb, dim, interpolation = cv2.INTER_AREA)
        tiempo_ciclo = timer()-inicio
        #print(tiempo_ciclo,'tiempo_ciclo\n')
        if tiempo_ciclo > timepo_max:
            timepo_max = tiempo_ciclo
        if id0 % 100 == 0:
            timepo_max = 0
        # Display the resulting frame
        draw_str(resized, (10, 50), "Streetflow.cl")
        draw_str(resized, (10, 70), "time_elapsed: "+str("{:.4f}".format(time_elapsed)))
        draw_str(resized, (10, 90), "timer: "+str("{:.4f},{:.4f}".format(timepo_max, tiempo_ciclo)))
        #draw_str(resized, (10, 110), "id0: "+str("{:d}".format(id0)))
        #draw_str(resized, (10, 130), "fr detec: "+str("{:d}".format(detect_interval)))
        #draw_str(resized, (10, 150), "fr gray: "+str(frame_gray.shape)) 
        draw_str(resized, (10, 170), "frame_idx: "+str(capture.get(cv2.CAP_PROP_POS_FRAMES))) 
        cv2.imshow('Streetflow.cl',resized)

# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()
np.savetxt("mtrx_log.csv", mtrx_log.astype("int"), delimiter=",")
