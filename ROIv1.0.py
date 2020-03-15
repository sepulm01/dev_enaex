#!/usr/bin/env python

'''
Recdata SPA.
Marz - 2019
Programa de definición de areas (ROI) para StreetFlow.
Version de prueba para la UOCT para camaras fijas.

'''
import cv2
import numpy as np
import csv
import os

cajas=[]
areas =[]
selection_in_progress = True; # support interactive region selection
boxes = np.array([], np.int32)
current_mouse_position = np.ones(2, dtype=np.int32);
vidcap = cv2.VideoCapture('/home/martin/Videos/videos_uoct/rotonda.mp4') 
#vidcap.set(1,59000)
success,frame = vidcap.read()
dim_y, dim_x = 416,416
frame = cv2.resize(frame,(int(dim_x),int(dim_y))) # disminuye la imagen

def onmouse(event,x,y,s,p):
    global boxes;
    global selection_in_progress;
    global areas

    current_mouse_position[0] = x;
    current_mouse_position[1] = y;

    if event == cv2.EVENT_LBUTTONDOWN:
        #boxes = [];
        sbox = [x, y];
        cv2.line(frame,(x,y),(x+1,y+1),(0,255,255),2)
        selection_in_progress = True;
        cajas.append(sbox);
        boxes = np.array([cajas], np.int32)
        if boxes.size > 3 and (boxes[0][0][0]-10) < sbox[0] < (boxes[0][0][0]+10) and (boxes[0][0][1]-10) < sbox[1] < (boxes[0][0][1]+10):
            areas.append(boxes)
            selection_in_progress = False

def linea(boxes):
    cv2.polylines(frame,[boxes],False,(0,255,255),2)

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def reset():
    global frame
    global areas
    success,frame = vidcap.read()
    frame = cv2.resize(frame,(int(dim_x),int(dim_y))) # disminuye la imagen
    #frame = image
    areas = []
    boxes = []
    boxes = np.array([], np.int32)

def guarda():
    #np.savetxt("foo.csv", boxes, delimiter=",")
    with open('areas.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, delimiter=',', quoting=csv.QUOTE_NONE)
        save = []
        for i in range(0, len(areas)):
            linea = []
            for u in range(0,len(areas[i])):
                for s in range(0,len(areas[i][u])):
                    for o in range(0,len(areas[i][u][s])):
                        linea.append(areas[i][u][s][o])
                        
            wr.writerow(linea)
        print('Exportadas',len(areas),'areas')
        cv2.imwrite('frame.jpg',frame)

def mapa_areas():#carga areas desde el archivo areas.csv
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
        areas.append([asd])
        #hace mapa de las áreas
    dibujo = []
    for z in range(0,len(areas)):
        cv2.polylines(frame,areas[z],True,(0,255,255),2)
        cv2.putText(frame, str(z), totuple((areas[z])[0:1][0][0]), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0),2)
        #dibujo.append(areas[z].tolist())

    return areas

winName = 'StreetFlow- Def areas'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(winName,onmouse);

if os.path.exists('areas.csv'):
    areas= mapa_areas()
    print(areas[0][0].shape)

while True:
    if (selection_in_progress):
        linea(boxes)
    else:
        linea(boxes)
        boxes = []
        boxes = np.array([], np.int32)
        cajas = []
        for z in range(0,len(areas)):
            cv2.polylines(frame,areas[z],True,(0,255,255),2)
            cv2.putText(frame, str(z), totuple((areas[z])[0:1][0][0]), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0),2)
    #print(cajas, areas)
        
    #paint()
   # print(frame.shape)
    cv2.imshow(winName,frame)
    k = cv2.waitKey(30) &0xFF
    if k == ord('q'): break
    if k == ord('z'): reset()
    if k == ord('s'): guarda()
