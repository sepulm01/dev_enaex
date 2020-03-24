#!/usr/bin/env python

'''
Recdata SPA.
Feb - 2019
Grabador de video para Raspberry con sensor de movimiento usando optical flow.
Uso vid3.py #camara(0 o 1) nombre(archivo video)
'''
import numpy as np
import cv2
import datetime
from datetime import timedelta
import time
import sys

time.sleep(1) # retraso para raspberry pi en startup
#break
def draw_flow(img, flow, step=10):
    #print(flow, flow.shape,'flow.shape')
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    movimiento = False
    if lines.sum() > 294090:
        movimiento = True

    cv2.polylines(vis, lines, 0, (0, 0, 255)) #(XXX, 2, 2) lines.shape donde XXX es el tamaño de la img
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (255, 0, 0), -1)

    return vis, movimiento

camara, nombre = int(sys.argv[1]), sys.argv[2]
direc = "/home/martin/Videos/"
archivo = direc + datetime.datetime.now().strftime("%Y-%m-%d@%H:%M") + nombre + ".avi"

print(archivo)


cap = cv2.VideoCapture(camara)
ret, prev = cap.read()

xn, yn = 640, 480 
prev = cv2.resize(prev,(xn,yn)) 
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

fourcc = cv2.VideoWriter_fourcc(*'XVID') #XVID X264
out = cv2.VideoWriter(archivo,fourcc, 20, (640,480)) #fps 60.0

unratito = datetime.datetime.now()
ayer = int(unratito.strftime("%H"))

while(cap.isOpened()):
    now = datetime.datetime.now()
    hoy = int(now.strftime("%H"))

    ret, frame = cap.read()

    gray = frame
    gray = cv2.resize(frame,(xn,yn)) 
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray
    

    if ret==True:

        (frame,0)
        img, mov = draw_flow(gray, flow)
        cv2.imshow("Frame", img)


        if mov is True:
            unratito =now +timedelta(seconds=30) #tiempo de timeout para grabar

        cv2.putText(frame, now.strftime("%Y-%m-%d %H:%M:%S") , (10,20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255))
        if now < unratito:

            out.write(frame)
        else:
            if hoy > ayer:
                print('cambio de día \n'*5)
                archivo =  direc + now.strftime("%Y-%m-%d@%H:%M") + nombre + ".avi"
                out = cv2.VideoWriter(archivo,fourcc, 30, (640,480)) #fps 60.0
                ayer = hoy

        #out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Libera las ventanas
cap.release()
out.release()
cv2.destroyAllWindows()