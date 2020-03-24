
from __future__ import print_function
from __future__ import division
import numpy as np
import cv2 as cv
from centroidtracker import CentroidTracker
import darknet
from drk import Drk
import queue
import threading
import time
from timeit import default_timer as timer
from common import anorm2, draw_str
import multiprocessing as mul
import socket
import random
from struct import *
from multiprocessing import Process, Queue, Pipe
import numpy as np
import queue

def f(q,fuente, conn):
    #q.put([42, None, 'hello'])
    #fuente =  0 #'6_35.mp4'
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
    #####################Backgroud detector ######################################

    cap = cv.VideoCapture(fuente)
    b_ground_sub = True
    bgshow = True
    bg_subtractor = cv.createBackgroundSubtractorMOG2(history=400, varThreshold = 20, detectShadows=True) #detecta mov de la camara
    winbGround = 'background subtraction'
    ct = CentroidTracker()

    while(cap.isOpened()):
        _ret, frame = cap.read()
        
        if _ret == False:
            print("termino flujo de video (ret==0).")
            break #TODO sacar en produccion
            while _ret == False:
                time.sleep(20)
                cap = cv.VideoCapture(fuente)
                _ret, frame = cap.read()

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
        if len(bg_detect) > 0:
            conn.send(frame)
            isort=conn.recv()
            for e in isort:
                x,y,a,b = e[0],e[1],e[2],e[3]
                cv.rectangle(frame, (x, y), (a, b) , (255, 255, 255), 1)
        #print("bg_detect", len(bg_detect) )
        if bgshow and b_ground_sub:
            cv.imshow(winbGround,fg_mask)
        ############fin background sub #######################################
        
        #q.put(frame)
        #print(q.qsize(), "desde el proces")


        draw_str(frame, (10, 15), 'Streetflow.cl')
        cv.imshow("Frame", frame)

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
        # if ch == ord('g'):
        #     if gammashow:
        #         gammashow = False
        #     else:
        #         gammashow = True
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

    cv.destroyAllWindows()
    #q.put("fin")

    conn.send("fin")
    conn.close()



if __name__ == '__main__':

######################### Darknet ###################################################
    netMain = None
    metaMain = None
    configPath = "./cfg/yolov3.cfg"
    #configPath = "./cfg/yolov3-8clases.cfg"
    weightPath = "./cfg/yolov3.weights" 
    #weightPath = "./data/yolov3-uoct_final.weights"
    #weightPath = "./data/yolov3-uoct_last.weights" 
    #weightPath = "./data/yolov3-8clases_16000.weights" 
    metaPath = "./cfg/coco.data"
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    def dark(frame, thresh=0.25):
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
        print("drk detect",timer()-antes, isort)
        return isort




    fuente = '6_35.mp4' #2
    parent_conn, child_conn = Pipe()
    
    q = Queue()
    p = Process(target=f, args=(q,fuente, child_conn))
    p.start()
    print(p.is_alive())
    while p.is_alive():
        #frame = q.get()
        #print(q.qsize(), "q.qsize")
        frame= parent_conn.recv()
        if frame == "fin":
            print("fin")
            p.join()
            break
        isort=dark(frame)    # prints "[42, None, 'hello']"
        parent_conn.send(isort)
    p.join()
