import pnet
#import darknet as pnet #yolo4
import cv2
import numpy as np
from timeit import default_timer as timer


class Drk:
	def __init__(self):
		self.isort = []
		self.netMain = None
		self.metaMain = None
		self.configPath = "net.cfg"
		self.weightPath = "net.wei"
		#self.configPath = "yolo4/yolov4.cfg" # yolo4
		#self.weightPath = "yolo4/yolov4.weights" # yolo4
		self.configPath = "608/yolov3.cfg" # yolo3 608
		self.weightPath = "608/yolov3.weights" # yolo3 608
		self.metaPath = "cat.dat"
		if self.netMain is None:
			self.netMain = pnet.load_net_custom(self.configPath.encode(
			"ascii"), self.weightPath.encode("ascii"), 0, 1)  # batch size = 1
		if self.metaMain is None:
			self.metaMain = pnet.load_meta(self.metaPath.encode("ascii"))
		self.pnet_image = pnet.make_image(pnet.network_width(self.netMain),
			pnet.network_height(self.netMain),3)

	def dark(self, frame, thresh=0.75):
		clases = [0,2,3,4,5,6,7,8,9,10] #clases a seguir
		
		def convertBack(x, y, w, h):
			xmin = int(round(x - (w / 2))) 
			xmax = int(round(x + (w / 2))) 
			ymin = int(round(y - (h / 2))) 
			ymax = int(round(y + (h / 2)))
			return xmin, ymin, xmax, ymax

		def Boxes(detections):
			det = []
			for detection in detections:
				x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
				xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
				prob = round(float(detection[1]),2)*100
				if int(detection[0].decode()) in clases:
					det.append([ xmin, ymin, xmax, ymax ,detection[0].decode(), prob])
			return det

		inicio=timer()
		self.isort = []
		if frame is not []:
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame_resized = cv2.resize(frame_rgb,
			                           (pnet.network_width(self.netMain),
			                            pnet.network_height(self.netMain)),
			                           interpolation=cv2.INTER_LINEAR)
			pnet.copy_image_from_bytes(self.pnet_image,frame_resized.tobytes())
			detections = pnet.detect_image(self.netMain, self.metaMain, self.pnet_image, thresh)
			#print(frame.shape, frame_resized.shape, "frame.shape, frame_resized" )
			det = Boxes(detections)
			self.isort=np.int32(det)  
			
			print("Deteccion en %.3f s y thresh_ %.2f" % (timer()-inicio, thresh) )
		return self.isort
