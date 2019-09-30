import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys


def startStream(selection):
	cap = cv2.VideoCapture(0)
	while(True):
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
		if selection == 1: #Canny
			cv2.imshow('frame',cv2.Canny(frame,100,200))
		if selection == 2: #sobel
			img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
			img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
			img_sobel = img_sobelx + img_sobely
			cv2.imshow('frame', img_sobel)
		if selection == 3: #prewitt
			kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
			kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
			img_prewittx = cv2.filter2D(img_gaussian, -1, kernely)
			cv2.imshow('frame', img_prewittx)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

startStream(int(sys.argv[1]))

