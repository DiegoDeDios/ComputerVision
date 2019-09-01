import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import glob

def FourierTransformImages():
    for i in range(0,720):
        img = cv2.imread('./NormalVideo/%s.jpg' % (str(i)),0)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        cv2.imwrite('./FourierVideo/%s.jpg' % (str(i)), magnitude_spectrum)

def splitVideo(filename):
	cap = cv2.VideoCapture(filename)
	if(cap.isOpened() == False):
		print("Error, chale")
	frameNumber = 0
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			print(frame)
			writeFrames(frame, frameNumber)
		else:
			break
		frameNumber+=1
	cap.release()
	cv2.destroyAllWindows()

def writeVideo():
    img_array = []
    for filename in glob.glob('./FourierVideo/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


    out = cv2.VideoWriter('fourier.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()