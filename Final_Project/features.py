import numpy as np
import cv2
from matplotlib import pyplot as plt
MIN_MATCHES = 15 
def computeFeatures(img):
	orb = cv2.ORB_create()
	kp = orb.detect(img, None)
	kp, des = orb.compute(img, kp)
	img2 = cv2.drawKeypoints(img, kp, img, color=(0,255,0), flags=0)
	return img2

def startStream():
	cap = cv2.VideoCapture(0)
	homography = None
	camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
	orb = cv2.ORB_create()
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	model = cv2.imread('ar_marker.png', 0)
	kp_model, des_model = orb.detectAndCompute(model, None)
	while(True):
		ret, frame = cap.read()
		frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		kp_frame, des_frame = orb.detectAndCompute(frame, None)
		matches = bf.match(des_model, des_frame)
		matches = sorted(matches, key=lambda x: x.distance)
		if(len(matches) > MIN_MATCHES):
			src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
			dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
			h, w = model.shape
			pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
			dst = cv2.perspectiveTransform(pts, M)
			frame = cv2.polylines(frame, [np.int32(dst)], True, 0, 3, cv2.LINE_AA)
			#frame = cv2.drawMatches(model, kp_model, frame, kp_frame,matches[:MIN_MATCHES], 0, flags=2)
		cv2.imshow('frame', frame)
		if(cv2.waitKey(1) & 0xFF==ord('q')):
			break
	cap.release()
	cv2.destroyAllWindows()


def matchFeatures(cap,img2):
	MIN_MATCHES = 15  
	model = cv2.imread(img2, 0)
	orb = cv2.ORB_create()              
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
	kp_model, des_model = orb.detectAndCompute(model, None)  
	kp_frame, des_frame = orb.detectAndCompute(cap, None)
	matches = bf.match(des_model, des_frame)
	matches = sorted(matches, key=lambda x: x.distance)
	if len(matches) > MIN_MATCHES:
		# draw first 15 matches.
		cap = cv2.drawMatches(model, kp_model, cap, kp_frame,matches[:MIN_MATCHES], 0, flags=2)
		# show result
		cv2.imshow('frame', cap)
		cv2.waitKey(0)
	else:
		print ("Not enough matches have been found - %d/%d" % (len(matches),MIN_MATCHES))
startStream()
