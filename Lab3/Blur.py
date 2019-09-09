import cv2
import numpy
import sys
src = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
 

dst = cv2.GaussianBlur(src,(9,9),cv2.BORDER_DEFAULT)
 

cv2.imshow("Gaussian Smoothing",numpy.hstack((dst)))
cv2.imwrite("Blurred.jpg",dst) 
