import cv2
import numpy
from matplotlib import pyplot as plt

print("Please choose an option:")
selection = input(" 1) Image\n 2) Video\n 3) Camera\n Select: ")


def loadImage(img_name):
	img = cv2.imread(img_name,0)
	cv2.imshow("Image",img)
	select = input("1) Histogram\n 2) Threshold\n Select:")
	if select == "1":
		histogram(img)
	elif select == "2":
		thresholding(img)
	
def histogram(img):
	hist,bins = numpy.histogram(img.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_normalized = cdf * hist.max()/ cdf.max()
	plt.plot(cdf_normalized, color = 'b')
	plt.hist(img.flatten(),256,[0,256], color = 'r')
	plt.xlim([0,256])
	plt.legend(('cdf','histogram'), loc = 'upper left')
	plt.show()
def thresholding(img):
	ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	plt.imshow(thresh1,'gray')
	plt.title("Threshold")
	plt.xticks([]),plt.yticks([])
	plt.show()
def starStream(select):
	cap = cv2.VideoCapture(0)
	while(True):
		ret, frame = cap.read()
		if select == "1":
			frm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		elif select == "2":
			frm = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
		ret,thresh1 = cv2.threshold(frm,127,255,cv2.THRESH_BINARY)
		cv2.imshow('frame',thresh1)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
def startVideo(file,selection):
	cap = cv2.VideoCapture(file)
	while(True):
		ret, frame = cap.read()
		if select == "1":
			frm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		elif select == "2":
			frm = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
		ret,thresh1 = cv2.threshold(frm,127,255,cv2.THRESH_BINARY)
		cv2.imshow('frame',thresh1)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
if(selection == "1"):
	image = input("Select image name: ")
	loadImage(image)
elif(selection == "2"):
	video = input("Select video name: ")
	select = input("Select threshold:\n 1)GrayScale Binary\n 2)Color Binary\n Select:")
	startVideo(video, select)
elif(selection == "3"):
	select = input("Select threshold:\n 1)GrayScale Binary\n 2)Color Binary\n Select:")
	starStream(select)

