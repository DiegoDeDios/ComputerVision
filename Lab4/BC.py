import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

img = cv2.imread(sys.argv[1])

new_image = np.zeros(img.shape, img.dtype)

alpha = float(sys.argv[2])
beta = int(sys.argv[3])

for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        for c in range(img.shape[2]):
            new_image[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)

cv2.imwrite("BC.jpg", new_image)
