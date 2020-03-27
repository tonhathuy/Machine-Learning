import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

img = cv.imread('smartied.png', cv.IMREAD_GRAYSCALE)
_, mask = cv.threshold(img, 220, 255, cv.THRESH_BINARY_INV)

# https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
# The kernel slides through the image (as in 2D convolution). A pixel in the original image (either 1 or 0) 
# will be considered 1 only if all the pixels under the kernel is 1,

kernal = np.ones((2,2), np.uint8)

dilation = cv.dilate(mask, kernal, iterations=2)
erosion = cv.erode(mask, kernal, iterations=3)
opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernal)
closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernal)
mg = cv.morphologyEx(mask, cv.MORPH_GRADIENT, kernal)
th = cv.morphologyEx(mask, cv.MORPH_TOPHAT, kernal)

titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'mg', 'th']
images = [img, mask, dilation, erosion, opening, closing, mg, th]

for i in range(8):
    plt.subplot(2, 4, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()