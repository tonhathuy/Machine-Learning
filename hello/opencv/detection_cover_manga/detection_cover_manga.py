from cv2 import cv2 as cv 
import numpy as np 
from matplotlib import pyplot as plt

ori = cv.imread('manga.jpg')
img = cv.imread('manga.jpg')
result = cv.imread('manga.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(img, 80, 255, cv.THRESH_BINARY_INV)
contours, h = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for contour in contours:
    cv.drawContours(result, [contour], -1, 255, -1)
    hull = cv.convexHull(contour)
    cv.drawContours(result, [hull], -1, 255, -1)

plt.subplot(121), plt.imshow(ori)
plt.subplot(122), plt.imshow(result)
plt.show()
