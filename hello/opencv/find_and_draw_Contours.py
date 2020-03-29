import numpy as np 
import cv2 as cv 
# https://acodary.wordpress.com/2018/08/26/opencv-contours/
img = cv.imread('bien_so.jpg')
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

print("number of contours: " + str(len(contours)))

cv.drawContours(img, contours, -1, (0,255, 0), 3)

countContours = 0

for contour in contours:
    x, y, w, h = contourRect = cv.boundingRect(contour)
    if 10000 < w * h < 100000:
        countContours += 1
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))

cv.imshow('Image', img)
cv.imshow('Image GRAY', imgray)
cv.waitKey(0)
cv.destroyAllWindows()