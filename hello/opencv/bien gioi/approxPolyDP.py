import numpy as np 
import cv2 as cv 

img = cv.imread('UK.png')
img = cv.resize(img, None,fx=0.25, fy=0.25, interpolation = cv.INTER_CUBIC)

canvas = np.zeros(img.shape, np.uint8)
img2gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

kernel = np.ones((5,5), np.float32)/25
img2gray = cv.filter2D(img2gray, -1, kernel)

ret, thresh = cv.threshold(img2gray, 250, 255, cv.THRESH_BINARY_INV)
contours, h = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

cnt = contours[0]
max_area = cv.contourArea(cnt)

for contour in contours:
    if cv.contourArea(contour) > max_area:
        cnt = contour
        max_area = cv.contourArea(contour)

approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)

hull = cv.convexHull(cnt)

print(cnt.shape)
cv.drawContours(canvas, cnt, -1, (0, 255, 0), 3)
cv.drawContours(canvas, approx, -1, (0, 0, 255), 3)

cv.imshow('img1', canvas)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()