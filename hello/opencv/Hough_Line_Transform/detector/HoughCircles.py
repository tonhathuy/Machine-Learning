# Read image as gray-scale
import cv2 as cv
import numpy as np 
img = cv.imread('eyes2.jpg', cv.IMREAD_COLOR)
# Convert to gray-scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Blur the image to reduce noise
img_blur = cv.medianBlur(gray, 5)
# edges = cv.Canny(gray, 50, 150, apertureSize=3)
# Apply hough transform on the image
circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=10, minRadius=30, maxRadius=32)
# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv.imshow('img2', img)
# cv.imshow('img', edges)
cv.waitKey(0)
cv.destroyAllWindows()
