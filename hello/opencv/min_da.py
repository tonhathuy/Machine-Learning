import numpy as np 
import cv2 as cv 

def nothing(x):
    pass

cv.namedWindow('image')

cv.createTrackbar('d', 'image', 1, 20, nothing)
cv.createTrackbar('color', 'image', 0, 255, nothing)
cv.createTrackbar('space', 'image', 0, 255, nothing)

while(1):
    img = cv.imread('tan-nhang.jpg')

    
    d = cv.getTrackbarPos('d', 'image')
    color = cv.getTrackbarPos('color', 'image')
    space = cv.getTrackbarPos('space', 'image')

    bilateralFilter = cv.bilateralFilter(img, d, color, space)

    cv.imshow("lam min", bilateralFilter)
    cv.imshow("tan nhang", img)
    k = cv.waitKey(1)
    if k ==27:
        break

cv.destroyAllWindows()