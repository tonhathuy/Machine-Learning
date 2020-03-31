import numpy as np 
import cv2 as cv 

img = cv.imread('shape.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(img_gray, 240, 255, cv.THRESH_BINARY)
contrours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

for controur in contrours:
    approx = cv.approxPolyDP(controur, 0.01* cv.arcLength(controur, True), True)
    '''
        param 
        curve: Input vector of a 2D point stored in numpy array (Python interface)
        epsilon: Parameter specifying the approximation accuracy. độ dài cạnh tối thiểu
        This is the maximum distance between the original curve and its approximatio
        closed: If true, the approximated curve is closed
        arcLength: Calculates a contour perimeter or a curve length
        https://www.programcreek.com/python/example/89328/cv2.approxPolyDP
    '''
    print('approx:', approx)
    cv.drawContours(img, [approx], 0, (0, 0, 0), 5)
    print('ravel: ', approx.ravel())
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    print('x:', x)
    print('y:', y)
    if len(approx) == 3:
        cv.putText(img, "Triangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 4:
        x, y, w, h = cv.boundingRect(approx)
        aspectRatio = float(w)/h
        print('chieu dai:', w)
        print('chieu rong:', h)
        print('ty le:', aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            cv.putText(img, "square", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        else:
            cv.putText(img, "rectangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 5:
        cv.putText(img, "Pentagon", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    elif len(approx) == 10:
        cv.putText(img, "star", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    else:
        cv.putText(img, "Circle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))


cv.imshow("shapes", img)
cv.waitKey(0)
cv.destroyAllWindows()