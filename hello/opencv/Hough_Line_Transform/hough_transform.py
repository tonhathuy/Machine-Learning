import cv2 as cv 
import numpy as np 

img = cv.imread('sudoku.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=3)
lines = cv.HoughLines(edges, rho=1, theta=np.pi/180, threshold=200)
print(lines)
for line in lines:
    rho, theta = line[0]
    # print(line)
    b = np.sin(theta)
    a = np.cos(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(img, (x1, y1), (x2, y2), (0,0,255), 2, cv.LINE_AA)

cv.imshow('img', img)
cv.imshow('img2', edges)
cv.waitKey(0)
cv.destroyAllWindows()

