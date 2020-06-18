import cv2 as cv 
import numpy as np 

img = cv.imread('lane0.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=3)
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=150, maxLineGap=10)
'''
    param:
    minLineLength: số điểm tối thiểu tạo nên đường thẳng
    maxLineGap: khoảng cách lớn nhất giữa 2 điểm vẫn coi chúng là 1 đường thẳng
'''
print(lines)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img, (x1, y1), (x2, y2), (0,0,255), 2, cv.LINE_AA)

cv.imshow('img', img)
cv.imshow('img2', edges)
cv.waitKey(0)
cv.destroyAllWindows()

