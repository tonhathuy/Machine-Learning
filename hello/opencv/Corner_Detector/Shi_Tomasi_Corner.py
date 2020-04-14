import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('shape2d.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(gray, maxCorners= 200, qualityLevel= 0.01, minDistance=10)
'''
    params
    maxCorner: Maximum number of corners to return
    qualityLevel: the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
    minDistance: Minimum possible Euclidean distance between the returned corners 
'''
corners = np.int0(corners)
print('int0 is int64:', np.int0 is np.int64)

for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)
plt.imshow(img),plt.show()