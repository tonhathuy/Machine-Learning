import cv2 as cv 
import numpy as np 

dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

markerImage = np.zeros((200, 200), dtype=np.uint8)
markerImage = cv.aruco.drawMarker(dictionary, 34, 200, markerImage, 1);

cv.imwrite("marker34.png", markerImage);

