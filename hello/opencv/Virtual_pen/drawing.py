import cv2 
import numpy as np 
import time 

load_from_disk = True

if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)

kernel = np.ones((5,5), np.uint8)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

x1,y1=0,0
canvas=None

noiseth = 300

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]
    else:
        lower_range = np.array([160, 92, 100])
        upper_range = np.array([178, 255, 255])

    mask = cv2.inRange(hsv, lower_range, upper_range)

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key= cv2.contourArea)) > noiseth:
        c = max(contours, key= cv2.contourArea)

        x2,y2,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x2,y2), (x2+w, y2+h), (0,25,255), 2)
        if x1 == 0 and y1 == 0:
            x1,y1=x2,y2
        else:
            canvas = cv2.line(canvas, (x1,y1),(x2,y2), [255,0,0], 4)
        x1,y1= x2, y2
    else:
        x1,y1=0,0

    
    frame = cv2.add(frame,canvas)
    stacked = np.hstack((canvas,frame))
    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.6,fy=0.6))

    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
