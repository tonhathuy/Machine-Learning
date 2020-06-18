import cv2 
import numpy as np 
import time 

load_from_disk = True

if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)

pen_img = cv2.resize(cv2.imread('pen.jpg',1), (50, 50))
eraser_img = cv2.resize(cv2.imread('eraser.jpg',1), (50, 50))
red_img = cv2.resize(cv2.imread('red.png',1), (50, 50))
blue_img = cv2.resize(cv2.imread('blue.png',1), (50, 50))

kernel = np.ones((5,5), np.uint8)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
background_threshold = 600

switch = 'Pen'
last_switch = time.time()
switch_img = 'Blue'


x1,y1=0,0
canvas=None
noiseth = 300

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)


    top_left = frame[0:50, 0:50]
    fgmask = backgroundobject.apply(top_left)

    top_right = frame[0:50, 100:150]
    rmask = backgroundobject.apply(top_right)

    switch_thresh = np.sum(fgmask==255)
    switch_th = np.sum(rmask==255)

    if switch_thresh>background_threshold and (time.time()-last_switch) >1:
        last_switch = time.time()

        if switch == 'Pen':
            switch = 'Eraser'
        else:
            switch = 'Pen'

    if switch_th>background_threshold and (time.time()-last_switch) >1:
        last_switch = time.time()
        
        if switch_img == 'Blue':
            switch_img = 'Reg'
        else:
            switch_img = 'Blue'
    

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
            if switch == 'Pen':
                color = [255,0,0] if switch_img =='Blue' else [0,0,255]
                canvas = cv2.line(canvas, (x1,y1),(x2,y2),color, 5)
            else:
                cv2.circle(canvas, (x2, y2), 20, (0,0,0), -1)
        x1,y1= x2, y2
    else:
        x1,y1=0,0

    _ , mask = cv2.threshold(cv2.cvtColor (canvas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
    background = cv2.bitwise_and(frame, frame,mask = cv2.bitwise_not(mask))
    frame = cv2.add(foreground,background)

    if switch != 'Pen':
        cv2.circle(frame, (x1, y1), 20, (255,255,255), -1)
        frame[0: 50, 0: 50] = eraser_img
    else:
        frame[0: 50, 0: 50] = pen_img

    if switch != 'Blue':
        frame[0: 50, 100: 150] = red_img
    else:
        frame[0: 50, 200: 150] = blue_img


    stacked = np.hstack((canvas,frame))
    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.6,fy=0.6))

    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
