import numpy as np 
import cv2 as cv 

# img = cv.imread('frame9.jpg')
cap = cv.VideoCapture('output.avi')
ret, img = cap.read()
x, y, w, h = 308, 330, 80, 100
track_window = (x, y, w, h)

# x, y, w, h = 314, 270, 100, 200
# track_window = (x, y, w, h)
roi = img[y: y+h, x: x+w]
cv.imshow("roi", roi)

hsv_roi = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])
# cv.normalize(roi_hist, roi_hist, 0, 255, norm_type= cv.NORM_MINMAX)

term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv.meanShift(mask, track_window, term_crit)
    x,y,w,h = track_window
    final_image = cv.rectangle(frame, (x,y), (x+w, y+h), 255, 3)

    cv.imshow("mask", mask)
    cv.imshow("frame", frame)

    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()