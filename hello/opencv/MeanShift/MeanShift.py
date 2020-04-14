import numpy as np 
import cv2 as cv 
# cap = cv.VideoCapture('slow_traffic_small.mp4')
cap = cv.VideoCapture('output.avi')

ret, frame = cap.read()

# x, y, w, h = 300, 200, 100, 50
x, y, w, h = 308, 330, 80, 100
track_window = (x, y, w, h)

roi = frame[y: y+h, x: x+w]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

# channel
# H (Hue) chỉ sắc thái có giá trị từ 00 - 3600 .
# S (Saturation) chỉ độ bảo hoà.
# V (Value) có giá trị từ 0 - 1. Các màu đạt giá trị bảo hòa khi s = 1 và v = 1.

#  Creating a Histogram from Data, tính toán hình dạng của biểu đồ Histogram.
roi_hist = cv.calcHist([hsv_roi], # C-style array of images, 8U or 32F
                    channels = [0], # C-style list of int’s identifying channels, channels 0 la h
                    mask = mask, #  pixels in ‘images’ count iff ‘mask’ nonzero
                    histSize = [180], #  C-style array, histogram sizes in each dim , hệ màu hue (0,180)
                    ranges = [0, 180] # C-style array of ‘dims’ pairs set bin sizes
                    )

'''
Normalize image to range [`min`, `max`]. -> vì roi_hist có giá trị lớn hơn 255 nên ta cần chuẩn hóa (0-255)

:param image: Image to be normalized
:param min: New minimum value of image
:param max: New maximum value of image
:param dtype: Output type of image. Default is same as `image`.
:return: The normalized image
'''
cv.normalize(roi_hist, roi_hist, 0, 255, norm_type= cv.NORM_MINMAX)
cv.imshow('roi', roi)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
# cv.imshow('roi',roi_hist)

cam = cv.VideoCapture(0)

while(1):
    ret, frame = cam.read()
    if ret == True:

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180],scale = 1)
        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, (x,y,w,h), term_crit)
        # Draw it on image
        x,y,w,h = track_window
        final_image = cv.rectangle(frame, (x,y), (x+w, y+h), 255, 3)

        cv.imshow('dst', dst)
        cv.imshow('final_image',final_image)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break