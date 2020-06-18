import numpy as np
import os
import cv2

img = cv2.imread('1.jpg')
Z = img.reshape((-1,3))
print('Z reshape: ',Z)
# convert to np.float32
Z = np.float32(Z)
print('Z float: ',Z)

# define criteria, number of clusters(K) and apply kmeans()
# https://docs.opencv.org/3.4/d9/d5d/classcv_1_1TermCriteria.html
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 100
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite("tmp.jpg", res2)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res = res.reshape(img.shape)
no = 1
for i in center:
    no += 1
    res2 = cv2.inRange(res, i, i)
    res2 = cv2.bitwise_not(res2)
    cv2.imwrite(".tmp.bmp", res2)
    os.system("potrace .tmp.bmp -s --flat")
    drawSVG(".tmp.svg", "#%02x%02x%02x" % (i[2], i[1], i[0]))
te.done() 