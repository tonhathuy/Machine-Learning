import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

img = cv.imread('gradient.png', 0)
_, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
_, th2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV) #INVERSE 
_, th3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
_, th4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO']
images = [img, th1 ,th2 ,th3 ,th4]

for i in range(5):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

# cv.imshow("Image", img)
# cv.imshow("th1", th1)
# cv.imshow("th2", th2)
# cv.imshow("th3", th3)
# cv.imshow("th4", th4)
plt.show()
# cv.waitKey(0)
# cv.destroyAllWindows()