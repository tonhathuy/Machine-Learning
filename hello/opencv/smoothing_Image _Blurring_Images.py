import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 
#https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
# https://blog.vietanhdev.com/posts/2018-09-29-loc-anh-image-filtering/

img = cv.imread('Noise_salt_and_pepper.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

kernel = np.ones((5,5), np.float32)/25


dst = cv.filter2D(img, -1, kernel)
blur = cv.blur(img, (5, 5))
gblur = cv.GaussianBlur(img, (5, 5), 0)
median = cv.medianBlur(img, 5)#  khu nhieu
bilateralFilter = cv.bilateralFilter(img, 9, 75, 75) # min da

titles = ['Image', '2D Convolution', 'blur', 'GaussianBlur', 'median', 'bilateral']
images = [img, dst, blur, gblur, median, bilateralFilter]

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

