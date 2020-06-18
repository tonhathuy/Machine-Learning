import numpy as np
import cv2 as cv

img = cv.imread('2.jpg')
img2 = cv.imread('LOGO.png')

print(img.shape)
print(img.size)
print(img.dtype)

b, g, r = cv.split(img)
img = cv.merge((b, g, r))

img = cv.resize(img, (500, 500))

# dst = cv.add(img, img2)
dst = cv.addWeighted(img, .3, img2, .7, 0)

cv.imshow('image', dst)
cv.waitKey()
cv.destroyAllWindows()