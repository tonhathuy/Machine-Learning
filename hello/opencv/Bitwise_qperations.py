import cv2 as cv 
import numpy as np 
 
img1 = np.zeros((250, 500, 3), np.uint8)
img1 = cv.rectangle(img1, (200, 0), (300, 100), (255, 255, 255), -1)
img2 = cv.imread("image_1.png")

# bitAnd = cv.bitwise_and(img2, img1)
bitOr = cv.bitwise_or(img2, img1)

cv.imshow("img1", img1)
cv.imshow("img2", img2)
# cv.imshow("bitAnd", bitAnd)
cv.imshow("bitOr", bitOr)

cv.waitKey(0)
cv.destroyAllWindows()