import numpy as np 
import cv2 as cv 

img = cv.imread('template.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
template = cv.imread('template_matching.jpg', 0)
w, h = template.shape[::-1]

res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
print(res)
loc = np.where(res >= 0.75)

for pt in zip(*loc[::-1]):
    cv.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0, 0, 255), 2)

cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()