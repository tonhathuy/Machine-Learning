import cv2 as cv 
import numpy as np 

if __name__ == '__main__':
    img_src = cv.imread('book2.jpg')
    pst_src = np.array([[363, 99], [607, 122], [257, 659], [562, 701]])
    img_dst = cv.imread('book1.jpg')
    pst_dst = np.array([[555, 221], [721, 293], [161, 379], [323, 528]])

    h, status = cv.findHomography(pst_src, pst_dst)

    img_out = cv.warpPerspective(img_src, h, (img_dst.shape[1], img_dst.shape[0]))
    
    cv.imshow('src', img_src)
    cv.imshow('dst', img_dst)
    cv.imshow('w', img_out)

    cv.waitKey(0)
    
