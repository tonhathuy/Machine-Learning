from cv2 import cv2 as cv 
import numpy as np 
import glob

list_img = glob.iglob("letters/*")

for img_title in list_img:
    img = cv.imread(img_title, cv.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
    img_and_magnitude = np.concatenate((img, magnitude_spectrum), axis=1)

    cv.imshow(img_title, img_and_magnitude)

cv.waitKey(0)
cv.destroyAllWindows()

# cap = cv.VideoCapture(0)

# while True:
#     _, frame = cap.read()
#     f = np.fft.fft2(frame)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = 20*np.log(np.abs(fshift))
#     magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
#     img_and_magnitude = np.concatenate((frame, magnitude_spectrum), axis=1)

#     cv.imshow("img", img_and_magnitude)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()