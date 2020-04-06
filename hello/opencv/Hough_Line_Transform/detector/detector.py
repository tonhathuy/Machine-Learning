import matplotlib.pylab as plt
import cv2
import numpy as np

image = cv2.imread('lane0.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [
    (0, height),
    (width/2, height/2),
    (width, height)
]

def region_of_interest(img, vertices):
    print('vertices:', vertices)
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    # match_mask_color = (255,) * channel_count # print (255, 255, 255)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color) # fill up mask by white triangle with vertices coordinates
    # print('mask:', mask)
    masked_image = cv2.bitwise_and(img, mask) # all number and 0 = 0
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
    
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny_image = cv2.Canny(gray_image, 100, 200)

cropped_image = region_of_interest(canny_image,
                np.array([region_of_interest_vertices], dtype=np.int32),) # fillpoly only use interger -> int32 

lines = cv2.HoughLinesP(cropped_image, 
                       rho = 6,
                        theta = np.pi/60,
                        threshold = 160,
                        lines= np.array([]),
                        minLineLength= 40,
                        maxLineGap= 25
                        )

img_with_lines = draw_the_lines(image, lines)

plt.imshow(img_with_lines)
plt.show()