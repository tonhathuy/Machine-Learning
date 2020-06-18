import cv2 as cv 
import numpy as np 

cap = cv.VideoCapture(0)

#Create old gray frame
_, frame = cap.read()
old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# Lucas kanade params
lk_params = dict(winSize = (10, 10),
                maxLevel = 2,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
                )

# Mouse func
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)

cv.namedWindow("Frame")
cv.setMouseCallback("Frame", select_point)

point_selected = False
point = ()
old_points = np.array([[]])

while True:
    _, frame = cap.read()
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if point_selected is True:
        cv.circle(frame, point, 5, (0,0,255), 2)

        new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_gray = gray_frame.copy()
        old_points = new_points
        #print(new_points)

        x, y = new_points.ravel()
        cv.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv.imshow("Frame", frame)


    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

cap.release()
cv.destroyAllWindows()