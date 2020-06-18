import cv2
import datetime

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(3, 3000)
cap.set(4,3000)

print(cap.get(3))
print(cap.get(4))


while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:
		out.write(frame)
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		font = cv2.FONT_HERSHEY_SIMPLEX
		text = 'Width: ' + str(cap.get(3)) + ' Height:' + str(cap.get(4))
		date = str(datetime.datetime.now())
		# img	=	cv.putText(	img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]	)
		frame = cv2.putText(frame, date, (10, 50), font, 1, (0, 225, 225), 2, cv2.LINE_AA)
		#(10, 50) : toa do
		cv2.imshow('frame', frame)
	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break
cap.release()
out.release()
cv2.destroyAllWindows()