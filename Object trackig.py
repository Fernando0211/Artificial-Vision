# importing libraries
import cv2 as cv
import numpy as np

vid = cv.VideoCapture('Object tracking.mp4')

if (vid.isOpened()== False):
	print("Error opening video file")

while(vid.isOpened()):
	
	ret, frame = vid.read()
	if ret == True:
		#Preprocessing
		imgray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		blur = cv.GaussianBlur(imgray, (31, 31), 0)
		edges = cv.Canny(imgray, 100, 150)

		#Contours
		contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

	    # Rectangle depending in the area of the contour

		for contour in contours:
			area = cv.contourArea(contour)
			if area > 100:
				x, y, w, h = cv.boundingRect(contour)
   
				cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

		cv.imshow('Frame', frame)

	# Press X on keyboard to exit
		if cv.waitKey(30) & 0xFF == ord('x'):
			break

# Break the loop
	else:
		break

vid.release()
cv.destroyAllWindows()