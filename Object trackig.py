import cv2
import numpy as np

video = cv2.VideoCapture('Move detection.mp4')

ret, frame1 = video.read()

frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1 = cv2.GaussianBlur(frame1, (31, 31), 0)
while True:
	ret, frame2 = video.read()
	if ret == False: break
	gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (31, 31), 0)


	dif = cv2.absdiff(gray, frame1)
	_, th = cv2.threshold(dif, 40, 255, cv2.THRESH_BINARY)
		
	cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
	for c in cnts:
		area = cv2.contourArea(c)
		if area > 120:
			x,y,w,h = cv2.boundingRect(c)
			cv2.rectangle(frame2, (x,y), (x+w,y+h),(0,255,0),2)

	cv2.imshow('Frame',frame2)

	if cv2.waitKey(30) & 0xFF == ord ('q'):
		break

video.release()
