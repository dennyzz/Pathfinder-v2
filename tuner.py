import time
import cv2
import numpy as np

cap = cv2.VideoCapture("0degree.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

while(True):
	ret, frame = cap.read()

	ysize = frame.shape[0]
	xsize = frame.shape[1]


	left = frame[0:ysize, 0:int(xsize/2)]
	right = frame[0:ysize, int(xsize/2):xsize]

	grayright = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
	grayleft = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

	edgesl = cv2.Canny(grayleft,75,100,apertureSize = 3)
	edgesr = cv2.Canny(grayright,75,100,apertureSize = 3)
	minLineLength = 100
	maxLineGap = 10
	linesl = cv2.HoughLines(edgesl,35,np.pi/180,100,minLineLength,maxLineGap)
	linesr = cv2.HoughLines(edgesr,35,np.pi/180,100,minLineLength,maxLineGap)

	if linesr != None:
		for rho,theta in linesr[0]:

			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			cv2.line(right,(x1,y1),(x2,y2),(0,0,255),2)

	if linesl != None:
		# print(linesl)
		for rho,theta in linesl[0]:

			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			cv2.line(left,(x1,y1),(x2,y2),(0,0,255),2)
		
	cv2.imshow('cannyl',edgesl)
	cv2.imshow('cannyr',edgesr)
	cv2.imshow('left', left)
	cv2.imshow('right', right)


	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# pipe.stdout.flush()

	# clear the stream in preparation for the next frame
	#rawCapture.truncate(0)

	#if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cap.release()