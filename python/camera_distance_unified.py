#!/usr/bin/python

# MIT License
# 
# Copyright (c) 2017 John Bryan Moore
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import VL53L0X
import cv2
import numpy as np

cap = cv2.VideoCapture("0degree.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

# Create a VL53L0X object
tof = VL53L0X.VL53L0X()

# Start ranging
tof.start_ranging(VL53L0X.VL53L0X_BETTER_ACCURACY_MODE)

while(True):
    distance = tof.get_distance()
    if (distance < 150 and distance != -1):
        # stop the motors from running
		print("Am breaking")
		break
	
	# run camera capture and image processing
	
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
	linesl = cv2.HoughLines(edgesl,35,np.pi/180,100)
	linesr = cv2.HoughLines(edgesr,35,np.pi/180,100)



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
		
    time.sleep(2000.00/1000000.00)

tof.stop_ranging()
cap.release()

