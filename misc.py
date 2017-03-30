#import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import motortest
import VL53L0X

# # initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(320, 240))

# Create a VL53L0X object
tof = VL53L0X.VL53L0X()

# Start ranging
tof.start_ranging(VL53L0X.VL53L0X_BETTER_ACCURACY_MODE)
 
# # allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    	# grab the raw NumPy array representing the image, then initialize the timestamp
   	# and occupied/unoccupied text
   	distance = tof.get_distance()
   	if (distance < 150 and distance != -1):
                # stop the motors from running
                motorcmd(1,0,N,1)
                motorcmd(2,0,S,1)
                print("Am breaking")
                break

        
        else:
                image = frame.array
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

                        sloper = (y1 - y2) / ((x2 - x1) + 0.0000001)

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

                        slopel = (y1 - y2) / ((x2 - x1) + 0.0000001)

        
                if sloper > 75 and slopel > 75:
                        motorcmd(1,2048,N,0)
                        motorcmd(2,2048,S,0)

                elif sloper < -75 and slopel < -75:
                        motorcmd(1,2048,N,0)
                        motorcmd(2,2048,S,0)

                elif sloper < 75 and sloper > 50 and slopel < 75 and slopel > 50:
                        motorcmd(1,2458,N,0)
                        motorcmd(2,1638,S,0)

                elif sloper > -75 and sloper < -50 and slopel > -75 and slopel < -50:
                        motorcmd(1,1638,N,0)
                        motorcmd(2,2458,S,0)

                elif sloper < 50 and sloper > 25 and slopel < 50 and slopel > 25:
                        motorcmd(1,3277,N,0)
                        motorcmd(1,819,S,0)

                elif sloper > -50 and sloper < -25 and slopel > -50 and slopel < -25:
                        motorcmd(1,819,N,0)
                        motorcmd(2,3277,S,0)

                        
                cv2.imshow('cannyl',edgesl)
                cv2.imshow('cannyr',edgesr)
                cv2.imshow('left', left)
                cv2.imshow('right', right)

                print(" %s " %(time.time() - start_time))

                # show the frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

        # clear the stream in preparation for the next frame
	rawCapture.truncate(0)

	#if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cap.release()
