   # import the necessary packages
# from picamera.array import PiRGBArray
# from picamera import PiCamera
import time
import cv2
import numpy as np

###################################################################
###################################################################
import subprocess as sp

FFMPEG_BIN = "ffmpeg.exe"

command = [FFMPEG_BIN, '-i', '0degree.mp4', '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-']
pipe = sp.Popen(command, stderr = sp.PIPE, stdout = sp.PIPE, bufsize = 10**8)

raw_image = pipe.stdout.read(320*240*3)
image = np.fromstring(raw_image, dtype = 'uint8')
image = image.reshape(320, 240, 3)

#pipe.stdout.flush()
###################################################################
###################################################################






# # initialize the camera and grab a reference to the raw camera capture
# camera = PiCamera()
# camera.resolution = (320, 240)
# camera.framerate = 30
# rawCapture = PiRGBArray(camera, size=(320, 240))
 
# # allow the camera to warmup
# time.sleep(0.1)
 
# capture frames from the camera
#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
#for frame in image:
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	#image = frame.array
ysize = image.shape[0]
xsize = image.shape[1]


left = image[0:ysize, 0:int(xsize/2)]
right = image[0:ysize, int(xsize/2):xsize]

grayright = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
grayleft = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

edgesl = cv2.Canny(grayleft,50,150,apertureSize = 3)
edgesr = cv2.Canny(grayright,50,150,apertureSize = 3)
minLineLength = 100
maxLineGap = 10
linesl = cv2.HoughLines(edgesl,1,np.pi/180,50)
linesr = cv2.HoughLines(edgesr,1,np.pi/180,50)



#if linesr != None:
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

#if linesl != None:
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
cv2.imshow("Frame", image)
# key = cv2.waitKey(1) & 0xFF

# pipe.stdout.flush()

# clear the stream in preparation for the next frame
#rawCapture.truncate(0)

# if the `q` key was pressed, break from the loop
# if key == ord("q"):
# 	break
