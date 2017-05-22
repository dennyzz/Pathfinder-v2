# Instructions!
# scipy: the easiest install on windows is to use prebuilt wheels.
#   pip install wheel
#   then go to http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
#   and download numpy+mkl and scipy
#   pip install those files


#import the necessary packages
from picamera import PiCamera
from picamera.array import PiArrayOutput
import time
import cv2
import numpy as np
import os
import sys


res_x = 320
res_y = 240
xsize = res_x
ysize = res_y

camera = PiCamera()
camera.resolution = (res_x, res_y)
camera.framerate = 30
rawCapture = PiArrayOutput(camera, size=(res_x, res_y))

# # allow the camera to warmup
time.sleep(0.1)
start_time = 0
# capture frames from the camera
# change settings to use the last full FOV sensor_mode
for frame in camera.capture_continuous(rawCapture, format="bgra", sensor_mode=4, use_video_port=True, resize=(res_x, res_y)):
#while True:
    # grab the raw NumPy array representing the image,
    frame = frame.array

    # maybe a speedup if we clear the stream here...?
    rawCapture.truncate(0)

    # step1: grayscale
    # change to bgra version to see if cutting alpha stripping helps
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    
    cv2.imshow('frame', frame)
    #cv2.imshow('left', leftblob)
    #cv2.imshow('right', rightblob)

    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    # try moving the stream clear
    # rawCapture.truncate(0)

    #if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    if start_time = 0:
        proc_time = 0
        fps_calc = 0
    else:
        proc_time = (time.time() - start_time)*1000
        
    if smooth_time == 0:
        smooth_time = proc_time
        fps_calc = 0
    else:
        smooth_time = 0.9*smooth_time + 0.1*proc_time
        fps_calc = int(1000/smooth_time)
                
    # sys.stdout.write("\rtimetot:%dmS fps:%d algotime:%dmS posttime:%dmS pretime:%dmS       " %(smooth_time, fps_calc, proc_algo_time_s, proc_post_time_s, proc_pre_time_s))
    sys.stdout.write("\rtime:%dmS, fps:%d       " % (smooth_time, fps_calc))
    sys.stdout.flush()
    #time it from here
    start_time = time.time()


sys.exit(0)
