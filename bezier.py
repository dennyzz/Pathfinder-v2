import time
import cv2
import numpy as np
import os
import scipy.signal
import sys

# cap = cv2.VideoCapture("radius2angle75.mp4")
# cap = cv2.VideoCapture("0degree.mp4 ")
# cap = cv2.VideoCapture("rootbeercar.mp4 ")
# fps = cap.get(cv2.CAP_PROP_FPS)

# print(fps)
w = 1/105
b = -1/105
smooth_time = 0;
# block_5_left = np.array([[b,b,b,b,b], [b,b,b,b,w], [b,b,b,w,w], [b,b,w,w,w], [b,w,w,w,w]])
# block_5_right = np.array([[b,b,b,b,b], [w,b,b,b,b], [w,w,b,b,b], [w,w,w,b,b], [w,w,w,w,b]])

block_15_left = np.array([
[b,b,b,b,b,b,b,b,b,b,b,b,b,b,b],
[b,b,b,b,b,b,b,b,b,b,b,b,b,b,w],
[b,b,b,b,b,b,b,b,b,b,b,b,b,w,w],
[b,b,b,b,b,b,b,b,b,b,b,b,w,w,w],
[b,b,b,b,b,b,b,b,b,b,b,w,w,w,w],
[b,b,b,b,b,b,b,b,b,b,w,w,w,w,w],
[b,b,b,b,b,b,b,b,b,w,w,w,w,w,w],
[b,b,b,b,b,b,b,b,w,w,w,w,w,w,w],
[b,b,b,b,b,b,b,w,w,w,w,w,w,w,w],
[b,b,b,b,b,b,w,w,w,w,w,w,w,w,w],
[b,b,b,b,b,w,w,w,w,w,w,w,w,w,w],
[b,b,b,b,w,w,w,w,w,w,w,w,w,w,w],
[b,b,b,w,w,w,w,w,w,w,w,w,w,w,w],
[b,b,w,w,w,w,w,w,w,w,w,w,w,w,w],
[b,w,w,w,w,w,w,w,w,w,w,w,w,w,w]
])
# block_15_right = np.array([
# [b,b,b,b,b,b,b,b,b,b,b,b,b,b,b],
# [w,b,b,b,b,b,b,b,b,b,b,b,b,b,b],
# [w,w,b,b,b,b,b,b,b,b,b,b,b,b,b],
# [w,w,w,b,b,b,b,b,b,b,b,b,b,b,b],
# [w,w,w,w,b,b,b,b,b,b,b,b,b,b,b],
# [w,w,w,w,w,b,b,b,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,b,b,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,b,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,w,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,w,w,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,w,w,w,b,b,b,b],
# [w,w,w,w,w,w,w,w,w,w,w,w,b,b,b],
# [w,w,w,w,w,w,w,w,w,w,w,w,w,b,b],
# [w,w,w,w,w,w,w,w,w,w,w,w,w,w,b]
# ])

block_15_right = np.array([
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b]
])


# block_20_right = np.array([
# [b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b],
# [w,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b],
# [w,w,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b],
# [w,w,w,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b],
# [w,w,w,w,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b],
# [w,w,w,w,w,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,b,b,b,b,b,b,b,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,b,b,b,b,b,b,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,w,b,b,b,b,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,w,w,b,b,b,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,w,w,w,b,b,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,w,w,w,w,b,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,w,w,w,w,w,w,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,b,b,b,b],
# [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,b,b,b],
# [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,b,b],
# [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,b]
# ])

# block_20_left = np.array([
# [b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b],
# [b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,w],
# [b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,w,w],
# [b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,w,w,w],
# [b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,w,w,w,w],
# [b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,w,w,w,w,w],
# [b,b,b,b,b,b,b,b,b,b,b,b,b,b,w,w,w,w,w,w],
# [b,b,b,b,b,b,b,b,b,b,b,b,b,w,w,w,w,w,w,w],
# [b,b,b,b,b,b,b,b,b,b,b,b,w,w,w,w,w,w,w,w],
# [b,b,b,b,b,b,b,b,b,b,b,w,w,w,w,w,w,w,w,w],
# [b,b,b,b,b,b,b,b,b,b,w,w,w,w,w,w,w,w,w,w],
# [b,b,b,b,b,b,b,b,b,w,w,w,w,w,w,w,w,w,w,w],
# [b,b,b,b,b,b,b,b,w,w,w,w,w,w,w,w,w,w,w,w],
# [b,b,b,b,b,b,b,w,w,w,w,w,w,w,w,w,w,w,w,w],
# [b,b,b,b,b,b,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
# [b,b,b,b,b,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
# [b,b,b,b,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
# [b,b,b,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
# [b,b,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
# [b,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w]
# ])


while(True):
    start_time = time.time()
    frame = cv2.imread("images/lanestill.png")
    # ret, frame = cap.read()
    # print(frame)
    if frame is None:
        print("exiting")
        break
    ysize = frame.shape[0]
    xsize = frame.shape[1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scanwidth = 300
    scanheight = 15
    leftindex = (0,ysize-150)
    rightindex = (xsize-300, ysize-150)
#  so grab 150x15 blocks
    leftscan = gray[leftindex[1]:leftindex[1]+scanheight , leftindex[0]:leftindex[0] + scanwidth]
    rightscan = gray[rightindex[1]:rightindex[1]+scanheight , rightindex[0]:rightindex[0] + scanwidth]
####### main process loop
    left = scipy.ndimage.convolve(leftscan, block_15_left)
    right = scipy.ndimage.convolve(rightscan, block_15_right)

    # print(left)
    # idx = np.argmax(left)
    # print("")
    # print(idx)
    # print(left[idx%500, int(idx/500)])
    # find the maximum and draw a rectangle!
    try:
        idxl = np.argmax(left)
        idxr = np.argmax(right)
        idxl = idxl % scanwidth, int(idxl/scanwidth)
        idxr = idxr % scanwidth, int(idxr/scanwidth) 
        green = (0,255,0)
        red = (0,0,255)
        blue = (255,0,0)
        rect = 15
        cv2.rectangle(right, idxr, (idxr[0] + rect, idxr[1] + rect), green,1,8)
        cv2.rectangle(left, idxl, (idxl[0] + rect, idxl[1] + rect), red,1,8)
        idxlf = (idxl[0]+leftindex[0], idxl[1]+leftindex[1])
        idxrf = (idxr[0]+rightindex[0] , idxr[1]+rightindex[1])
        cv2.rectangle(frame, idxlf, (idxlf[0] + rect, idxlf[1] + rect), red,1,8)
        cv2.rectangle(frame, idxrf, (idxrf[0] + rect, idxrf[1] + rect), green,1,8)
        leftindex = (idxlf[0] - 150, idxlf[1]+15)
        rightindex = (idxrf[0] - 150, idxrf[1]+15)
    except: 
        print("failed to find line")
        leftindex = (idxlf[0], idxlf[1]+15)
        rightindex = (idxrf[0], idxrf[1]+15)
    leftscan = gray[leftindex[1]:leftindex[1]+scanheight , leftindex[0]:leftindex[0] + scanwidth]
    rightscan = gray[rightindex[1]:rightindex[1]+scanheight , rightindex[0]:rightindex[0] + scanwidth]

    # print(start.size)
    # print(left.size)





####### end processing
    proc_time = time.time() - start_time
    smooth_time = 0.95*smooth_time + 0.05*proc_time
    fps_calc = int(1/smooth_time) 
    # print(smooth_time, "\r") 
    sys.stdout.write("\rtime: %f, frames: %d               " % (smooth_time, fps_calc))
    sys.stdout.flush()
#time it from here

    cv2.imshow('frame', frame)
    cv2.imshow('left', left)
    cv2.imshow('right', right)
    # show the frame
    #cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    # rawCapture.truncate(0)

    #if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cap.release()
sys.exit(0)