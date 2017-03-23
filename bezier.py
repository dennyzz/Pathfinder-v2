import time
import cv2
import numpy as np
import os
import scipy.signal
import sys

# cap = cv2.VideoCapture("footage/radius2angle75.mp4")
# cap = cv2.VideoCapture("footage/0degree.mp4 ")
cap = cv2.VideoCapture("footage/rootbeercar.mp4 ")
fps = cap.get(cv2.CAP_PROP_FPS)

# print(fps)
w = -1/105
b = 1/105
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
block_15_right = np.array([
[b,b,b,b,b,b,b,b,b,b,b,b,b,b,b],
[w,b,b,b,b,b,b,b,b,b,b,b,b,b,b],
[w,w,b,b,b,b,b,b,b,b,b,b,b,b,b],
[w,w,w,b,b,b,b,b,b,b,b,b,b,b,b],
[w,w,w,w,b,b,b,b,b,b,b,b,b,b,b],
[w,w,w,w,w,b,b,b,b,b,b,b,b,b,b],
[w,w,w,w,w,w,b,b,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,b,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,w,b,b,b,b,b,b],
[w,w,w,w,w,w,w,w,w,w,b,b,b,b,b],
[w,w,w,w,w,w,w,w,w,w,w,b,b,b,b],
[w,w,w,w,w,w,w,w,w,w,w,w,b,b,b],
[w,w,w,w,w,w,w,w,w,w,w,w,w,b,b],
[w,w,w,w,w,w,w,w,w,w,w,w,w,w,b]
])

# block_15_right = np.array([
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b],
# [w,w,w,w,w,w,w,w,b,b,b,b,b,b,b]
# ])


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


# frame = cv2.imread("images/lanestill.png")
# print(frame)

while cap:
    ret, frame = cap.read()
    ysize = frame.shape[0]
    xsize = frame.shape[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    leftindex = [0,ysize-150]
    rightindex = [xsize-300, ysize-150]
    start_time = time.time()
    scanwidth = 300
    scanheight = 15
####### main process loop
    for x in range(0,20):
        leftscan = gray[leftindex[1]:leftindex[1]+scanheight , leftindex[0]:leftindex[0] + scanwidth]
        rightscan = gray[rightindex[1]:rightindex[1]+scanheight , rightindex[0]:rightindex[0] + scanwidth]
        left = scipy.ndimage.convolve(leftscan, block_15_left)
        right = scipy.ndimage.convolve(rightscan, block_15_right)
        next = 0

        # print(left)
        # idx = np.argmax(left)
        # print("")
        # print(idx)
        # print(left[idx%500, int(idx/500)])
        # find the maximum and draw a rectangle!
        # try:
        if True:
            idxl = np.argmax(left)
            idxr = np.argmax(right)
            idxl = [idxl % scanwidth, 0]
            idxr = [idxr % scanwidth, 0]
            # print("left max:"+str(idxl))
            # print("right max:"+str(idxr))
            green = (0,255,0)
            red = (0,0,255)
            blue = (255,0,0)
            rect = 15
            cv2.rectangle(right, tuple(idxr), (idxr[0] + rect, idxr[1] + rect), 255)
            cv2.rectangle(left, tuple(idxl), (idxl[0] + rect, idxl[1] + rect), 255)
            idxlf = (idxl[0]+leftindex[0], idxl[1]+leftindex[1])
            idxrf = (idxr[0]+rightindex[0] , idxr[1]+rightindex[1])
            # print("left at frame loc:"+str(idxlf))
            # print("right at frame loc:"+str(idxrf))
            cv2.rectangle(frame, tuple(leftindex), (leftindex[0] + scanwidth, leftindex[1] + scanheight), green, 2)
            cv2.rectangle(frame, tuple(rightindex), (rightindex[0] + scanwidth, rightindex[1] + scanheight), green, 2)
            cv2.rectangle(frame, idxlf, (idxlf[0] + rect, idxlf[1] + rect), red, 3)
            cv2.rectangle(frame, idxrf, (idxrf[0] + rect, idxrf[1] + rect), blue, 3)
            # move the bounding box to next position
            scanwidth = 100
            leftindex = [idxlf[0] - int(scanwidth/2), idxlf[1]-15]
            rightindex = [idxrf[0] - int(scanwidth/2), idxrf[1]-15]
            if leftindex[0] < 0:
                leftindex[0] = 0
            if rightindex[0] > xsize-scanwidth:
                rightindex[0] = xsize-scanwidth
            # print("new left index:"+str(leftindex))
            # print("new right index:"+str(rightindex))
        # except: 
            # print("failed to find line")
            # leftindex = (idxlf[0], idxlf[1]+15)
            # rightindex = (idxrf[0], idxrf[1]+15)
        # print(start.size)
        # print(left.size)
    ####### end processing
    proc_time = time.time() - start_time
    if smooth_time == 0:
        smooth_time = proc_time
    else:
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
    if key == ord("n"):
        print("next")
        next = 1
    if key == ord("q"):
        break

cap.release()
sys.exit(0)