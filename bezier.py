import time
import cv2
import numpy as np
import os
import scipy.signal
import sys

cap = cv2.VideoCapture("footage/radius2angle75.mp4")
# cap = cv2.VideoCapture("footage/0degree.mp4 ")
# cap = cv2.VideoCapture("footage/rootbeercar.mp4 ")
# fps = cap.get(cv2.CAP_PROP_FPS)

# print(fps)
w = 1/600
b = -1/600
smooth_time = 0
# block_5_left = np.array([[b,b,b,b,b], [b,b,b,b,w], [b,b,b,w,w], [b,b,w,w,w], [b,w,w,w,w]])
# block_5_right = np.array([[b,b,b,b,b], [w,b,b,b,b], [w,w,b,b,b], [w,w,w,b,b], [w,w,w,w,b]])

block_15_left = np.array([
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
[b,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
[w,w,w,w,w,w,w,w,w,w,w,w,w,w,w]
])
block_15_right = np.array([
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
[w,w,w,w,w,w,w,w,w,w,w,w,w,w,b],
[w,w,w,w,w,w,w,w,w,w,w,w,w,w,w]
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

leftblob = np.empty((20*15, 300))
rightblob = np.empty((20*15, 300))
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
    scanwidthl = 300
    scanwidthr = 300
####### main process loop
    for x in range(0,20):
        leftscan = gray[leftindex[1]:leftindex[1]+scanheight , leftindex[0]:leftindex[0] + scanwidthl]
        rightscan = gray[rightindex[1]:rightindex[1]+scanheight , rightindex[0]:rightindex[0] + scanwidthr]
        left = scipy.ndimage.convolve(leftscan, block_15_left)
        right = scipy.ndimage.convolve(rightscan, block_15_right)

        np.copyto(leftblob[x*15:x*15+15, 0:left.shape[1]], left)
        np.copyto(rightblob[x*15:x*15+15, 0:right.shape[1]], right)
        # print(left) 
        # print(right)
        # print(left)
        # idx = np.argmax(left)
        # print("")
        # print(idx)
        # print(left[idx%500, int(idx/500)])
        # find the maximum and draw a rectangle!
        # try:
        # so idxL/R is the index of the max thing, or the best boundary location as an x offset from the scan box width
        # idxLRf is the location of the box in the frame
        # leftindex and rightindex are the top left point of the scan box.
        if True:
            idxl = np.argmax(left[7])
            idxr = np.argmax(right[7])
            # print("left max:"+str(idxl))
            # print("right max:"+str(idxr))
            green = (0,255,0)
            red = (0,0,255)
            blue = (255,0,0)
            rect = 15
            # cv2.rectangle(right, (idxr, 0), (idxr + rect, rect), 255)
            # cv2.rectangle(left, (idxl, 0), (idxl + rect, rect), 255)
            idxlf = (idxl + leftindex[0], leftindex[1])
            idxrf = (idxr + rightindex[0] , rightindex[1])
            # print("left at frame loc:"+str(idxlf))
            # print("right at frame loc:"+str(idxrf))
            cv2.rectangle(frame, tuple(leftindex), (leftindex[0] + scanwidthl, leftindex[1] + scanheight), green, 2)
            cv2.rectangle(frame, tuple(rightindex), (rightindex[0] + scanwidthr, rightindex[1] + scanheight), green, 2)
            cv2.rectangle(frame, idxlf, (idxlf[0] + rect, idxlf[1] + rect), red, 3)
            cv2.rectangle(frame, idxrf, (idxrf[0] + rect, idxrf[1] + rect), blue, 3)
            # move the bounding box to next position
            if left[7][idxl] < 50:
                scanwidthl = 300
                # print("left poop")
                leftindex = [leftindex[0], leftindex[1] - 15]
            else:
                scanwidthl = 100
                leftindex = [idxlf[0] - int(scanwidthl/2), idxlf[1]-15]

            if right[7][idxr] < 50:
                scanwidthr = 300
                # print("right poop")
                rightindex = [rightindex[0], rightindex[1]-15]
            else:        
                scanwidthr = 100
                rightindex = [idxrf[0] - int(scanwidthr/2), idxrf[1]-15]

            # rightindex = [idxrf[0] - int(scanwidthr/2), idxrf[1]-15]
            # leftindex = [idxlf[0] - int(scanwidthl/2), idxlf[1]-15]
            if leftindex[0] < 0:
                leftindex[0] = 0
            if rightindex[0] > xsize-scanwidthr:
                rightindex[0] = xsize-scanwidthr
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
    cv2.imshow('left', leftblob)
    cv2.imshow('right', rightblob)

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