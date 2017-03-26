# Instructions!
# scipy: the easiest install on windows is to use prebuilt wheels.
#   pip install wheel
#   then go to http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
#   and download numpy+mkl and scipy
#   pip install those files






import time
import cv2
import numpy as np
import os
import scipy.signal
import sys

# assume that we have proper inputs for our application
def eigenvalue(input, filter):
    output = np.empty()


    return output


# cap = cv2.VideoCapture("footage/radius2angle75.mp4")
# cap = cv2.VideoCapture("footage/0degree.mp4 ")
cap = cv2.VideoCapture("footage/rootbeercar.mp4 ")
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

block_left = block_15_left
block_right = block_15_right
blocksize = 15
halfblock = int(floor(blocksize/2))
algotype = "correlation"

f = open('workfile.txt', 'w')
print(f)
scanwidth = 300
scanheight = 15
scanspacing = 30
while cap:
    ret, frame = cap.read()
    ysize = frame.shape[0]
    xsize = frame.shape[1]
    # step1: grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # step2: define top left corner of starting scan block
    L_index = [0,ysize-150]
    R_index = [xsize-300, ysize-150]

    # reset some parameters
    leftblob = np.empty((20*15, 300))
    rightblob = np.empty((20*15, 300))
    scanwidthl = scanwidth
    scanwidthr = scanwidth

    # begin algo timing
    start_time = time.time()

####### main process loop
    # for loop controls how many blocks vertically are checked
    for x in range(0,10):

        # step3: grab the proper block of pixels for the current scan block
        leftscan = gray[L_index[1]:L_index[1]+scanheight , L_index[0]:L_index[0] + scanwidthl]
        rightscan = gray[R_index[1]:R_index[1]+scanheight , R_index[0]:R_index[0] + scanwidthr]
                # cv2.imshow("left", leftscan)
        # cv2.imshow("right", rightscan)

        if algotype == "convolution":
            # step4: run the correlation/eigenvalue/convolution thing
            left = scipy.ndimage.convolve(leftscan, block_left)
            right = scipy.ndimage.convolve(rightscan, block_right)

            # post process into a single intensity vector
            left = left[7][int(np.floor(blocksize/2)):int(scanwidthl-np.floor(blocksize/2))]
            right = right[7][int(np.floor(blocksize/2)):int(scanwidthr-np.floor(blocksize/2))]

            # copy for visualization
            np.copyto(leftblob[x*15:x*15+15, 0:left.shape[0]], left)
            np.copyto(rightblob[x*15:x*15+15, 0:right.shape[0]], right)

        elif algotype == "correlation":
            # step4: run the correlation/eigenvalue/convolution thing
            left = scipy.ndimage.correlate(leftscan, block_left)
            right = scipy.ndimage.correlate(rightscan, block_right)

            # post process into a single intensity vector
            left = left[7][int(np.floor(blocksize/2)):int(scanwidthl-np.floor(blocksize/2))]
            right = right[7][int(np.floor(blocksize/2)):int(scanwidthr-np.floor(blocksize/2))]

            # copy for visualization
            np.copyto(leftblob[x*15:x*15+15, 0:left.shape[0]], left)
            np.copyto(rightblob[x*15:x*15+15, 0:right.shape[0]], right)

        elif algotype == "eigenvalue":
            # step4: run the correlation/eigenvalue/convolution thing
            left = scipy.ndimage.correlate(leftscan, block_left)
            right = scipy.ndimage.correlate(rightscan, block_right)

            # post process into a single intensity vector
            left = left[7][int(np.floor(blocksize/2)):int(scanwidthl-np.floor(blocksize/2))]
            right = right[7][int(np.floor(blocksize/2)):int(scanwidthr-np.floor(blocksize/2))]

            # copy for visualization
            np.copyto(leftblob[x*15:x*15+15, 0:left.shape[0]], left)
            np.copyto(rightblob[x*15:x*15+15, 0:right.shape[0]], right)

        else: 
            print("NO ALGORITHM TYPE DEFINED")
            exit(1)



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
        # L_index and R_index are the top left point of the scan box.
        
        if True:
            # left and right at this point contain a line of values corresponding to all valid correlation overlaps
            # thus the index is the center of each block, which means within each scan block, the center of the max block is (idxl+7, 7)
            idxl = np.argmax(left)
            idxr = np.argmax(right)
            # print("left max:"+str(idxl))
            # print("right max:"+str(idxr))
            green = (0,255,0)
            red = (0,0,255)
            blue = (255,0,0)
            rect = 15
            # cv2.rectangle(right, (idxr, 0), (idxr + rect, rect), 255)
            # cv2.rectangle(left, (idxl, 0), (idxl + rect, rect), 255)

            # idxl-f stands for the index in the actual frame, this converts our idxl location to the correct pixel location on the full input
            idxlf = (halfblock + idxl + L_index[0], L_index[1])
            idxrf = (halfblock + idxr + R_index[0] , R_index[1])
            # print("left at frame loc:"+str(idxlf))
            # print("right at frame loc:"+str(idxrf))
            
            # draw the green scan box, and the red/blue locators
            cv2.rectangle(frame, tuple(L_index), (L_index[0] + scanwidthl, L_index[1] + scanheight), green, 2)
            cv2.rectangle(frame, tuple(R_index), (R_index[0] + scanwidthr, R_index[1] + scanheight), green, 2)
            cv2.rectangle(frame, idxlf, (idxlf[0] + rect, idxlf[1] + rect), red, 3)
            cv2.rectangle(frame, idxrf, (idxrf[0] + rect, idxrf[1] + rect), blue, 3)
            # move the bounding box to next position by scanspacing pixels
            if left[idxl] < 10:
                scanwidthl = 300
                # print("left poop")
                L_index = [L_index[0], L_index[1] - scanspacing]
            else:
                scanwidthl = 100
                L_index = [idxlf[0] - int(scanwidthl/2), idxlf[1]-scanspacing]

            if right[idxr] < 10:
                scanwidthr = 300
                # print("right poop")
                R_index = [R_index[0], R_index[1]-scanspacing]
            else:        
                scanwidthr = 100
                R_index = [idxrf[0] - int(scanwidthr/2), idxrf[1]-scanspacing]

            # R_index = [idxrf[0] - int(scanwidthr/2), idxrf[1]-15]
            # L_index = [idxlf[0] - int(scanwidthl/2), idxlf[1]-15]
            if L_index[0] < 0:
                L_index[0] = 0
            if R_index[0] > xsize-scanwidthr:
                R_index[0] = xsize-scanwidthr
        # except: 
            # print("failed to find line")
            # L_index = (idxlf[0], idxlf[1]+15)
            # R_index = (idxrf[0], idxrf[1]+15)
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