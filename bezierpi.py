# Instructions!
# scipy: the easiest install on windows is to use prebuilt wheels.
#   pip install wheel
#   then go to http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
#   and download numpy+mkl and scipy
#   pip install those files


#import the necessary packages
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import cv2
import numpy as np
import os
import scipy.signal
import sys
import pathfindershield

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False


'''least square qbezier fit using penrose pseudoinverse
    >>> V=array
    >>> E,  W,  N,  S =  V((1,0)), V((-1,0)), V((0,1)), V((0,-1))
    >>> cw = 100
    >>> ch = 300
    >>> cpb = V((0, 0))
    >>> cpe = V((cw, 0))
    >>> xys=[cpb,cpb+ch*N+E*cw/8,cpe+ch*N+E*cw/8, cpe]            
    >>> 
    >>> ts = V(range(11))/10
    >>> M = bezierM (ts)
    >>> points = M*xys #produces the points on the bezier curve at t in ts
    >>> 
    >>> control_points=lsqfit(points, M)
    >>> linalg.norm(control_points-xys)<10e-5
    True
    >>> control_points.tolist()[1]
    [12.500000000000037, 300.00000000000017]

'''
from numpy import array, linalg, matrix
from scipy.misc import comb as nOk
Mtk = lambda n, t, k: t**(k)*(1-t)**(n-k)*nOk(n,k)
bezierM = lambda ts: matrix([[Mtk(3,t,k) for k in range(4)] for t in ts])
def lsqfit(points,M):
    M_ = linalg.pinv(M)
    return M_ * points


# cap = cv2.VideoCapture("footage/radius2angle75.mp4")
# cap = cv2.VideoCapture("footage/0degree.mp4 ")
# cap = cv2.VideoCapture("footage/rootbeercar.mp4 ")
# fps = cap.get(cv2.CAP_PROP_FPS)

# print(fps)
w = 1/20
b = -1/20
smooth_time = 0
block_5_left = np.array([
[b,b,b,b,b],
[b,b,b,b,w], 
[b,b,b,w,w], 
[b,b,w,w,w], 
[b,w,w,w,w]
])

block_5_right = np.array([
[b,b,b,b,b], 
[w,b,b,b,b], 
[w,w,b,b,b], 
[w,w,w,b,b], 
[w,w,w,w,b]
])

block_5_left_flip = np.array([
[b,w,w,w,w],
[b,b,w,w,w], 
[b,b,b,w,w], 
[b,b,b,b,w], 
[b,b,b,b,b]
])

block_5_right_flip = np.array([
[w,w,w,w,b],
[w,w,w,b,b], 
[w,w,b,b,b], 
[w,b,b,b,b], 
[b,b,b,b,b] 
])
 
# BLOCK CONFIGURATION
block_left = block_5_left
block_right = block_5_right 
block_left_flip = block_5_left_flip
block_right_flip = block_5_right_flip
blocksize = 5
halfblock = int(np.floor(blocksize/2))
### END BLOCK CONFIG ###
### MOST GLOBAL TUNING PARAMETERS ###

# width of the initial scan block
scanwidth = 100
# offset pixels inwards (x) for the initial scan block
scanoffset = 25
# width of the scan block when a valid point has been found previously (smaller)
scanwidthmin = 30
# height of the scan block
scanheight = 5
# space between scan blocks
scanspacing = 0
# total number of scan lines vertically
scanlines = 18
# pixels from the bottom that the scanlines first index starts from
scanstartline = 45
# the threshold for detection for post correlation
threshold = 1

### END GLOBAL TUNING PARAMETERS ###

# Colors!
green = (0,255,0)
red = (0,0,255)
blue = (255,0,0)
yellow = (0,255,255)
orange = (51, 153, 255)
# lane points saved into an array with a count variable 
laneleft = np.empty((scanlines,2), dtype = np.int32)
laneright= np.empty((scanlines,2), dtype = np.int32)
laneleftcount = 0
lanerightcount = 0


# # initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(320, 240))

# # allow the camera to warmup
time.sleep(0.1)

# initialize the VL53L0x
tof.start_ranging(VL53L0X.VL53L0X_BETTER_ACCURACY_MODE)


# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True, resize=(320,240)):
    # grab the raw NumPy array representing the image,
    start_time = time.time()

    frame = frame.array
    ysize = frame.shape[0]
    xsize = frame.shape[1]

    # maybe a speedup if we clear the stream here...?
    rawCapture.truncate(0)

    # step1: grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # step2: define top left corner of starting scan block
    L_index = [scanoffset, ysize - scanstartline]
    R_index = [xsize - scanwidth - scanoffset, ysize - scanstartline]

    # reset some parameters
    leftblob = np.empty((scanlines*blocksize, scanwidth-blocksize+1))
    rightblob = np.empty((scanlines*blocksize, scanwidth-blocksize+1))
    scanwidthl = scanwidth
    scanwidthr = scanwidth
    laneleftcount = 0
    lanerightcount = 0

    # begin algo timing

####### main process loop
    # for loop controls how many blocks vertically are checked
    for x in range(0,scanlines):

        # step3: grab the proper block of pixels for the current scan block
        leftscan = gray[L_index[1]:L_index[1]+scanheight , L_index[0]:L_index[0] + scanwidthl]
        rightscan = gray[R_index[1]:R_index[1]+scanheight , R_index[0]:R_index[0] + scanwidthr]
        # cv2.imshow("left", leftscan)
        # cv2.imshow("right", rightscan)

        # step4: run the correlation/eigenvalue/convolution thing
        left = scipy.signal.correlate2d(leftscan, block_left, mode='valid')[0]
        right = scipy.signal.correlate2d(rightscan, block_right, mode='valid')[0]

        # step 4.5 if it returns nothing of adequate similarity, try the reversed masks
        if max(left) < threshold:
            left = scipy.signal.correlate2d(leftscan, block_left_flip, mode='valid')[0]
        if max(right) < threshold:
            right = scipy.signal.correlate2d(rightscan, block_right_flip, mode='valid')[0]

        # f.write('leftmax:' + str(np.max(left)) + ' ' + str(np.min(left)) + '\n')
        # f.write('rightmax:' + str(np.max(right)) + ' ' + str(np.min(right)) + '\n')
        # copy for visualization
        np.copyto(leftblob[(scanlines-x-1)*15:(scanlines-x)*15, 0:left.shape[0]], left)
        np.copyto(rightblob[(scanlines-x-1)*15:(scanlines-x)*15, 0:right.shape[0]], right)

        # so idxL/R is the index of the max thing, or the best boundary location as an x offset from the scan box width
        # idxLRf is the location of the box in the frame
        # L_index and R_index are the top left point of the scan box.
        
        if True:
            # left and right at this point contain a line of values corresponding to all valid correlation overlaps
            # thus the index is the center of each block, which means within each scan block, the center of the max block is (idxl+7, 7)
            idxl = np.argmax(left)
            idxr = np.argmax(right)

            # idxl-f stands for the index in the actual frame, this converts our idxl location to the correct pixel location on the full input
            idxlf = (halfblock + idxl + L_index[0], L_index[1] + halfblock)
            idxrf = (halfblock + idxr + R_index[0] , R_index[1] + halfblock)
            # print("left at frame loc:"+str(idxlf))
            # print("right at frame loc:"+str(idxrf))
            
            # draw the green scan box, and the red/blue locators
            cv2.rectangle(frame, tuple(L_index), (L_index[0] + scanwidthl, L_index[1] + scanheight-1), green, 1)
            cv2.rectangle(frame, tuple(R_index), (R_index[0] + scanwidthr, R_index[1] + scanheight-1), green, 1)

            # move the bounding box to next position by scanheight + scanspacing pixels
            if left[idxl] < threshold:
                # if cannot find lane line
                if scanwidthl == scanwidthmin: # if from good to failing
                    L_index[0] = int(L_index[0] - ((scanwidth - scanwidthmin) / 2))
                cv2.rectangle(frame, (idxlf[0]-halfblock, idxlf[1]-halfblock), (idxlf[0]+halfblock, idxlf[1]+halfblock), yellow, 2)
                scanwidthl = scanwidth
                # print("left BAD")
                L_index = [L_index[0], L_index[1] - scanspacing - scanheight]
            else:
                laneleft[laneleftcount] = idxlf
                laneleftcount += 1
                cv2.rectangle(frame, (idxlf[0]-halfblock, idxlf[1]-halfblock), (idxlf[0]+halfblock, idxlf[1]+halfblock), red, 1)
                scanwidthl = scanwidthmin
                L_index = [idxlf[0] - int(scanwidthl/2), idxlf[1] - halfblock - scanspacing - scanheight]

            if right[idxr] < threshold:
                cv2.rectangle(frame, (idxrf[0]-halfblock, idxrf[1]-halfblock), (idxrf[0]+halfblock, idxrf[1]+halfblock), yellow, 1)
                scanwidthr = scanwidth
                # print("right BAD")
                R_index = [R_index[0], R_index[1] - scanspacing - scanheight]    
            else:
                laneright[lanerightcount] = idxrf
                lanerightcount += 1
                cv2.rectangle(frame, (idxrf[0]-halfblock, idxrf[1]-halfblock), (idxrf[0]+halfblock, idxrf[1]+halfblock), blue, 1)
                scanwidthr = scanwidthmin
                R_index = [idxrf[0] - int(scanwidthr/2), idxrf[1] - halfblock - scanspacing - scanheight]

            if L_index[0] < 0:
                L_index[0] = 0
            if R_index[0] > xsize-scanwidthr:
                R_index[0] = xsize-scanwidthr

    ####### end processing

    leftblob = np.multiply(leftblob, 0.1)
    rightblob = np.multiply(rightblob, 0.1)
    leftangle = 0
    rightangle = 0
    leftx = xsize/2
    rightx = xsize/2
    if(laneleftcount > 4):
        L1 = line(laneleft[0], laneleft[1])
        L2 = line(laneleft[laneleftcount-1], laneleft[laneleftcount-2])

        R = intersection(L1, L2)
        if R:
            R = (int(R[0]), int(R[1]))
            cv2.line(frame, (laneleft[0][0], laneleft[0][1]), R, orange, 1)
            cv2.line(frame, (laneleft[laneleftcount-1][0], laneleft[laneleftcount-1][1]), R, orange, 1)
            # print ("Intersection detected:", R)
        #else:
            # print ("leftside: No single intersection point detected")

        #semi jank all linear approx.
        a = laneleft[0]
        b = laneleft[laneleftcount-1]
        imagine = (a[0]-b[0]) + 1j*(a[1] - b[1])
        leftangle = np.angle(imagine, deg=True)
        leftx = leftx - laneleft[0][0]        
    if(lanerightcount > 4):
        L1 = line(laneright[0], laneright[1])
        L2 = line(laneright[lanerightcount-1], laneright[lanerightcount-2])

        R = intersection(L1, L2)
        if R:
            R = (int(R[0]), int(R[1]))
            cv2.line(frame, (laneright[0][0], laneright[0][1]), R, orange, 1)
            cv2.line(frame, (laneright[lanerightcount-1][0], laneright[lanerightcount-1][1]), R, orange, 1)
            # print ("Intersection detected:", R)
        #else:
            # print ("right side: No single intersection point detected")
        a = laneright[0]
        b = laneright[1]
        imagine = (a[0]-b[0]) + 1j*(a[1] - b[1])
        rightangle = np.angle(imagine, deg=True)
        rightx = laneright[0][0] - rightx
    cv2.imshow('frame', frame)
    #cv2.imshow('left', leftblob)
    #cv2.imshow('right', rightblob)

    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    # try moving the stream clear
    # rawCapture.truncate(0)

    #if the `q` key was pressed, break from the loop
    if key == ord("n"):
        print("next")
        next = 1
    if key == ord("q"):
        break

    distance = tof.get_distance()

    offset = leftx - rightx
    angle = 132 - int(((leftangle + rightangle)/2)-90)*3 + int(offset/2)
    if distance < 50:
        pathfindershield.motorservocmd4(0,0,0,angle)
    else:
        pathfindershield.motorservocmd4(70, 0, 0, angle)

    proc_time = time.time() - start_time
    if smooth_time == 0:
        smooth_time = proc_time
    else:
        smooth_time = 0.95*smooth_time + 0.05*proc_time
    fps_calc = int(1/smooth_time) 
    sys.stdout.write("\rtime: %f, frames: %d  offset: %d  left:%.2fdeg right:%.2fdeg outangle:%d mm:%d       " % (smooth_time, fps_calc, offset, leftangle, rightangle, angle, distance))
    sys.stdout.flush()
    #time it from here



cap.release()
sys.exit(0)
