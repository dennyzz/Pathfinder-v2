# Instructions!
# scipy: the easiest install on windows is to use prebuilt wheels.
#   pip install wheel
#   then go to http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
#   and download numpy+mkl and scipy
#   pip install those files


#import the necessary packages
from picamera import PiCamera
from picamera.array import PiRGBArray
from scipy.optimize import curve_fit
import time
import cv2
import numpy as np
import os
import scipy.signal
import sys
import pathfindershield
import VL53L0X
import PID
import threading

exit = 0
#image resolution values
res_x = 320
res_y = 240
xsize = res_x
ysize = res_y


# TUNING PARAMETERS FOR OUTPUT STAGE

# turn off the output and drive commands
output = 1
# Distance for collision detection
stopdistance = 100
# Servo value for approximate middle value
servo_center = 132


def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def d_quadratic(x, a, b, c):
    return 2*a*x + b

def cubic(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def quartic(x, a, b, c, d, e):
    return a*x*x*x*x + b*x*x*x + c*x*x + d*x + e

def exponential(x, a, b):
    return a**x + b


def Thread_Capture(buffer, flag, buff_lock):
    global exit, res_x, res_y
    proc_time_s = 0
    camera = PiCamera()
    camera.resolution = (res_x, res_y)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(res_x, res_y))

    # allow the camera to warmup
    time.sleep(0.1)
    # start_cap_time = time.time()
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True, resize=(res_x, res_y)):
        buff_lock.acquire()
        buffer[:] = frame.array
        buff_lock.release()
    
        rawCapture.truncate(0)
        flag.set()
        if exit:
            break
        # proc_time = (time.time() - start_cap_time)*1000
        # if proc_time_s == 0:
        #     proc_time_s = proc_time
        # else:
        #     proc_time_s = 0.9*proc_time_s + 0.1*proc_time
        # fps_calc = 1000/proc_time_s
        # sys.stdout.write("\rtime:%dmS, fps:%d       " % (proc_time, fps_calc))
        # sys.stdout.flush()
        # start_cap_time = time.time()
    print("Capture Thread Terminated")


def Thread_Distance(flag, ret):
    # initialize the VL53L0x
    proc_time_s = 0
    tof = VL53L0X.VL53L0X()
    tof.start_ranging(VL53L0X.VL53L0X_BETTER_ACCURACY_MODE)
    start_cap_time = time.time()
    while not exit:
        distance = tof.get_distance()
        ret[0] = distance
        flag.set()
        # proc_time = (time.time() - start_cap_time)*1000
        # if proc_time_s == 0:
        #     proc_time_s = proc_time
        # else:
        #     proc_time_s = 0.9*proc_time_s + 0.1*proc_time
        # fps_calc = 1000/proc_time_s
        # sys.stdout.write("\rtime:%dmS, fps:%d dist:%d     " % (proc_time, fps_calc, distance))
        # sys.stdout.flush()
        # start_cap_time = time.time()
    print("Distance Thread Terminated")


def Thread_Process(buffer, flag, out_flag, buff_lock, outlist):
    global exit, res_x, res_y
    w = 1/20
    b = -1/20
    frame = np.empty((res_y, res_x, 3), dtype=np.uint8)
    # smooth_time = 0
    # proc_algo_time_s = 0
    # proc_post_time_s = 0
    # proc_pre_time_s = 0
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
    scanwidth = 110
    # width of the scan block when a valid point has been found previously (smaller)
    scanwidthmin = 30
    # height of the scan block
    scanheight = 5
    # space between scan blocks
    scanspacing = 0
    # total number of scan lines vertically
    scanlines = 15
    # offset pixels inwards (x) for the initial scan block
    scanstartoffset = 15
    # pixels from the bottom that the scanlines first index starts from
    scanstartline = 50
    # scan angle distance is the distance from scanstartline that we take our angle measurement from. 
    # this gives us some amount of "predictive" driving allowing us to see the curve ahead a bit
    scanangledistance = 50
    # the threshold for detection for post correlation
    threshold = 1
    # value for minimum number of good edges detected for curve fitting 
    min_data_good = 6

    ### END PROCESS TUNING PARAMETERS ###

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

    # angle and offset datas used for course correction
    leftangle = 0
    rightangle = 0
    leftx = xsize/2
    rightx = xsize/2

    # # initialize the camera and grab a reference to the raw camera capture

    while not exit:
        flag.wait()
        flag.clear()
        buff_lock.acquire()
        #print("process buffer locked")
        frame[:] = buffer
        buff_lock.release()
        #print("process buffer unlocked")

        # step1: grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # step2: define top left corner of starting scan block
        L_index = [scanstartoffset, ysize - scanstartline]
        R_index = [xsize - scanwidth - scanstartoffset, ysize - scanstartline]

        # reset some parameters
        # leftblob = np.empty((scanlines*blocksize, scanwidth-blocksize+1))
        # rightblob = np.empty((scanlines*blocksize, scanwidth-blocksize+1))
        scanwidthl = scanwidth
        scanwidthr = scanwidth
        laneleftcount = 0
        lanerightcount = 0

        # begin algo timing
        # proc_pre_time = (time.time() - start_pre_time) * 1000
        # start_algo_time = time.time()
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
            # np.copyto(leftblob[(scanlines-x-1)*15:(scanlines-x)*15, 0:left.shape[0]], left)
            # np.copyto(rightblob[(scanlines-x-1)*15:(scanlines-x)*15, 0:right.shape[0]], right)

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
                    cv2.rectangle(frame, (idxlf[0]-halfblock, idxlf[1]-halfblock), (idxlf[0]+halfblock, idxlf[1]+halfblock), yellow, 1)
                    scanwidthl = scanwidth
                    # print("left BAD")
                    L_index = [L_index[0], L_index[1] - scanspacing - scanheight]
                else:
                    laneleft[laneleftcount] = idxlf
                    laneleftcount += 1
                    cv2.rectangle(frame, (idxlf[0]-halfblock, idxlf[1]-halfblock), (idxlf[0]+halfblock, idxlf[1]+halfblock), red, 1)
                    scanwidthl = scanwidthmin
                    # scanblock delta calculation for the steeper curves
                    if laneleftcount >= 2:
                        delta = laneleft[laneleftcount-1][0] - laneleft[laneleftcount-2][0]
                    else:
                        delta = 0
                    # L_index = [idxlf[0] - int(scanwidthl/2) + int(delta/2), idxlf[1] - halfblock - scanspacing - scanheight]
                    L_index = [idxlf[0] - int(scanwidthl/2) + delta, idxlf[1] - halfblock - scanspacing - scanheight]
                if right[idxr] < threshold:
                    cv2.rectangle(frame, (idxrf[0]-halfblock, idxrf[1]-halfblock), (idxrf[0]+halfblock, idxrf[1]+halfblock), yellow, 1)
                    scanwidthr = scanwidth
                    R_index = [R_index[0], R_index[1] - scanspacing - scanheight]    
                else:
                    laneright[lanerightcount] = idxrf
                    lanerightcount += 1
                    cv2.rectangle(frame, (idxrf[0]-halfblock, idxrf[1]-halfblock), (idxrf[0]+halfblock, idxrf[1]+halfblock), blue, 1)
                    scanwidthr = scanwidthmin
                    # scanblock delta calculation for the steeper curves
                    if lanerightcount >= 2:
                        delta = laneright[lanerightcount-1][0] - laneright[lanerightcount-2][0]
                    else:
                        delta = 0
                    # R_index = [idxrf[0] - int(scanwidthr/2) + int(delta/2), idxrf[1] - halfblock - scanspacing - scanheight]
                    R_index = [idxrf[0] - int(scanwidthr/2) + delta, idxrf[1] - halfblock - scanspacing - scanheight]

                if L_index[0] < 0:
                    L_index[0] = 0
                elif L_index[0] > xsize-scanwidthr:
                    L_index[0] = xsize-scanwidthr
                if R_index[0] > xsize-scanwidthr:
                    R_index[0] = xsize-scanwidthr
                elif R_index[0] < 0:
                    R_index[0] = 0
        ####### end processing
        
        # leftblob = np.multiply(leftblob, 0.1)
        # rightblob = np.multiply(rightblob, 0.1)


        goodcheck = 0x31
        if(laneleftcount > min_data_good):
            # flip the axes to get a real function
            x = laneleft[0:laneleftcount, 1]
            y = laneleft[0:laneleftcount, 0]
            popt, pcov = curve_fit(quadratic, x, y)

            prevpoint = (int(quadratic(0, popt[0], popt[1], popt[2])), 0)
            for y in range(10, ysize, 10):
                x = int(quadratic(y, popt[0], popt[1], popt[2]))
                cv2.line(frame,prevpoint,(x,y),orange,2)
                prevpoint = (x,y)

            # offset computed from curve fit at scan start location
            leftx = xsize/2 - quadratic(ysize-scanstartline, popt[0], popt[1], popt[2])
            # angle computed from tangent of curve fit at scan start location
            slope = d_quadratic(ysize-scanstartline-scanangledistance, popt[0], popt[1], popt[2])
            rads = np.arctan(slope)
            leftangle = rads/np.pi*180

            goodcheck &= ~0x01
            
        if(lanerightcount > min_data_good):
            # popt, pcov = curve_fit(quadratic, x, y)
            x = laneright[0:lanerightcount, 1]
            y = laneright[0:lanerightcount, 0]
            popt, pcov = curve_fit(quadratic, x, y)
            x = 0
            y = quadratic(0, popt[0], popt[1], popt[2])
            prevpoint = (int(quadratic(0, popt[0], popt[1], popt[2])), 0)
            for y in range(10, ysize, 10):
                x = int(quadratic(y, popt[0], popt[1], popt[2]))
                cv2.line(frame,prevpoint,(x,y),orange,2)
                prevpoint = (x,y)

            # offset computed from curve fit at scan start location
            rightx = quadratic(ysize-scanstartline, popt[0], popt[1], popt[2]) - xsize/2
            # angle computed from tangent of curve fit at scan start location
            slope = d_quadratic(ysize-scanstartline-scanangledistance, popt[0], popt[1], popt[2])
            rads = np.arctan(slope)
            rightangle = rads/np.pi*180

            goodcheck &= ~0x10
        
        offseterror = leftx - rightx 
        angleerror = ((leftangle + rightangle)/2)

        outlist[0] = offseterror
        outlist[1] = angleerror
        outlist[2] = goodcheck
        out_flag.set()

        cv2.imshow('frame', frame)
        # cv2.imshow('left', leftblob)
        # cv2.imshow('right', rightblob)

        key = cv2.waitKey(1) & 0xFF

        #if the `q` key was pressed, break from the loop
        if key == ord("n"):
            print("next")
            next = 1
        if key == ord("q"):
            exit = 1
            output_ready.set() # allow the main loop to proceed to exit
            break

    print("Process Thread Terminated")






# main is the output task!

# try file writing and plotting error
f = open("log.csv", 'w')
f.write("Error log data for Pathfinder\r\n")
f.write("time, t_taken, offerror, angleerror, offsetpid, anglepid, output\r\n")

proc_time_s = 0

PIDoffset = PID.PID(0.4, 0.0, 1.0)
PIDangle = PID.PID(1.0, 0.0, 1.0)

img_buf = np.empty((res_y, res_x, 3), dtype=np.uint8)

image_ready = threading.Event()
distance_ready = threading.Event()
output_ready = threading.Event()
image_buffer_lock = threading.Lock()

ret_dist = [2000]
error_list = [0,0,0]

# start threads
Capture_Thread = threading.Thread(target=Thread_Capture, args=(img_buf, image_ready, image_buffer_lock))
Process_Thread = threading.Thread(target=Thread_Process, args=(img_buf, image_ready, output_ready, image_buffer_lock, error_list))
Distance_Thread = threading.Thread(target=Thread_Distance, args=(distance_ready, ret_dist))
print("threads created")

Process_Thread.start()
Capture_Thread.start()
Distance_Thread.start()
print("threads started")
start_time = time.time()
while not exit:
    output_ready.wait()
    output_ready.clear()
    offseterror = error_list[0]
    angleerror = error_list[1]
    #offset error in pixels from center screen +means turn left to correct
    offset_adj = PIDoffset.update_error(offseterror);
    #angle error in degrees from vertical +means turn left to correct
    angle_adj = PIDangle.update_error(angleerror);

    servocmd = int(servo_center + offset_adj + angle_adj)
    if servocmd > 255:
       servocmd = 255
    elif servocmd < 0:
       servocmd = 0
       
    distance = ret_dist[0]
    leds = error_list[2]

    if output:
       if distance < stopdistance:
           pathfindershield.motorservocmd4(0,0,1,servo_center)
           leds |= 0xFF
       else:
           pathfindershield.motorservocmd4(55, 0, 0, servocmd)

    pathfindershield.motorservoledcmd(leds)

    proc_time = (time.time() - start_time)*1000
    f.write("%d, %d, %d, %d, %.2f, %.2f, %d\r\n" % (time.time()*1000, proc_time, offseterror, angleerror, offset_adj, angle_adj, servocmd))
    if proc_time_s == 0:
        proc_time_s = proc_time
    else:
        proc_time_s = 0.9*proc_time_s + 0.1*proc_time
    fps_calc = int(1000/proc_time_s)
    # sys.stdout.write("\rtimetot:%dmS fps:%d algotime:%dmS posttime:%dmS pretime:%dmS       " %(smooth_time, fps_calc, proc_algo_time_s, proc_post_time_s, proc_pre_time_s))
    sys.stdout.write("\rtime:%dmS,%dfps,d:%d       " % (proc_time_s, fps_calc, distance))
    sys.stdout.flush()    
    start_time = time.time()
    

Process_Thread.join()
Capture_Thread.join()
Distance_Thread.join()

print("all threads terminated, exited normally")

sys.exit(0)





