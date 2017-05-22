import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    ysize = frame.shape[0]
    xsize = frame.shape[1]

    left = frame[0:ysize, 0:int(xsize/2)]
    right = frame[0:ysize, int(xsize/2):xsize]
    # frame = cv2.imread('road.jpg')

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayright = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    grayleft = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    edgesl = cv2.Canny(grayleft,50,150,apertureSize = 3)
    edgesr = cv2.Canny(grayright,50,150,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 10
    linesl = cv2.HoughLines(edgesl,1,np.pi/180,100)
    linesr = cv2.HoughLines(edgesr,1,np.pi/180,100)
    # print(lines)
    # if lines != None:
    # for line in lines:
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
    cv2.imshow('image',frame)
    cv2.imshow('cannyl',edgesl)
    cv2.imshow('cannyr',edgesr)
    cv2.imshow('left', left)
    cv2.imshow('right', right)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()