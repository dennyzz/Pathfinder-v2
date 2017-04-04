import pathfindershield
import sys
import time

# def motorservocmd4(Bspeed, Bdir, Bbrake, Servo2):

print('running around')
speed = 0
dir = 0
brake = 0
angle = 0

while True:
    if angle > 511:
        angle = 0
    elif angle > 255:
        print(511-angle)
        pathfindershield.motorservocmd4(speed, dir, brake, 511-angle)
    else:
        pathfindershield.motorservocmd4(speed, dir, brake, angle)
        print(angle)
    
    time.sleep(.01)
    angle += 1
