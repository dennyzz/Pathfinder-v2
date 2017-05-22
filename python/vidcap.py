#!/bin/python

import time
import picamera as cam


camera = cam.PiCamera();

camera.start_preview();
time.sleep(5);
camera.stop_preview();
