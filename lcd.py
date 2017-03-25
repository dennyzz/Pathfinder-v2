#!/usr/bin/python

from Adafruit_CharLCD import Adafruit_CharLCD
from subprocess import *
from time import sleep, strftime
from datetime import datetime

lcd = Adafruit_CharLCD(2,3,4,17,27,22, 16, 2)

cmd = "ip addr show wlan0 | grep -m 1 inet | awk '{print $2}' | cut -d/ -f1"


def run_cmd(cmd):
        p = Popen(cmd, shell=True, stdout=PIPE)
        output = p.communicate()[0]
        return output

while 1:
        lcd.home()
        ipaddr = run_cmd(cmd)
        #print datetime.now().strftime('%b %d  %H:%M:%S\n')
        #print ipaddr        
        lcd.message(datetime.now().strftime('%b %d  %H:%M:%S\n'))
        lcd.message('IP %s' % ( ipaddr ) )
        sleep(1)
