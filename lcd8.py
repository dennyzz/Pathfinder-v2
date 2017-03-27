#!/usr/bin/python

from subprocess import *
from time import sleep, strftime
from datetime import datetime
import Adafruit_CharLCD
import time
import string

lcd = Adafruit_CharLCD(2,14,3,4,17,27,22,16,2)
lcd.clear()

cmd = "ip addr show wlan0 | grep -m 1 inet | awk '{print $2}' | cut -d/ -f1"


def run_cmd(cmd):
        p = Popen(cmd, shell=True, stdout=PIPE)
        output = p.communicate()[0]
        return output

while 1:
        lcd.home()
        ipaddr = run_cmd(cmd)
        ips = string.split(ipaddr, '.')
        
        lcd.message('%s.%s     \n' % (ips[0], ips[1]) )
        lcd.message('%s.%s     \n' % (ips[2], ips[3]) )
        sleep(1)
