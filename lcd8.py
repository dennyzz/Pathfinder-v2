#!/usr/bin/python

from subprocess import *
from time import sleep, strftime
from datetime import datetime
import Adafruit_CharLCD
import Adafruit_GPIO as GPIO
import time
import string

rs = 26
rw = 19
en = 13
d4 = 12
d5 = 16
d6 = 20
d7 = 21
cols = 8
lines = 2

lcd = Adafruit_CharLCD(rs, en, d4, d5, d6, d7, cols, lines)
gpio=GPIO.get_platform_gpio()
gpio.setup(rw, GPIO.OUT)
gpio.output(rw, False)
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
