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
v0 = 6
cols = 8
lines = 2

gpio=GPIO.get_platform_gpio()
gpio.setup(rw, GPIO.OUT)
gpio.output(rw, False)
gpio.setup(v0, GPIO.OUT)
sleep(1)
lcd = Adafruit_CharLCD.Adafruit_CharLCD(rs, en, d4, d5, d6, d7, cols, lines)
lcd.clear()

cmd = "ip addr show wlan0 | grep -m 1 inet | awk '{print $2}' | cut -d/ -f1"


def run_cmd(cmd):
        p = Popen(cmd, shell=True, stdout=PIPE)
        output = p.communicate()[0]
        return output

while True:
        lcd.home()
        ipaddr = run_cmd(cmd)
        ips = string.split(ipaddr, '.')
        if len(ips) == 4:
                lcd.message('i%s.%s\n' % (ips[0], ips[1]) )
                lcd.message('p%s.%s' % (ips[2], ips[3]) )
        else:
                lcd.message('   no   \n')
                lcd.message('network!')
        sleep(1)
