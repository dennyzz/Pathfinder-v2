#!/usr/bin/python

import smbus
import sys
import subprocess

# Usage: 
# [0] status? blank
# [1] MOTORA_CTRL
# [2] MOTORA_SPEED
# [3] MOTORB_CTRL
# [4] MOTORB_SPEED
# [5] Servo1
# [6] Servo2
# [7] LEDs

def motorservocmd1(Aspeed, Adir, Abrake, Bspeed, Bdir, Bbrake, Servo1, Servo2):

    bus = smbus.SMBus(1)    # 0 = /dev/i2c-0 (port I2C0), 1 = /dev/i2c-1 (port I2C1)
    DEVICE_ADDRESS = 0x5A      #7 bit address (will be left shifted to add the read write bit)
    REG = 0x01

    CTRLA = 0
    CTRLB = 0
    if Adir:
        CTRLA |= 1<<0
    if Bdir:
        CTRLB |= 1<<0
    if Abrake:
        CTRLA |= 1<<1
    if Bbrake:
        CTRLB |= 1<<1

    assert Aspeed < 256 & Aspeed >=0
    assert Bspeed < 256 & Bspeed >=0
    assert Servo1 < 256 & Servo1 >=0
    assert Servo2 < 256 & Servo2 >=0

    cmd = {CTRLA, Aspeed, CTRLB, Bspeed, Servo1, Servo2}

    bus.write_i2c_block_data(DEVICE_ADDRESS, REG, cmd)


def motorservocmd2(Aspeed, Adir, Abrake, Bspeed, Bdir, Bbrake):

    bus = smbus.SMBus(1)    # 0 = /dev/i2c-0 (port I2C0), 1 = /dev/i2c-1 (port I2C1)
    DEVICE_ADDRESS = 0x5A      #7 bit address (will be left shifted to add the read write bit)
    REG = 0x01

    CTRLA = 0
    CTRLB = 0
    if Adir:
        CTRLA |= 1<<0
    if Bdir:
        CTRLB |= 1<<0
    if Abrake:
        CTRLA |= 1<<1
    if Bbrake:
        CTRLB |= 1<<1

    assert Aspeed < 256 & Aspeed >=0
    assert Bspeed < 256 & Bspeed >=0

    cmd = {CTRLA, Aspeed, CTRLB, Bspeed}

    bus.write_i2c_block_data(DEVICE_ADDRESS, REG, cmd)


def motorservocmd3(Aspeed, Adir, Abrake, Servo1):

    bus = smbus.SMBus(1)    # 0 = /dev/i2c-0 (port I2C0), 1 = /dev/i2c-1 (port I2C1)
    DEVICE_ADDRESS = 0x5A      #7 bit address (will be left shifted to add the read write bit)
    REG = 0x01
    REG_SERVO1 = 0x05
    CTRLA = 0
    if Adir:
        CTRLA |= 1<<0
    if Abrake:
        CTRLA |= 1<<1

    assert Aspeed < 256 & Aspeed >=0
    assert Servo1 < 256 & Servo1 >=0

    cmd1 = [CTRLA, Aspeed]
    cmd2 = Servo

    bus.write_i2c_block_data(DEVICE_ADDRESS, REG, cmd1)
    bus.write_byte_data(DEVICE_ADDRESS, REG_SERVO1, cmd2)

def motorservocmd4(Bspeed, Bdir, Bbrake, Servo2):

    bus = smbus.SMBus(1)    # 0 = /dev/i2c-0 (port I2C0), 1 = /dev/i2c-1 (port I2C1)
    DEVICE_ADDRESS = 0x5A      #7 bit address (will be left shifted to add the read write bit)
    REG_MOTB = 0x03
    REG_SERVO2 = 0x05

    CTRLB = 0
    if Bdir:
        CTRLB |= 1<<0
    if Bbrake:
        CTRLB |= 1<<1

    #assert (Bspeed < 256 & Bspeed >=0)
    #assert (Servo2 < 256 & Servo2 >=0)

    cmd1 = [CTRLB, Bspeed]
    cmd2 = Servo2

    try:
        bus.write_i2c_block_data(DEVICE_ADDRESS, REG_MOTB, cmd1)
        bus.write_byte_data(DEVICE_ADDRESS, REG_SERVO2, cmd2)
    except IOError:
        subprocess.call(['i2cdetect', '-y', '1'])
        flag = 1     #optional flag to signal your code to resend or something

def motorservoledcmd(leds):
    bus = smbus.SMBus(1)    # 0 = /dev/i2c-0 (port I2C0), 1 = /dev/i2c-1 (port I2C1)
    DEVICE_ADDRESS = 0x5A      #7 bit address (will be left shifted to add the read write bit)
    REG_LED = 0x07

    bus.write_byte_data(DEVICE_ADDRESS, REG_LED, leds)
