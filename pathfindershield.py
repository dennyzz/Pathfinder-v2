#!/usr/bin/python

#define PCA9685_MODE1 0x0
#define PCA9685_PRESCALE 0xFE

import smbus
import sys


# Usage: 


def motorservocmd1(Aspeed, Adir, Abrake, Bspeed, Bdir, Bbrake, Servo1, Servo2):

    bus = smbus.SMBus(1)    # 0 = /dev/i2c-0 (port I2C0), 1 = /dev/i2c-1 (port I2C1)
    DEVICE_ADDRESS = 0x5A      #7 bit address (will be left shifted to add the read write bit)
    MOTOR_CMD_1 = 0x01

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

    bus.write_i2c_block_data(DEVICE_ADDRESS, MOTOR_CMD_1, cmd)


def motorservocmd2(Aspeed, Adir, Abrake, Bspeed, Bdir, Bbrake):

    bus = smbus.SMBus(1)    # 0 = /dev/i2c-0 (port I2C0), 1 = /dev/i2c-1 (port I2C1)
    DEVICE_ADDRESS = 0x5A      #7 bit address (will be left shifted to add the read write bit)
    MOTOR_CMD_2 = 0x02

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

    bus.write_i2c_block_data(DEVICE_ADDRESS, MOTOR_CMD_2, cmd)


def motorservocmd3(Aspeed, Adir, Abrake, Servo1):

    bus = smbus.SMBus(1)    # 0 = /dev/i2c-0 (port I2C0), 1 = /dev/i2c-1 (port I2C1)
    DEVICE_ADDRESS = 0x5A      #7 bit address (will be left shifted to add the read write bit)
    MOTOR_CMD_3 = 0x03

    CTRLA = 0
    if Adir:
        CTRLA |= 1<<0
    if Abrake:
        CTRLA |= 1<<1

    assert Aspeed < 256 & Aspeed >=0
    assert Servo1 < 256 & Servo1 >=0

    cmd = {CTRLA, Aspeed, Servo1}

    bus.write_i2c_block_data(DEVICE_ADDRESS, MOTOR_CMD_3, cmd)

def motorservocmd4(Bspeed, Bdir, Bbrake, Servo2):

    bus = smbus.SMBus(1)    # 0 = /dev/i2c-0 (port I2C0), 1 = /dev/i2c-1 (port I2C1)
    DEVICE_ADDRESS = 0x5A      #7 bit address (will be left shifted to add the read write bit)
    MOTOR_CMD_4 = 0x04

    CTRLB = 0
    if Bdir:
        CTRLB |= 1<<0
    if Bbrake:
        CTRLB |= 1<<1

    assert Bspeed < 256 & Bspeed >=0
    assert Servo2 < 256 & Servo2 >=0

    cmd = {CTRLB, Bspeed, Servo2}

    bus.write_i2c_block_data(DEVICE_ADDRESS, MOTOR_CMD_4, cmd)


def motorservoledcmd(leds):
    bus = smbus.SMBus(1)    # 0 = /dev/i2c-0 (port I2C0), 1 = /dev/i2c-1 (port I2C1)
    DEVICE_ADDRESS = 0x5A      #7 bit address (will be left shifted to add the read write bit)
    LED_CMD = 0x10

    bus.write_byte_data(DEVICE_ADDRESS, LED_CMD, leds)
