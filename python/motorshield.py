#!/usr/bin/python

#define PCA9685_MODE1 0x0
#define PCA9685_PRESCALE 0xFE

import smbus
import sys


# Usage: 
# Select is motor 1 or 2 , none other are implemented
# Speed is a value from 0 to 4096 corresponding to 0 to full scale pwm
# dir is the direction of rotation, the motors will need to be opposites to move forwards
# brake is if 1 then brake, and ignore all else

def motorcmd(select, speed, dir, brake):
    # motor select 1 and 2 are M1 M2 channels
    bus = smbus.SMBus(1)    # 0 = /dev/i2c-0 (port I2C0), 1 = /dev/i2c-1 (port I2C1)
    DEVICE_ADDRESS = 0x60      #7 bit address (will be left shifted to add the read write bit)
    DEVICE_REG_LED_BASE = 0x06
    DEVICE_REG_MOT1PWM = 0x06+8*4
    DEVICE_REG_MOT1A = 0x06+10*4
    DEVICE_REG_MOT1B = 0x06+9*4
    DEVICE_REG_MOT2PWM = 0x06+13*4
    DEVICE_REG_MOT2A = 0x06+11*4
    DEVICE_REG_MOT2B = 0x06+12*4

    PCA9685_MODE1 = 0x0
    PCA9685_PRESCALE = 0xFE
    PCA_MODE = 0x21
    bus.write_byte_data(DEVICE_ADDRESS, PCA9685_MODE1, PCA_MODE | 0x10)
    bus.write_byte_data(DEVICE_ADDRESS, PCA9685_PRESCALE, 0x64) #54Hz ish
    bus.write_byte_data(DEVICE_ADDRESS, PCA9685_PRESCALE, 0x0) #54Hz ish
    bus.write_byte_data(DEVICE_ADDRESS, PCA9685_MODE1, PCA_MODE)
    pwm_on = [0x00, 0x10, 0xFF, 0x0F]
    pwm_off = [0x00, 0x00, 0xFF, 0x1F]
    bitspeed = speed & 0xFFF
    pwm_speed = [bitspeed & 0xFF, (bitspeed >> 8)&0x0F, 0xFF, 0x0F]
    #Write a single register
    # bus.write_byte_data(DEVICE_ADDRESS, DEVICE_REG_MODE1, 0x80)

    #Write an array of
    if (select == 1):
        motA = DEVICE_REG_MOT1A
        motB = DEVICE_REG_MOT1B
        motPWM = DEVICE_REG_MOT1PWM
    elif(select == 2):
        motA = DEVICE_REG_MOT2A
        motB = DEVICE_REG_MOT2B
        motPWM = DEVICE_REG_MOT2PWM
    else:
        print("wtf")

    bus.write_i2c_block_data(DEVICE_ADDRESS, motPWM, pwm_on)
    if(brake):
        bus.write_i2c_block_data(DEVICE_ADDRESS, motA, pwm_on)
        bus.write_i2c_block_data(DEVICE_ADDRESS, motB, pwm_on)
    elif(speed == 4096):
        bus.write_i2c_block_data(DEVICE_ADDRESS, motA, pwm_off)
        bus.write_i2c_block_data(DEVICE_ADDRESS, motB, pwm_off)
    else:
        if(dir == 1):
            bus.write_i2c_block_data(DEVICE_ADDRESS, motA, pwm_speed)
            bus.write_i2c_block_data(DEVICE_ADDRESS, motB, pwm_off)
        else:
            bus.write_i2c_block_data(DEVICE_ADDRESS, motA, pwm_off)
            bus.write_i2c_block_data(DEVICE_ADDRESS, motB, pwm_speed)
    

    