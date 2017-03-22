#!/usr/bin/python

#define PCA9685_MODE1 0x0
#define PCA9685_PRESCALE 0xFE

import smbus

bus = smbus.SMBus(1)    # 0 = /dev/i2c-0 (port I2C0), 1 = /dev/i2c-1 (port I2C1)
DEVICE_ADDRESS = 0x60      #7 bit address (will be left shifted to add the read write bit)
DEVICE_REG_LED_BASE = 0x06
DEVICE_REG_MOTPWM = 0x06+8*4
DEVICE_REG_MOT1A = 0x06+9*4
DEVICE_REG_MOT1B = 0x06+10*4

PCA9685_MODE1 = 0x0
PCA9685_PRESCALE = 0xFE
PCA_MODE = 0x21
bus.write_byte_data(DEVICE_ADDRESS, PCA9685_MODE1, PCA_MODE | 0x10)
bus.write_byte_data(DEVICE_ADDRESS, PCA9685_PRESCALE, 0x06)
bus.write_byte_data(DEVICE_ADDRESS, PCA9685_MODE1, PCA_MODE)
pwm_on = [0x00, 0x10, 0xFF, 0x0F]
pwm_off = [0x00, 0x00, 0xFF, 0x1F]
while(True):
    #Write a single register
    # bus.write_byte_data(DEVICE_ADDRESS, DEVICE_REG_MODE1, 0x80)

    #Write an array of
    bus.write_i2c_block_data(DEVICE_ADDRESS, DEVICE_REG_MOT1A, pwm_on)
    bus.write_i2c_block_data(DEVICE_ADDRESS, DEVICE_REG_MOTPWM, pwm_on)
    bus.write_i2c_block_data(DEVICE_ADDRESS, DEVICE_REG_MOT1B, pwm_off)
    

# void Adafruit_MS_PWMServoDriver::setPWM(uint8_t num, uint16_t on, uint16_t off) {
#   //Serial.print("Setting PWM "); Serial.print(num); Serial.print(": "); Serial.print(on); Serial.print("->"); Serial.println(off);

#   WIRE.beginTransmission(_i2caddr);
# #if ARDUINO >= 100
#   WIRE.write(LED0_ON_L+4*num);
#   WIRE.write(on);
#   WIRE.write(on>>8);
#   WIRE.write(off);
#   WIRE.write(off>>8);
# #else
#   WIRE.send(LED0_ON_L+4*num);
#   WIRE.send((uint8_t)on);
#   WIRE.send((uint8_t)(on>>8));
#   WIRE.send((uint8_t)off);
#   WIRE.send((uint8_t)(off>>8));
# #endif
#   WIRE.endTransmission();
# }

#define MOTOR1_A 2
#define MOTOR1_B 3
#define MOTOR2_A 1
#define MOTOR2_B 4


#define LED0_ON_L 0x6
