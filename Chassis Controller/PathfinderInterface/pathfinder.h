#ifndef PATHFINDER_H_
#define PATHFINDER_H_
#ifndef F_CPU
#define F_CPU 8000000UL
#endif

#include <avr/io.h>
#include <util/delay.h>
#include <stdlib.h>
#include <avr/interrupt.h>
#include "I2C_slave.h"
#include <avr/sleep.h>
#include <stdio.h>
#include <string.h>

#define sbi(b,n) ((b) |= (1<<(n)))          /* Set bit number n in byte b */
#define cbi(b,n) ((b) &= (~(1<<(n))))       /* Clear bit number n in byte b   */
#define fbi(b,n) ((b) ^= (1<<(n)))          /* Flip bit number n in byte b    */
#define rbi(b,n) ((b) & (1<<(n)))           /* Read bit number n in byte b    */
#define ASLP PB0
#define BSLP PD7

void init(void);
void update_servo1(uint8_t angle);
void update_servo2(uint8_t angle);
void update_motorA(uint8_t control, uint8_t speed);
void update_motorB(uint8_t control, uint8_t speed);
void setLED(uint8_t led);
void update(void);



#endif // PATHFINDER_H_