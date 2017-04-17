#include "pathfinder.h"

#define I2CTIMEOUT 50 // 50 ticks of 50Hz is 1 second

uint8_t timeout = 0;

void init(void){
/* pins
* PB0 ASLP
* PB1 OC1A Servo1
* PB2 OC1B Servo2
* PB3 OC2A AIN1
* PD3 OC2B AIN2 
* PD5 OC0B BIN2
* PD6 OC0A BIN1
* PD7 BSLP
* 
* PC4 SDA
* PC5 SCL
* 
* PC0 PC1 PC2 PC3 are misc GPIO pins defined as LED outputs
* PD0 PD1 PD2 PD4 are misc GPIO pins defined as LED outputs
* 
*/
		// ENABLE AIN1 AIN2 ASLP
		DDRB |= (1<<DDB0) | (1<<DDB3);
		DDRD |= (1<<DDD3);
		PORTB &= ~((1<<PORTB0) | (1<<PORTB3));
		PORTD &= ~(1<<PORTD3);

		// ENABLE BIN1 BIN2 BSLP
		DDRD |= (1<<DDD5) | (1<<DDD6) | (1<<DDD7);
		PORTD &= ~((1<<PORTD5) | (1<<PORTD6) | (1<<PORTD7));

		// ENABLE *new* Servo1 and Servo2
		DDRB |= (1<<DDB1) | (1<<DDB2);
		PORTB = ~((1<<PORTB1) | (1<<PORTB2));

		// 8 LED out pin inits
		DDRC |= (1<<DDC0) | (1<<DDC1) | (1<<DDC2) | (1<<DDC3);
		DDRD |= (1<<DDD0) | (1<<DDD1) | (1<<DDD2) | (1<<DDD4);

		// time calculations 8MHz clk, 8 bit resolution 8,000,000 / 255
		// OPTION 1: prescale [001] = /1 , 31Khz
		// OPTION 2: prescale [010] = /8 , 3.9kHz
		// OPTION 3: prescale [011] = /64 , 488Hz
		// >>>OPTION 4: prescale [100] = /256 , 122Hz
		// OPTION 5: prescale [101] = /1024 , 30Hz
		// TIMER0 (MOTOR B PWM) Fast PWM, A/B non-inverting, FOC is off // clock select? 
		// interrupts off
		TCCR0A = (1<<WGM00) | (1<<WGM01) | (0<<COM0B0) | (1<<COM0B1) | (0<<COM0A0) | (1<<COM0A1);
		TCCR0B = (0<<CS00) | (0<<CS01) | (1<<CS02) | (0<<WGM02) | (0<<FOC0B) | (0<<FOC0A);
		TIMSK0 = (0<<TOIE0) | (0<<OCIE0A) | (0<<OCIE0B);
		OCR0A = 0;
		OCR0B = 0;
		PORTD &= ~(1<<7); 

		// TIMER1 (SERVOS) 
		// input capture noise cancellers: off
		// required frequency = 50Hz 20mS with maximal resolution
		// SELECT: prescale [010] = /8 , TOP = 20000, gives 20000 steps with 50Hz cycle
		// FAST PWM mode with ICR1 as TOP A/B non-inverting 
		TCCR1A = (0<<WGM10) | (1<<WGM11) | (0<<COM1B0) | (1<<COM1B1) | (0<<COM1A0) | (1<<COM1A1);
		TCCR1B = (0<<CS10) | (1<<CS11) | (0<<CS12) | (1<<WGM12) | (1<<WGM13) | (0<<ICES1) | (0<<ICNC1);
		TCCR1C = 0;
		ICR1 = 20000;
		OCR1A = 1500; // set default to 1.5mS or neutral
		OCR1B = 1500; // set default to 1.5mS or neutral
		TIMSK1 = (1<<TOIE1) | (0<<OCIE1A) | (0<<OCIE1B) | (0<<ICIE1);

		//TIMER2 (MOTOR A PWM) Same as MOTOR B
		TCCR2A = (1<<WGM20) | (1<<WGM21) | (0<<COM2B0) | (1<<COM2B1) | (0<<COM2A0) | (1<<COM2A1);
		TCCR2B = (0<<CS20) | (0<<CS21) | (1<<CS22) | (0<<WGM22) | (0<<FOC2B) | (0<<FOC2A);
		TIMSK0 = (0<<TOIE0) | (0<<OCIE0A) | (0<<OCIE0B);
		OCR2A = 0;
		OCR2B = 0;
		PORTB &= ~(1<<PORTB0);
}

const uint16_t servo_min = 1200;
const uint16_t servo_max = 1700;
// convert a 0x00 - 0xFF command to a proper PWM value for the servos
void update_servo1(uint8_t angle){
	// since the region is approx 500 counts, and 2x8bit = 512, we'll simplify the math to avoid division
	if(angle < 4){
		OCR1A = servo_min;
	}
	else if (angle > 252){
		OCR1A = servo_max;
	}
	else{
		OCR1A = ((uint16_t)angle - 3)*2 + servo_min;
	}
}

void update_servo2(uint8_t angle){
	// since the region is approx 500 counts, and 2x8bit = 512, we'll simplify the math to avoid division
	if(angle < 4){
		OCR1B = servo_min;
	}
	else if (angle > 252){
		OCR1B = servo_max;
	}
	else{
		OCR1B = ((uint16_t)angle - 3)*2 + servo_min;
	}
}

#define DIR_BIT 0
#define BRAKE_BIT 1
// take in motor control bytes and process them 
void update_motorA(uint8_t control, uint8_t speed){
	if(control & (1<<BRAKE_BIT)){
		//apply the brakes
		sbi(PORTB, ASLP);
		OCR2A = 0xFF;
		OCR2B = 0xFF;
	}
	else if (speed == 0){
		cbi(PORTB, ASLP);
	}
	else if (control & (1<<DIR_BIT)){
		sbi(PORTB, ASLP);
		OCR2A = speed;
		OCR2B = 0;
	}
	else{
		sbi(PORTB, ASLP);
		OCR2A = 0;
		OCR2B = speed;
	}
}

void update_motorB(uint8_t control, uint8_t speed){
	if(control & (1<<BRAKE_BIT)){
		//apply the brakes
		sbi(PORTD, BSLP);
		OCR0A = 0xFF;
		OCR0B = 0xFF;
	}
	else if (speed == 0){
		cbi(PORTD, BSLP);
	}
	else if (control & (1<<DIR_BIT)){
		sbi(PORTD, BSLP);
		OCR0A = speed;
		OCR0B = 0;
	}
	else{
		sbi(PORTD, BSLP);
		OCR0A = 0;
		OCR0B = speed;
	}
}

// we are using the extra GPIO pins to run LEDs	
void setLED(uint8_t led){
	if(led & 1<<0){PORTC |= (1<<0);}
	else{PORTC &= ~(1<<0);}
	if(led & 1<<1){PORTC |= (1<<1);}
	else{PORTC &= ~(1<<1);}
	if(led & 1<<2){PORTC |= (1<<2);}
	else{PORTC &= ~(1<<2);}
	if(led & 1<<3){PORTC |= (1<<3);}
	else{PORTC &= ~(1<<3);}
	if(led & 1<<4){PORTD |= (1<<0);}
	else{PORTD &= ~(1<<0);}
	if(led & 1<<5){PORTD |= (1<<1);}
	else{PORTD &= ~(1<<1);}
	if(led & 1<<6){PORTD |= (1<<2);}
	else{PORTD &= ~(1<<2);}
	if(led & 1<<7){PORTD |= (1<<4);}
	else{PORTD &= ~(1<<4);}
}

void update(void){
// [0] status? blank
// [1] MOTORA_CTRL
// [2] MOTORA_SPEED
// [3] MOTORB_CTRL
// [4] MOTORB_SPEED
// [5] Servo1
// [6] Servo2
// [7] LEDs
	update_motorA(rxbuffer[1], rxbuffer[2]);
	update_motorB(rxbuffer[3], rxbuffer[4]);
	update_servo2(rxbuffer[5]);
	update_servo1(rxbuffer[6]);
	setLED(rxbuffer[7]);
}

// double usage of timer 1 to generate a timeout blinking from the 50Hz overflow 
ISR(TIMER1_OVF_vect){
	timeout++;
	if(timeout < I2CTIMEOUT){
		// do nothing?
	}
	else if (timeout < I2CTIMEOUT + 25){
		memset(rxbuffer,0,8);
		rxbuffer[5] = 128;
		rxbuffer[6] = 128;
	}
	else if (timeout < I2CTIMEOUT + 50){
		memset(rxbuffer,0,5);
		rxbuffer[5] = 128;
		rxbuffer[6] = 128;
		rxbuffer[7] = 0xEE;
	}
	else{
		timeout = I2CTIMEOUT;
	}
}