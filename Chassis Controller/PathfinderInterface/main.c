#include "pathfinder.h"

int main(void){
	init();
	// default "register" values
	rxbuffer[0] = 0;
	rxbuffer[1] = 0;
	rxbuffer[2] = 0;
	rxbuffer[3] = 0;
	rxbuffer[4] = 0;
	rxbuffer[5] = 128;
	rxbuffer[6] = 128;
	rxbuffer[7] = 0;

	I2C_init(0x5A); // initalize as slave with address 0x5A
	
	// allow interrupts
	sei();
	set_sleep_mode(SLEEP_MODE_IDLE);
	
	while(1){
		sleep_enable();
		sei();
		sleep_cpu();
		sleep_disable();
		// on every interrupt, awke and update from the registers
		update();
		// convert receiver buffer index 0 to character array and send it via UART
	}
	
	return 0;
}
