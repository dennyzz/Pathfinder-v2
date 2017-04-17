You are gonna need Python 3.x
The final working script for Pathfinder (RC Car) 
is 
# python3 curvepithreads.py

There are some Docs that explain setting up specific components on the Raspi in /Docs
/footage and /images are some resources we used for offline testing to verify the plausibilty of algorithms from a computer running python3
VL53L0X_rasp contains the library for the VL53L0X component 

The final files used for Pathfinder (RC Car) are
/curvepithreads.py << main program
/pathfindershield.py << import for commmuncations to chassis board
/PID.py 
support files
/VL53L0X_rasp liibrary needs to have # make run on it to compile the library for used
/logplot.py used for plotting the log file when logging is turned on
/lcd8+LED.py currently used as a startup script to display the ip address and blink the headlights on boot
Extra files
Previous iterations
/curve.py and /threadex.py are meant to run on a regular computer for testing
/curvepi.py is the non-threaded version of the pathfinder program
/VL53L0X_ex.py is the test file for that component

Other project resources
lots of other files of code written...
PCB is located in /Chassis Controller/PCB made in Altium, though Gerber files are zipped
Firmware for the ATMega328P on the Chassis is in /Chassis Controller/Pathfinderinterface made in Atmel Studio 