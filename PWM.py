from RPIO import PWM

# Setup PWM and DMA channel 0
PWM.setup()
PWM.init_channel(0)

# Add some pulses to the subcycle
PWM.add_channel_pulse(0, 17, 0, 50)
PWM.add_channel_pulse(0, 17, 100, 50)

# Stop PWM for specific GPIO on channel 0
PWM.clear_channel_gpio(0, 17)

while True:
    #hellos
    print("hi")




    
# Shutdown all PWM and DMA activity
PWM.cleanup()


servo = PWM.Servo()

# Set servo on GPIO17 to 1200µs (1.2ms)
servo.set_servo(17, 1200)

# Set servo on GPIO17 to 2000µs (2.0ms)
servo.set_servo(17, 2000)

# Clear servo on GPIO17
servo.stop_servo(17)
