import pathfindershield

# def motorservocmd4(Bspeed, Bdir, Bbrake, Servo2):

speed = int(sys.argv[1])
dir = int(sys.argv[2])
brake = int(sys.argv[3])
angle = int(sys.argv[4])
print("speed: " + str(speed))
print("direction: " + str(dir))
print("brakes: " + str(brake))
print("angle" + str(angle))

motorservocmd4(speed, dir, brake, angle)


