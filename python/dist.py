import smbus
bus = smbus.SMBus(1)

DEVICE_ADDRESS = 0x29
DEVICE_REG_MODE1 = 0x00
x = bus.read_byte_data(DEVICE_ADDRESS, 0xC0)
print(format(x, '02X'))
x1 = bus.read_byte_data(DEVICE_ADDRESS, 0xC1)
print(format(x1, '02X'))
x2 = bus.read_byte_data(DEVICE_ADDRESS, 0xC2)
print(format(x2, '02X'))
x3 = bus.read_word_data(DEVICE_ADDRESS, 0x51)
print(format(x3, '02X'))
x4 = bus.read_word_data(DEVICE_ADDRESS, 0x61)
print(format(x4, '02X'))

