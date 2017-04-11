import csv
import matplotlib.pyplot as plt
import numpy as np

t = []
t_taken = []
off = []
angle = []
offpid = []
anglepid = []
output = []
with open('log.csv', 'rt') as csvfile:
	logreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	next(logreader)
	next(logreader)
	for row in logreader:
		t.append(float(row[0]))
		t_taken.append(float(row[1]))
		off.append(float(row[2]))
		angle.append(float(row[3]))
		offpid.append(float(row[4]))
		anglepid.append(float(row[5]))
		output.append(int(row[6]))
print(t)
print(t_taken)
plt.plot(t, t_taken)
plt.figure(2)
plt.plot(t, off, 'r', t, angle, 'b', t, offpid, 'g-', t, output, 'y')
plt.show()
