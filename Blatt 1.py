import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["font.family"] = "arial"
mpl.rcParams["font.size"] = "14"

# Aufgabe 1a

fig, a = plt.subplots()

x = np.linspace(0.001,10,100)


f1 = np.sqrt(x)
f2 = (np.log(x)) / (x+1)

a.plot(x, f1, color="green",label="$y_1(x)$")
a.plot(x, f2, color="blue",label="$y_1(x)$")

a.set_xlabel("x-Axis")
a.set_ylabel("y-Axis")
a.legend()

plt.show()

# Aufgabe 1b

fig, axis1b = plt.subplots(1, 2, figsize=(12, 3))

x = np.linspace(0, 10, 100)

f1 = np.exp(x)

axis1b[0].semilogy(x, f1, color="green", label="$y_1(x)$")
axis1b[1].plot(x, f1, color="red", label="$y_2(x)$")

for axis in axis1b:
    axis.set_xlabel("x-Axis")
    axis.set_ylabel("y-Axis")
    axis.legend()

plt.show()

# Aufgabe 1c

fig, axis1c = plt.subplots(1, 2, figsize=(12, 3))

x = np.linspace(0, 10, 100)

f1 = np.sqrt(x)

axis1c[0].loglog(x, f1, color="green", label="$y_1(x)$")
axis1c[1].plot(x, f1, color="red", label="$y_2(x)$")

for axis in axis1c:
    axis.set_xlabel("x-Axis")
    axis.set_ylabel("y-Axis")
    axis.legend()

plt.show()