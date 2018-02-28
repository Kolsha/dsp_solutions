import random
import math
import matplotlib.pyplot as plt
import numpy as np

pi2 = np.pi * 2.0

def DFT_slow(data):
    n = len(data)
    out = []
    for k in range(n):
        v = np.imag(0)
        v.flags['WRITEABLE'] = True
        for t in range(n):
            angle = 1j * pi2 * t * k / n
            a = data[t] * np.exp(-angle)
            v = v + a
        out.append(v)
    return out

def iDFT_slow(data):
    n = len(data)
    out = []
    for k in range(n):
        v = np.imag(0)
        v.flags['WRITEABLE'] = True
        for t in range(n):
            angle = 1j * pi2 * t * k / n
            a = data[t] * np.exp(angle)
            v = v + a
        out.append(v / n)
    return out


x1 = np.linspace(0, pi2 * 3/7, int(500 * 3/7))
x2 = np.linspace(pi2 * 5/7, pi2, int(500 * 2/7))

x = np.linspace(0, pi2 * 5 / 7, 356)
data = np.append(np.sin(x1), np.sin(x2))

dft = iDFT_slow(data)

ndft = np.fft.ifft(data);

plt.plot(x, data, label='data')
plt.plot(x, np.absolute(dft), label='amplitude')

plt.plot(x, np.absolute(ndft), label='N amplitude')

plt.legend(loc=(0.1, 0.25))
plt.show()







x = np.linspace(0, pi2, 500)
data = (np.sin(100 * x) + np.sin(101 * x)) / np.sqrt(2)

dft = iDFT_slow(data)
ndft = np.fft.ifft(data);

plt.subplot(3, 1, 1)
plt.plot(x, data, label='data')
plt.plot(x, np.absolute(dft), label='amplitude')


plt.subplot(3, 1, 2)
plt.plot(x, np.absolute(dft), label='amplitude')
plt.subplot(3, 1, 3)
plt.plot(x, np.absolute(ndft), label='N amplitude')


plt.legend(loc=(0.4, 0.75))
plt.show()





