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


def fConv_slow(f1, f2):
    dft1 = DFT_slow(f1)
    dft2 = DFT_slow(f2)
    l1 = len(dft1)
    l2 = len(dft2)

    n = max(l1, l2)
    if(n > l1):
        np.pad(dft1, (0, n), 'constant', constant_values=0)
    if(n > l2):
        np.pad(dft2, (0, n), 'constant', constant_values=0)

    return iDFT_slow(np.multiply(dft1, dft2))

def Conv_slow(f1, f2):
    n = len(f1)
    n2 = len(f2)
    out = []
    for k in range(n):
        v = np.imag(0)
        v.flags['WRITEABLE'] = True
        for t in range(n):
            f2v = 0
            if (k - t) < n2:
                f2v = f2[k - t] 
            a = f1[t] * f2v
            v = v + a
        out.append(v)
    return out




x = np.linspace(0, 2*pi2, 20)
data1 = (np.sin(1 * x[0:10])) / np.sqrt(2)

data2 = (np.sin(20.8 * x)) / np.sqrt(2)


conv1 = Conv_slow(data1, data2)
conv2 = fConv_slow(data1, data2)

plt.subplot(4, 1, 1)

plt.plot(x, [d.real for d in data1], label='data1')
plt.subplot(4, 1, 2)
plt.plot(x, [d.real for d in data2], label='data2')


plt.subplot(4, 1, 3)

plt.plot(x, [d.real for d in conv1], label='conv')


plt.subplot(4, 1, 4)

plt.plot(x, [d.real for d in np.subtract(conv1, conv2)], label='fconv')


plt.show()
