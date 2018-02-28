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


N = 500
T = 1.0 # s
Nyq = N / (2 * T)
df = 1 / T
f = 115.5 # Hz
w = 2 * np.pi * f
 
t = np.linspace(0, T, N)
s = [np.sin(w * (_t if i < N * 3 / 5 else (_t + np.pi / 4)) / T) for i, _t in enumerate(t)]
 
 
dft = iDFT_slow(s)

dft = np.fft.fftshift(dft)

ndft = np.fft.ifft(s)


plt.subplot(4, 1, 1)

plt.plot(t, s, label='data')


plt.subplot(4, 1, 2)

plt.plot(t, np.angle(dft), label='amplitude')

#plt.plot(t, np.absolute(ndft), label='N amplitude')

#plt.legend(loc=(0.1, 0.25))
#plt.show()

#differ result 






x = np.linspace(0, pi2, 2000)
data = (np.sin(100 * x) + np.sin(101.8 * x)) / np.sqrt(2)

dft = iDFT_slow(data)
dft = np.fft.fftshift(dft)
#ndft = np.fft.ifft(data);

#plt.subplot(3, 1, 1)
plt.subplot(4, 1, 3)
plt.plot(x, data, label='data')
plt.subplot(4, 1, 4)
plt.plot(x, np.angle(dft), label='amplitude')


#plt.subplot(3, 1, 2)
#plt.plot(x, np.absolute(np.subtract(np.absolute(ndft), np.absolute(dft))), label='amplitude')
#plt.plot(x, np.absolute(ndft), label='N amplitude')




#plt.legend(loc=(0.4, 0.75))
plt.show()





