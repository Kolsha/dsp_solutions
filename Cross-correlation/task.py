import random
import math
import matplotlib.pyplot as plt
import numpy as np

pi2 = np.pi * 2.0


def fCorrelation(f1, f2):
    n1 = len(f1);
    n2 = len(f2)
    n = n1 + n2 - 1
    f1 = np.pad(f1, (0, n - n1), 'constant', constant_values=(0, 0))
    f2 = np.pad(f2, (0, n - n2), 'constant', constant_values=(0, 0))
    dft1 = np.fft.fft(f1)
    dft1 = np.conjugate(dft1)

    dft2 = np.fft.fft(f2)
    res = np.fft.ifft(dft1 * dft2)

    return np.fft.fftshift(res)

def Correlation_slow(f1, f2):
    n1 = len(f1);
    n2 = len(f2)
    n = n1 + n2 - 1
    f1 = np.pad(f1, (0, n - n1), 'constant', constant_values=(0, 0))
    f2 = np.pad(f2, (n - n2, 0), 'constant', constant_values=(0, 0))
    out = np.empty(n)
    for k in range(n):
        v = np.imag(0)
        v.flags['WRITEABLE'] = True
        for t in range(n):
            f2v = np.imag(0)
            if (k + t) < n:
                f2v = f2[k + t] 

            a = f2v * np.conjugate(f1[t]) 
            v = v + a
        out[k] = (v)
    return out




x = range(0, 50)

xf = range(0, 50 * 2 - 1)


data1 = np.empty(50)
data1.fill(0)
data1[30:40] = 1

data2 = np.empty(50)
data2.fill(0)

data2[25:35] = 1


data1 = np.random.rand(50)
data2 = np.random.rand(50)

'''x = np.linspace(0, (2 * pi2), 100)
data1 = (np.sin(1 * x)) / np.sqrt(2)

data2 = (np.sin(1 * x)) / np.sqrt(2)

xf = np.arange(199)
'''

Sconv = Correlation_slow(data1, data2)
Fconv = fCorrelation(data1, data2)
NPconv = np.correlate(data1, data2, mode='full')[::-1]

print len(x), len(NPconv), len(Sconv), len(Fconv)



numPlots = 5

plt.subplot(numPlots, 1, 1).set_title('Datas')
plt.plot(x, [d.real for d in data1], label='data1')
plt.plot(x, [d.real for d in data2], label='data2')
plt.legend(loc='upper right')



plt.subplot(numPlots, 1, 2).set_title('Slow and Furie')
plt.plot(xf, [d.real for d in Sconv], label='Slow conv')
plt.plot(xf, [d.real for d in Fconv], label='Fconv')
plt.legend(loc='upper right')


plt.subplot(numPlots, 1, 3).set_title('Slow and NumPy')
plt.plot(xf, [d.real for d in Sconv], label='Slow conv')
plt.plot(xf, [d.real for d in NPconv], label='NPconv')
plt.legend(loc='upper right')


plt.subplot(numPlots, 1, 4).set_title('Furie and NumPy')
plt.plot(xf, [d.real for d in Fconv], label='Fconv')
plt.plot(xf, [d.real for d in NPconv], label='NPconv')
plt.legend(loc='upper right')


plt.subplot(numPlots, 1, 5).set_title('Deltas')
plt.plot(xf, [d.real for d in np.subtract(Sconv, Fconv)], label='Sl. & Furie')
plt.plot(xf, [d.real for d in np.subtract(Sconv, NPconv)], label='Sl. & NP')
plt.plot(xf, [d.real for d in np.subtract(Fconv, NPconv)], label='Furie. & NP')

plt.legend(loc='upper right')

'''plt.subplot(6, 1, 5).set_title('Delta Slow and Furie')
plt.plot(x, [d.real for d in np.subtract(Sconv, Fconv)], label='delta')


plt.subplot(6, 1, 6).set_title('Delta Slow and NumPy')
plt.plot(x, [d.real for d in np.subtract(Sconv, NPconv)], label='delta')
'''


#plt.ylim(-1.5, 2.0)


plt.show()
