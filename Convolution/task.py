import random
import math
import matplotlib.pyplot as plt
import numpy as np

pi2 = np.pi * 2.0


def fConv(f1, f2):
    dft1 = np.fft.fft(f1)
    dft2 = np.fft.fft(f2)
    n = min(len(dft1), len(dft2))
    return np.fft.ifft(dft1[:n] * dft2[:n])

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




x = np.linspace(0, (2 * pi2), 640)
data1 = (np.sin(1 * x)) / np.sqrt(2)

data2 = (np.sin(1 * x + np.pi / 2.0)) / np.sqrt(2)


Sconv = Conv_slow(data1, data2)
Fconv = fConv(data1, data2)
NPconv = np.convolve(data1, data2, mode='same')

#print len(x), len(NPconv)

numPlots = 5

plt.subplot(numPlots, 1, 1).set_title('Datas')
plt.plot(x, [d.real for d in data1], label='data1')
plt.plot(x, [d.real for d in data2], label='data2')
plt.legend(loc='upper right')



plt.subplot(numPlots, 1, 2).set_title('Slow and Furie')
plt.plot(x, [d.real for d in Sconv], label='Slow conv')
plt.plot(x, [d.real for d in Fconv], label='Fconv')
plt.legend(loc='upper right')


plt.subplot(numPlots, 1, 3).set_title('Slow and NumPy')
plt.plot(x, [d.real for d in Sconv], label='Slow conv')
plt.plot(x, [d.real for d in NPconv], label='NPconv')
plt.legend(loc='upper right')


plt.subplot(numPlots, 1, 4).set_title('Furie and NumPy')
plt.plot(x, [d.real for d in Fconv], label='Fconv')
plt.plot(x, [d.real for d in NPconv], label='NPconv')
plt.legend(loc='upper right')


plt.subplot(numPlots, 1, 5).set_title('Deltas')
plt.plot(x, [d.real for d in np.subtract(Sconv, Fconv)], label='Sl. & Furie')
plt.plot(x, [d.real for d in np.subtract(Sconv, NPconv)], label='Sl. & NP')
plt.plot(x, [d.real for d in np.subtract(Fconv, NPconv)], label='Furie. & NP')

plt.legend(loc='upper right')

'''plt.subplot(6, 1, 5).set_title('Delta Slow and Furie')
plt.plot(x, [d.real for d in np.subtract(Sconv, Fconv)], label='delta')


plt.subplot(6, 1, 6).set_title('Delta Slow and NumPy')
plt.plot(x, [d.real for d in np.subtract(Sconv, NPconv)], label='delta')
'''


#plt.ylim(-1.5, 2.0)


plt.show()
