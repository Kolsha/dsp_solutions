import random
import math
import matplotlib.pyplot as plt
import numpy as np

pi2 = np.pi * 2.0


def create_filter_median(kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size should be odd")
    return [1 / kernel_size for i in range(kernel_size)]
    
def apply_filter_median(signal, filt):
    return np.convolve(signal, filt, mode='valid')

def impulse_response_median(filt):
    f = np.pad(filt, (0, len(filt) * 4), 'constant', constant_values=(0,0))
    return np.fft.fft(f)[:int(len(f) / 2)]


def prod(iterable):
    from functools import reduce
    from operator import mul
    return reduce(mul, iterable, 1)

def transfer_function_butterworth(Rp, Rs, w0, w1):
    ep = np.sqrt(np.power(10, Rp / 10.) - 1)
    es = np.sqrt(np.power(10, Rs / 10.) - 1)
    k = w0 / w1
    k1 = ep / es
    N = int(np.ceil(np.log(k1) / np.log(k)))
    L = N // 2
    r = N % 2
    a = np.power(ep, - 1. / N)
    th = [(2 * n + 1) / 2 / N * np.pi for n in range(L)]
    
    return lambda s: 1. / (ep * np.power(s + a, r) * prod(s*s + 2*a*np.sin(thn)*s + a*a for thn in th))



N = 2000
T = 2# s

t = np.linspace(0, T, N)
s = np.ones(N)

df = 1. / T
k = [i * df for i in range(N)]

Rp = 1.0 
Rs = 30.0
w0 = 17.0 #/ 10.0
w1 = 25.0 #/ 100.0
HGen = transfer_function_butterworth(Rp, Rs, w0, w1)

H_jw = [HGen(1j*i / w0) for i in k]
afr = np.absolute(H_jw)
phase = np.angle(H_jw)


# sym = np.append(afr[::-1], afr)
coef = np.fft.ifft(afr)


# coef = [HGen((1 - 1. / z) / (1 + 1. / z) * 2 * N/T) for z in t[1:]]
# coef = coef[:len(coef) // 2]



window = 10

coef[window:-window] = 0
H_jw = np.fft.fft(coef.real)
afr = np.absolute(H_jw)
afr = afr[:len(afr) // 2]


numPlots = 3


plt.subplot(numPlots, 1, 1).set_title('AFR')

plt.plot(k[:len(afr)], afr, label='AFR')


plt.legend(loc='upper right')

'''



plt.subplot(numPlots, 1, 2).set_title('Coef')

plt.plot(t, coef.real, label='coef real')


plt.plot(t, coef.imag, label='coef imag')

plt.legend(loc='upper right')





plt.subplot(numPlots, 1, 3).set_title('Phase')

plt.plot(t, phase, label='Phase')

plt.legend(loc='upper right')
'''


plt.show()

''''

####

f1 = 1
f2 = 8
w1 = 2 * np.pi * f1
w2 = 2 * np.pi * f2


s = [np.sin(w1 * i) + np.sin(w2 * i) for i in t]

filtsym = np.append(afr, afr[::-1])

filtered = np.fft.ifft(np.fft.fft(s) * filtsym).real


plt.plot(t, s, label='signal')
plt.plot(t, filtered, label='filtered')

plt.legend(loc='upper right')

plt.show()

######



f1 = 1
f2 = 8
w1 = 2 * np.pi * f1
w2 = 2 * np.pi * f2

s = [np.sin(w1 * i) + np.sin(w2 * i) for i in t]

filtsym = np.append(afr, afr[::-1])

filtered = np.fft.ifft(np.fft.fft(s) * filtsym).real

kernel_size = 601
filt = create_filter_median(kernel_size)
filtered_median = np.append(np.zeros(kernel_size // 2), apply_filter_median(s, filt))




plt.plot(t, s, label='signal')
plt.plot(t, filtered, label='filtered')

plt.plot(t[:len(filtered_median)], filtered_median, label='filtered median')

plt.legend(loc='upper right')

plt.show()

'''