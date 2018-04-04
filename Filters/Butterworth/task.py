import random
import math
import matplotlib.pyplot as plt
import numpy as np

pi2 = np.pi * 2.0


def create_filter_butterworth(signalLength, sampleFreq, order, f0, Gain):
    nBins = signalLength // 2
    binWidth = sampleFreq / signalLength
    
    filt = np.array([0 for i in range(nBins)], dtype=complex)
    
    for i in range(nBins):
        binFreq = binWidth * i
        gain = Gain / np.sqrt(1 + np.power(binFreq / f0, 2 * order))
        filt[i] = gain
        
#     df = 2 * np.pi / f0
#     filt[i] *= np.exp(1j * df * i)
    
    rect = [1 if i < 10 else 0 for i in range(signalLength)]
    rect = np.fft.fft(rect)[:len(rect) // 2]
    
    for i in range(len(filt)):
        filt[i] *= np.exp(1j * -np.angle(rect[i]))
    return filt

def apply_filter_butterworth(s, filt):
    N = len(s)
    ffts = np.fft.fft(s)
    
    nBins = N // 2
    
    for i in range(nBins):
        ffts[i] *= filt[i]
        ffts[N - i - 1] *= filt[i]
    return np.real(np.fft.ifft(ffts))

def impulse_response_butterworth(filt):
    return filt

def prod(iterable):
    from functools import reduce
    from operator import mul
    return reduce(mul, iterable, 1)



def transfer_function_butterworth(Rp, Rs, w0, w1):
    ep = np.sqrt(np.power(10, float(Rp) / 10.) - 1)
    es = np.sqrt(np.power(10, float(Rs) / 10.) - 1)
    k = (w0) / float(w1)

    #print w0, w1, k
    k1 = float(ep) / float(es)
    N = int(np.ceil(np.log(k1) / np.log(k))) # xz

    L = N // 2
    r = N % 2
    a = np.power(float(ep), - 1. / N)
    th = [(2 * n + 1) / 2 / N * np.pi for n in range(L)]
    
    return lambda s: 1. / (ep * np.power(s + a, r) * prod(s*s + 2*a*np.sin(thn)*s + a*a for thn in th))




N = 20
T = 1.0 * N # s

Fs = 25
f1 = 1
f2 = 8
w1 = 2 * np.pi * f1
w2 = 2 * np.pi * f2

t = np.linspace(0, T, N)
s = [np.sin(w1 / Fs * i) + np.sin(w2 / Fs * i) for i in t]

filt = create_filter_butterworth(N, 25, 3, 1.5, 1.0)
fs = apply_filter_butterworth(s, filt)
# fs = butterworth_filter(s, 25, 3, 1.5, 1.0)
ir = impulse_response_butterworth(filt)

sym = np.append(ir[::-1], ir)

coeffs = np.fft.ifft(sym.real) * 10
#coeffs = coeffs[:len(coeffs) // 2]

#coeffs[2:-2] = 0

#coeffs = np.fft.fft(coeffs)


numPlots = 3


plt.subplot(numPlots, 1, 1).set_title('Datas')

plt.plot(t, s, label='signal')

plt.plot(t, fs, label='filtered')



plt.legend(loc='upper right')

plt.subplot(numPlots, 1, 2).set_title('Chars')


plt.plot([i for i in range(len(ir.real))], np.absolute(ir), label='IR')

plt.plot([i for i in range(len(ir.real))], np.angle(ir) * 0.2, label='Angle')

plt.legend(loc='upper right')





N = 50
T = 10.0 # s

t = np.linspace(0, T, N)
s = np.ones(N)

df = 1. / T
k = [i * df for i in range(N)]

HGen = transfer_function_butterworth(20, 30, 1.0, 5.0)

afr = [np.power(np.absolute(HGen(1j*i)), 2) for i in k]
phase = [np.angle(HGen(1j*i)) for i in k]
ir = np.fft.fft(afr)

ir = ir[:len(ir) / 2]

plt.subplot(numPlots, 1, 3).set_title('Chars')



plt.plot(k, afr, label='AFR')


plt.plot([i for i in range(len(coeffs.real))], np.real(coeffs), label='real')

plt.plot([i for i in range(len(coeffs.imag))], np.imag(coeffs), label='imag')

#plt.plot(k, phase, label='Phase')

#plt.plot(k, np.absolute(ir), label='IR')

plt.legend(loc='upper right')












plt.show()

