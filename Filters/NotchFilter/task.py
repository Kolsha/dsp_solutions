import random
import math
import matplotlib.pyplot as plt
import numpy as np

pi2 = np.pi * 2.0


def NotchFilter(data, window_size):
    data_len = len(data)
    fltr = np.ones(window_size) / window_size
    #fltr = np.pad(fltr, (0, data_len - window_size), 'constant', constant_values=(0, 0))

    res = np.convolve(data, fltr, mode='valid')

    return res


def ImpulseResponse(window_size):
    fltr = np.ones(window_size) / window_size

    pd_len = len(fltr) + 1000
    fltr = np.pad(fltr, (0, pd_len), 'constant', constant_values=(0, 0))

    fft = np.fft.fft(fltr)
    #fft = np.fft.fftshift(fft)
    return fft[:len(fft)/2]





data1 = np.random.rand(100) * 100

x = range(0, len(data1))



NPconv = NotchFilter(data1, 15)

xf = range(0, len(NPconv))


print len(data1), len(x), len(NPconv)


numPlots = 2

plt.subplot(numPlots, 1, 1).set_title('Datas')
plt.plot(x, [d.real for d in data1], label='data1')


plt.plot(xf, [d.real for d in NPconv], label='convolve')


ir = ImpulseResponse(15)

xir = range(len(ir))


#fft = np.fft.fft(NPconv)
#ir = np.fft.fftshift(fft)

#xir = range(len(ir))



plt.subplot(numPlots, 1, 2).set_title('IR')

plt.plot(xir, [np.absolute(d) for d in ir], label='IR')



plt.legend(loc='upper right')






plt.show()