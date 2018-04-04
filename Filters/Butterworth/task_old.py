import random
import math
import matplotlib.pyplot as plt
import numpy as np

pi2 = np.pi * 2.0


def ButterworthFilter(signal, sampleFrequency, order, f0, DCGain):
    N = len(signal)
    signalFFT = (np.fft.fft(signal))  #
    #signalFFT = np.fft.fftshift(signalFFT)
    if(f0 > 0):
        binWidth = float(sampleFrequency) / float(N)
        for i in range(1, N/2):
            binFreq = binWidth * i

            #print binWidth, i

            gain = DCGain / np.sqrt(1 + np.power(binFreq / f0, 2 * order))

            #print gain

            signalFFT[i] *= gain
            signalFFT[N - i] *= gain

    return (np.fft.ifft( signalFFT ))





def ImpulseResponse(N, sampleFrequency, order, f0, DCGain):


    ir = np.array([1 for i in range(0, N)], dtype = complex)
    
    if(f0 > 0):
        binWidth = float(sampleFrequency) / float(N)

        df = pi2 / float(sampleFrequency)

        for i in range(1, N):
            binFreq = binWidth * i

            gain = DCGain / np.sqrt(1 + np.power(binFreq / f0, 2 * order))

            #tmp = np.imag(df * i)
            #tmp.flags['WRITEABLE'] = True
            #tmp.real = gain
            ir[i] = gain #* np.exp(1j * df * i)
            #ir[N - i] *= gain



    

    #ir += ir[::-1]


    fft = np.fft.ifft(ir)
    fft = np.fft.fftshift(fft)


    return fft[len(fft)/2:]




sampleCount = 700

rect = np.empty(sampleCount)
rect.fill(0)

rect[:10] = 1

numPlots = 3

rf = np.fft.fft(rect)


plt.subplot(numPlots, 1, 1).set_title('Datas')
#plt.plot( [d.real for d in rect], label='signal')

#plt.plot( [np.absolute(d) for d in rf], label='ir')

phase = np.angle(rf)

plt.plot(range(0, sampleCount), [np.absolute(d) for d in phase], label='ph')






sampleFrequency = 10



ir = ImpulseResponse(sampleCount, sampleFrequency, 30, 1.5, 1.1)


mir = np.array([1 for i in range(0, len(ir))], dtype = complex)

rf =  np.imag(rf)
for i in range(len(ir)):

    tmp = np.complex(ir[i].real, rf[i].imag)
    #tmp.flags['WRITEABLE'] = True
    #tmp.imag = 

    

    #tmp.real = 

    mir[i] = tmp



plt.plot(range(0, len(mir)), [np.absolute(d * 100) for d in mir], label='new ir')


plt.legend(loc='upper right')

plt.show()

'''
Fs = 25 # Sample rate, 25Hz
f1 = 1  # 1 Hz Signal
f2 = 8  # 8 Hz Signal


sampleCount = 100

sampleFrequency = 10

signal = np.empty(sampleCount)

for t in range(0, sampleCount):
    signal[t] = np.sin( pi2 * f1 / Fs * t ) + np.sin( pi2 * f2 / Fs * t )
 




filteredSignal = ButterworthFilter(signal, sampleFrequency, 30, 1.5, 1.1 );



print len(signal), len(filteredSignal)

x = range(len(signal))


numPlots = 3

plt.subplot(numPlots, 1, 1).set_title('Datas')
plt.plot(x, [d.real for d in signal], label='signal')


plt.plot(x, [d.real for d in filteredSignal], label='filteredSignal')

plt.legend(loc='upper right')



plt.subplot(numPlots, 1, 2).set_title('IR')

ir = ImpulseResponse(sampleCount, sampleFrequency, 30, 1.5, 1.1) * 2

xir = range(len(ir))

plt.plot(xir, [np.absolute(d) for d in ir], label='IR')

phase = np.angle(ir)

plt.plot(xir, [(d) for d in phase], label='Ph')

ir += ir[::-1]
ircnt = (np.fft.ifft(ir))
ircnt = np.fft.fftshift(ircnt) 

plt.legend(loc='upper right')

plt.subplot(numPlots, 1, 3).set_title('CNT')
plt.plot(xir, [(d.real) for d in ircnt], label='Cnt')


phase = np.angle(ircnt)
plt.plot(xir, [(d.imag) for d in ircnt], label='Ph')

#plt.plot(x, [(d.real) for d in filteredSignal], label='filteredSignal')


plt.legend(loc='upper right')

'''

'''





plt.show()

'''

