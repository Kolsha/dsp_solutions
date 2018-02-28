# Discrete Fourier Transform (DFT)


import random
import math
import cmath
import matplotlib.pyplot as plt
import numpy as np

pi2 = cmath.pi * 2.0
def DFT(fnList):
    N = len(fnList)
    FmList = []
    for m in range(N):
        Fm = 0.0
        for n in range(N):
            Fm += fnList[n] * cmath.exp(- 1j * pi2 * m * n / N)
        FmList.append(Fm / N)
    return FmList
        
def InverseDFT(FmList):
    N = len(FmList)
    fnList = []
    for n in range(N):
        fn = 0.0
        for m in range(N):
            fn += FmList[m] * cmath.exp(1j * pi2 * m * n / N)
        fnList.append(fn)
    return fnList

import cmath
def compute_dft_complex(input):
    n = len(input)
    output = []
    for k in range(n):  # For each output element
        s = complex(0)
        for t in range(n):  # For each input element
            angle = 2j * cmath.pi * t * k / n
            s += input[t] * cmath.exp(-angle)
        output.append(s)
    return output



# TEST
print "Input Sine Wave Signal:"
N = 150 # degrees (Number of samples)
a = float(random.randint(1, 100))
f = float(random.randint(1, 100))
p = float(random.randint(0, 360))
print "frequency = " + str(f)
print "amplitude = " + str(a)
print "phase ang = " + str(p)
print
fnList = []

for n in range(N):
    t = float(n) / N * pi2
    fn = a * math.sin(f * t + p / 360 * pi2)
    fnList.append(complex(fn, 0))

for n in range(N):
    if(n > 20 and n < 40):
        del fnList[n]

print "DFT Calculation Results:"
FmList = compute_dft_complex(fnList)

threshold = 0.001
for (i, Fm) in enumerate(FmList):
    if abs(Fm) > threshold:
        print "frequency = " + str(i)
        p = int(((cmath.phase(Fm.real) + pi2 + pi2 / 4.0) % pi2) / pi2 * 360 + 0.5)
        print "phase ang = " + str(p)

        print "amplitude(r) = " + str(abs(Fm.real) * 2.0)
        

        print "amplitude(i) = " + str(abs(Fm.imag) * 2.0)
        print




plt.title('Digital filter frequency response')
plt.subplot(1, 2, 1)

plt.plot([abs(FmList[i])  for i in range(len(FmList))], 'b')

#plt.plot([abs(FmList[i].real) for i in range(len(FmList))], 'r')
plt.subplot(1, 2, 2)
plt.plot(fnList, 'g')
plt.grid()
plt.show()



### Recreate input signal from DFT results and compare to input signal
##fnList2 = InverseDFT(FmList)
##for n in range(N):
##    print fnList[n], fnList2[n].real