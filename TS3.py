#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 14:42:30 2025

@author: milenawaichnan
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy import signal

def mi_funcion_sen (A0, offset, f0, phase, nn, fs):
    tiempo = np.arange(0, N * Ts, Ts)
    x = A0 * np.sin(2 * np.pi * f0 * tiempo)
    x = x - x.mean()
    x = x / np.sqrt(np.var(x))
    var_x = np.var(x)
    return tiempo, x, var_x

N = 1000
fs = N   ##esto quiere decir que yo voy a tomar 1000 muestras por segundo. 
deltaF = fs / N
Ts = 1 / fs
freqs = np.arange(N) * deltaF

tiempo1, x1, var_x1 = mi_funcion_sen(A0 = 1, offset = 0, f0 = N/4 * deltaF , phase = 0, nn = N, fs = fs/2)
tiempo2, x2, var_x2 = mi_funcion_sen(A0 = 1, offset = 0, f0 = ((N/4) + 0.25) * deltaF, phase = 0, nn = N, fs = fs)
tiempo3, x3, var_x3 = mi_funcion_sen(A0 = 1, offset = 0, f0 = ((N/4) + 0.5) * deltaF , phase = 0, nn = N, fs = fs)


X1 = fft(x1)
X1abs = 1/N * np.abs(X1)


X2 = fft(x2)
X2abs = 1/N * np.abs(X2)


X3 = fft(x3)
X3abs = 1/N * np.abs(X3)



# Graficar solo hasta N/2
plt.figure()
#plt.stem(freqs, X1abs, 'x', label = 'X1 abs')
plt.plot(freqs, 10 * np.log10(X1abs ** 2), 'x', label = 'PSD de X1 (N/4)')
#plt.stem(freqs, X2abs, 'o', label ='X2 abs')
plt.plot(freqs, 10 * np.log10(X2abs ** 2), 'x', label = 'PSD de X2 (N/4 + 0,25)')
#plt.stem(freqs, X3abs, 'x', label = 'X3 abs')
plt.plot(freqs, 10 * np.log10(X3abs ** 2), 'x', label = 'PSD de X3 (N/4 + 0,5)')
plt.legend()

plt.title("Densidades espectrales de potencia")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [dB]")
plt.grid(True)
plt.xlim([0, fs/2])
plt.show()

# Graficar PSDs en subplots para visualizar mejor los picos
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(freqs, 10 * np.log10(2 * X1abs ** 2),"x", color='blue')
plt.title('PSD de X1 (f0 = N/4)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid(True)
plt.xlim([200, 300])

plt.subplot(3, 1, 2)
plt.plot(freqs, 10 * np.log10(2 * X2abs ** 2), "x", color='orange')
plt.title('PSD de X2 (f0 = N/4 + 0.25)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid(True)
plt.xlim([200, 300])

plt.subplot(3, 1, 3)
plt.plot(freqs, 10 * np.log10(2 * X3abs ** 2), "x", color='green')
plt.title('PSD de X3 (f0 = N/4 + 0.5)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid(True)
plt.xlim([200, 300])

plt.tight_layout()
plt.show()

#Parseval
potT_x1 = 1/N * np.sum(np.abs(x1) ** 2)
potT_x2 = 1/N * np.sum(np.abs(x2) ** 2)
potT_x3 = 1/N * np.sum(np.abs(x3) ** 2)
potF_X1 = 1/N ** 2 * np.sum(np.abs(X1) ** 2)
potF_X2 = 1/N ** 2 * np.sum(np.abs(X2) ** 2)
potF_X3 = 1/N ** 2 * np.sum(np.abs(X3) ** 2)

def Parseval(potT_x, PotF_X, tol=1e-10):
    if np.isclose(potT_x, PotF_X, rtol = 1e-10, atol = 1e-12):
        print("Se cumple Parseval")
    else:
        print("No se cumple Parseval")


print("Para X1: ")        
ParsevalX1 = Parseval(potT_x1, potF_X1)
print("Para X2: ")        
ParsevalX2 = Parseval(potT_x2, potF_X2)
print("Para X3: ")        
ParsevalX3 = Parseval(potT_x3, potF_X3)


##Zero padding

zeroPadding_x1 = np.zeros(10 * N)
zeroPadding_x2 = np.zeros(10 * N)
zeroPadding_x3 = np.zeros(10 * N)

zeroPadding_x1[0:N] = x1 
zeroPadding_x2[0:N] = x2
zeroPadding_x3[0:N] = x3

fft_zeroPadding_x1 = 1/N * np.abs(fft(zeroPadding_x1))
fft_zeroPadding_x2 = 1/N * np.abs(fft(zeroPadding_x2))
fft_zeroPadding_x3 = 1/N * np.abs(fft(zeroPadding_x3))

deltaF1 = fs / (10*N)
freqs1 = np.arange(10 * N) * deltaF1
#freq1 = np.abs(fft_zeroPadding) ** 2

#Graficos
plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(freqs1, fft_zeroPadding_x1,label = 'Zero Padding X1')
plt.plot(freqs, X1abs, '--', label = 'X1 (N/4)')
plt.title('Zero padding: fs/4')
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD")
plt.grid(True)
plt.legend()
plt.xlim(220, 280)

plt.subplot(3,1,2)
plt.plot(freqs1, fft_zeroPadding_x2,label = 'Zero Padding X2')
plt.plot(freqs, X2abs, '--', label = 'X2 (N/4+0,25)')
plt.title('Zero padding: fs/4')
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD")
plt.grid(True)
plt.legend()
plt.xlim(220, 280)

plt.subplot(3,1,3)
plt.plot(freqs1, fft_zeroPadding_x3,label = 'Zero Padding X3')
plt.plot(freqs, X3abs, '--', label = 'X2 (N/4+0,5)')
plt.title('Zero padding: fs/4')
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD")
plt.grid(True)
plt.legend()
plt.xlim(220, 280)

plt.tight_layout()
plt.show()

#Graficos en dB
plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(freqs1, fft_zeroPadding_x1,label = 'Zero Padding X1')
plt.plot(freqs, X1abs, '--', label = 'X1 (N/4)')
plt.title('Zero padding: fs/4')
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD")
plt.grid(True)
plt.legend()
plt.xlim(220, 280)

plt.subplot(3,1,2)
plt.plot(freqs1, np.log10(fft_zeroPadding_x2)*10,label = 'Zero Padding X2')
plt.plot(freqs, 10 * np.log10(X2abs), '--', label = 'X2 (N/4+0,25)')
plt.title('Zero padding: fs/4')
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [dB]")
plt.grid(True)
plt.legend()
plt.xlim(220, 280)

plt.subplot(3,1,3)
plt.plot(freqs1, np.log10(fft_zeroPadding_x3)*10,label = 'Zero Padding X3')
plt.plot(freqs, 10 * np.log10(X3abs), '--', label = 'X2 (N/4+0,5)')
plt.title('Zero padding: fs/4')
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [dB]")
plt.grid(True)
plt.legend()
plt.xlim(220, 280)

plt.tight_layout()
plt.show()

#Calculo de rta en frecuencia de los sistemas LTI
tiempo0, x0, var_x0 = mi_funcion_sen(A0 = 1, offset = 0, f0 = 50000 , phase = 0, nn = N, fs = fs)
freqsLTI = np.fft.fftfreq(N, Ts)
delta = np.zeros(len(x0))
delta[0] = 1

#sistema 1
a0 = np.array([1, -1.5, 0.5]) #coeficientes de y
b0 = np.array([0.03, 0.05, 0.03]) #coeficientes de x

h0 = signal.lfilter(b0, a0, delta)
fft_h0 = fft(h0)

#sistema 2
a1 = np.array([1]) #coeficientes de y
b1 = np.zeros(11) #coeficientes de x
b1[0] = 1
b1[10] = 3

h1 = signal.lfilter(b1, a1, delta)
fft_h1 = fft(h1)

#sistema 3
a2 = np.zeros(11) #coeficientes de y
a2[0] = 1
a2[10] = -3
b2 = np.array([1]) #coeficientes de x

h2 = signal.lfilter(b2, a2, delta)
fft_h2 = fft(h2)

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(freqsLTI, np.log10(np.abs(fft_h0))*10,label = 'Rta al impulso en frecuencia 1')
plt.title('Rta al impulso: y[n]-1.5⋅y[n−1]+0.5⋅y[n−2]=3⋅10−2⋅x[n]+5⋅10−2⋅x[n−1]+3⋅10−2⋅x[n−2]')
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.legend()

plt.subplot(3,1,2)
plt.plot(freqsLTI, np.log10(np.abs(fft_h1))*10,label = 'Rta al impulso en frecuencia 2')
plt.title('Rta al impulso: y[n]=x[n]+3⋅x[n−10] ')
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.legend()

plt.subplot(3,1,3)
plt.plot(freqsLTI, np.log10(np.abs(fft_h2))*10,label = 'Rta ak impulso en frecuencia 3')
plt.title('Rta al impulso: y[n]-3⋅y[n−10]=x[n]')
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()









