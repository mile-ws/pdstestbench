#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 20:46:52 2025

@author: milenawaichnan
"""

import matplotlib.pyplot as plt
import numpy as np
#import scipy as sc
from scipy import signal
from numpy.fft import fft

fs = 1000 # frecuencia de muestreo, aplica para todas las señales
N = fs  # cantidad de muestras, aplica para todas las señales 
#fx = 2000 #frecuencia para las senoidales
Ts = 1/fs
deltaF = fs/N
fr = 0

A0 = np.sqrt(2)

omega0 = fs / 4
omega1 = omega0 + (fr * deltaF)



SNRdb = np.random.uniform(3, 10)
sigmaCuad = 10**(-SNRdb/10)
print("varianza = ", sigmaCuad)



n = np.arange(N)
s = A0 * np.sin(omega1 * n)
varS = np.var(s)
print("Var s: ", varS)

S_fft= fft(s)
freqs = np.arange(N) * deltaF

plt.figure()
plt.plot(freqs, 20*np.log10(np.abs(S_fft)), label = 'S abs dB')
plt.legend()

Na = np.random.normal(0, np.sqrt(sigmaCuad), N) #ruido
#Na = 3
#print(Na)
varNa = np.var(Na)
print("Var Na: ", varNa)

x = s + Na
varX = np.var(x)
print("Var x: ", varX)

X_fft= fft(x)
Xabs = np.abs(X_fft)
Xang = np.angle(x)
Xcuad = Xabs ** 2

# freqs = np.arange(N) * deltaF
# plt.figure()
# plt.plot(freqs, np.log10(Xcuad) * 10, label = 'X1 abs dB')
# plt.legend()

# #ventanas
# ventana_rectangular = np.ones(N)
# ventana_BH = signal.windows.blackmanharris(N)
# ventana_Hamming = signal.windows.hamming(N)
# ventana_Hann = signal.windows.hann(N)
# ventana_FT = signal.windows.flattop(N)

# # Aplicar ventana a la señal
# x_rect = x * ventana_rectangular
# x_hamming = x * ventana_Hamming
# x_hann = x * ventana_Hann
# x_BH = x * ventana_BH
# x_FT = x * ventana_FT

# # FFT de cada una
# X_rect = np.abs(fft(x_rect))**2
# X_hamming = np.abs(fft(x_hamming))**2
# X_hann = np.abs(fft(x_hann))**2
# X_BH = np.abs(fft(x_BH))**2
# X_FT = np.abs(fft(x_FT))**2

# # Frecuencias
# freqs = np.arange(N) * deltaF

# # Graficar todas
# plt.figure(figsize=(12,6))
# plt.plot(freqs, 10*np.log10(X_rect), label="Rectangular")
# plt.plot(freqs, 10*np.log10(X_hamming), label="Hamming")
# plt.plot(freqs, 10*np.log10(X_hann), label="Hann")
# plt.plot(freqs, 10*np.log10(X_BH), label="Blackman-Harris")
# plt.plot(freqs, 10*np.log10(X_FT), label="Flat-top")

# plt.xlim(0, fs/2)   # me quedo con la mitad del espectro
# plt.xlabel("Frecuencia [Hz]")
# plt.ylabel("Magnitud [dB]")
# plt.title("Comparación de ventanas en el espectro")
# plt.legend()
# plt.grid(True)
# plt.show()


  

