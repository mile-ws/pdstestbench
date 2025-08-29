#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 21:08:55 2025

@author: milenawaichnan
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

def mi_funcion_sen (A0, offset, fx, phase, nn, fs):
    tiempo = np.arange(0, N * Ts, Ts)
    x = A0 * np.sin(2 * np.pi * fx * tiempo)
    return tiempo, x

#todos estos parametros sirven si no defino mi funcion, uso np.sin(2 * np.pi * fx * tiempo) directamente
N = 1000
#fs = 500 ## si mi fs != a N estoy cambiando mi relacion de 1/deltaf = N.Ts = 1 y cambio el tiempo.
fs = N   ##esto quiere decir que yo voy a tomar 1000 muestras por segundo. 
deltaF = fs / N
Ts = 1 / fs
#fx = 1000


# grilla temporal
tiempo1, x1 = mi_funcion_sen(A0 = 1, offset = 0, fx = 1 , phase = 0, nn = N, fs = fs)

tiempo2, x2 = mi_funcion_sen(A0 = np.sqrt(2), offset = 0, fx = (N/4) * deltaF, phase = 0, nn = N, fs = fs)
tiempo3, x3 = mi_funcion_sen(A0 = 1, offset = 0, fx = ((N/4) + 0.5) * deltaF , phase = 0, nn = N, fs = fs)

X1 = fft(x1)
X1abs = np.abs(X1)
X1ang = np.angle(X1)

X2 = fft(x2)
X2abs = np.abs(X2)
X2ang = np.angle(X2)

X3 = fft(x3)
X3abs = np.abs(X3)
X3ang = np.angle(X3)


freqs = np.arange(N) * deltaF

# Graficar solo hasta N/2
plt.figure()
#plt.stem(freqs, X1abs, 'x', label = 'X1 abs')
plt.plot(freqs, np.log10(X1abs) * 20, 'x', label = 'X1 abs dB')
#plt.stem(freqs, X2abs, 'o', label ='X2 abs')
plt.plot(freqs, np.log10(X2abs) * 20, 'x', label = 'X2 abs dB')
#plt.stem(freqs, X3abs, 'x', label = 'X3 abs')
plt.plot(freqs, np.log10(X3abs) * 20, 'x', label = 'X3 abs dB')
plt.legend()

plt.title("Comparacion")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.xlim([0, fs/2])
# plt.show()


# %%

varianza = np.var(x2)
print(varianza)


modulo_cuadrado = np.abs(X2) ** 2
#moduloDB = 10 * np.log10(modulo_cuadrado)

plt.figure()
plt.plot(freqs, np.log10(modulo_cuadrado) * 10, 'x', label = 'X1 abs dB')
plt.legend()


sumaModulo = np.sum(modulo_cuadrado)
sumaCuadrado = np.sum(x2 ** 2)

if sumaModulo == sumaCuadrado:
    print("Se cumple Parseval")
else: 
    print("No se cumple Parseval")
