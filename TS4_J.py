#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 19:33:09 2025

@author: milenawaichnan
"""

import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

N = 1000
fs = N
omega0 = np.pi/2

Ts = 1/fs
k = np.arange(N)*Ts
t = np.arange(N)*Ts

"""
def senoidal_estocastica(N, omega0, A0=2): 
    k = np.arange(N)
    #fr = np.random.uniform(-2,2)   
    #DeltaOmega = 2*np.pi/N
    omega1 = omega0 #+ fr * DeltaOmega
    x = A0*np.sin(omega1*k) 
    x = x - x.mean() ## aca le saco la media
    x = x / np.sqrt(np.var(x)) ## aca lo divido por la des estandar
    var_x = np.var(x) ## calculo la varianza
    return x,var_x 
"""
def senoidal_estocastica(N, fs=fs/4, A0=2.0, fase=0.0):
    Ts = 1/fs
    t = np.arange(N)*Ts
    deltaF = fs/N
    fr = np.random.uniform(-2, 2)
    f1 = (fs/4) + fr*deltaF
    x = A0*np.sin(2*np.pi*((fs/4)+0.5)*t + fase)
    x = x - x.mean() ## aca le saco la media
    x = x / np.sqrt(np.var(x)) ## aca lo divido por la des estandar
    var_x = np.var(x) ## calculo la varianza
    return x,var_x
# ---------- SNR y ruido ----------
SNRdb = np.random.uniform(3, 10)
print(f"{SNRdb:3.5f}")

def ruido_para_snr(N, SNRdb):
    var_n = 10**(-SNRdb/10)
    std_n = np.sqrt(var_n) ##std desviacion estandar
    n = np.random.normal(0, std_n, N)
    return n, var_n

#x, var_x = senoidal_estocastica(N)
ruido, var_ruido = ruido_para_snr(N, SNRdb)

fr = np.random.uniform(-2, 2)
x = A0*np.sin(2 * np.pi * ((N/4) + fr) * df * t)

                          
xn = x + ruido ## modelo de señal

var_xn = np.var(xn)
print(f"{var_xn:3.1f}")
print(f"{var_x:3.1f}")
print(f"{var_ruido:3.1f}")

X = fft(x) * 1/N
Xabs = 2* np.abs(X)

R = fft(ruido) * 1/N
Rabs = np.abs(R)

Xn = 1/N* fft(xn)
Xnabs = np.abs(Xn)


plt.figure()
plt.title("FFT")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X[k]|")
plt.grid(True)
#plt.plot(k,  np.log10(Xabs) * 20, label = 'X')
plt.plot(k, 10*np.log10(Xabs**2) , label = 'dens esp pot') ##densidad espectral de potencia
plt.xlim((0,fs/2))
#plt.plot(k, 2* np.log10(Rabs) * 20, label = 'Ruido')
#plt.plot(k, np.log10(Xnabs) * 20, label = 'Modelo de señal')
plt.legend()




