#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 18:45:59 2025

@author: milenawaichnan
;s6a555x555555555555555555555555555555555555555555555555555555555z56zxz5Z466$"""


#aca hago la senoideal normal la comunacha
import matplotlib.pyplot as plt
import numpy as np

#n = np.arange(0,2*np.pi, 0.1)
#x = np.sin(n)

#fs/2: Nyquist, deltaF=fs/N, 1/deltaF=N*Ts

def mi_funcion_sen (A0, offset, fx, phase, nn, fs):
    tiempo = np.arange(0, N * Ts, Ts)
    x = A0 * np.sin(2 * np.pi * fx * tiempo)
    return tiempo, x

#todos estos parametros sirven si no defino mi funcion, uso np.sin(2 * np.pi * fx * tiempo) directamente
fs = 1000 ## si mi fs != a N estoy cambiando mi relacion de 1/deltaf = N.Ts = 1 y cambio el tiempo.
N = fs   ##esto quiere decir que yo voy a tomar 1000 muestras por segundo. 
deltaF = fs/N
Ts = 1/(N*deltaF)
fx = 1000


tiempo, x = mi_funcion_sen(A0 = 1, offset = 0, fx = 1, phase = 0, nn = N, fs = fs)
plt.figure(figsize=(8,4))
plt.plot(tiempo, x, label='Señal senoidal')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [v]')
plt.grid(True)
plt.legend()
plt.show()


#ff = 500 Hz
tiempo, x = mi_funcion_sen(A0 = 1, offset = 0, fx = 500, phase = 0, nn = N, fs = fs)
plt.figure(figsize=(8,4))
plt.plot(tiempo, x, label='Señal senoidal')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [v]')
plt.grid(True)
plt.legend()
plt.show()

#ff = 999 Hz
tiempo, x = mi_funcion_sen(A0 = 1, offset = 0, fx = 999, phase = 0, nn = N, fs = fs)
plt.figure(figsize=(8,4))
plt.plot(tiempo, x, label='Señal senoidal')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [v]')
plt.grid(True)
plt.legend()
plt.show()

#ff = 1001 Hz
tiempo, x = mi_funcion_sen(A0 = 1, offset = 0, fx = 1001, phase = 0, nn = N, fs = fs)
plt.figure(figsize=(8,4))
plt.plot(tiempo, x, label='Señal senoidal')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [v]')
plt.grid(True)
plt.legend()
plt.show()

#ff = 2001 Hz
tiempo, x = mi_funcion_sen(A0 = 1, offset = 0, fx = 2001, phase = 0, nn = N, fs = fs)
plt.figure(figsize=(8,4))
plt.plot(tiempo, x, label='Señal senoidal')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [v]')
plt.grid(True)
plt.legend()
plt.show()










