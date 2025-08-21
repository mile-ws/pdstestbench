#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 18:51:49 2025

@author: milenawaichnan
"""
import matplotlib.pyplot as plt
import numpy as np


#todos estos parametros sirven si no defino mi funcion, uso np.sin(2 * np.pi * fx * tiempo) directamente
fs = 100000 ## si mi fs != a N estoy cambiando mi relacion de 1/deltaf = N.Ts = 1 y cambio el tiempo.
N = 500  ##esto quiere decir que yo voy a tomar 1000 muestras por segundo. 
deltaF = fs/N
Ts = 1/(N*deltaF)
fx = 2000

def mi_funcion_sen (A0, fx, phase, nn, fs):
    tiempo = np.arange(0, N * Ts, Ts)
    x = A0 * np.sin(2 * np.pi * fx * tiempo)
    return tiempo, x


tiempo, x = mi_funcion_sen(A0 = 1, fx = fx, phase = 0, nn = N, fs = fs)
##modular
tiempo, x_aux = mi_funcion_sen(A0 = 1, fx = fx/2, phase = 0, nn = N, fs = fs)
x1 = x * x_aux

##recorto el 75% de la amplitud

x2 = np.clip(x, -0.75, 0.75, out=None)


plt.subplot(2,2,1)
plt.plot(tiempo, x, label='Señal senoidal de 2kHz')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [v]')
plt.grid(True)

# Misma señal modulada en amplitud por otra señal sinusoidal de la mitad de la frecuencia.

plt.subplot(2,2,2)
plt.plot(tiempo, x1, label='Señal modulada con otra señal de la mitad de la f')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [v]')
plt.grid(True)

plt.subplot(2,2,3)
plt.plot(tiempo, x2, label='recorto al 75% de la señal')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [v]')
plt.grid(True)

plt.legend()
plt.show()