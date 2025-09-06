#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 19:58:07 2025

@author: milenawaichnan
"""

import matplotlib.pyplot as plt
import numpy as np
#import scipy as sc
from scipy import signal

#𝑦[𝑛]-1.5⋅𝑦[𝑛−1]+0.5⋅𝑦[𝑛−2]=3⋅10−2⋅𝑥[𝑛]+5⋅10−2⋅𝑥[𝑛−1]+3⋅10−2⋅𝑥[𝑛−2]
 

a = np.array([1, -1.5, 0.5]) #coeficientes de y
b = np.array([0.03, 0.05, 0.03]) #coeficientes de x

fs = 50000 # frecuencia de muestreo, aplica para todas las señales
N = 700  # cantidad de muestras, aplica para todas las señales 
fx = 2000 #frecuencia para las senoidales
Ts = 1/fs
t = np.arange(N) * Ts


def mi_funcion_sen (A0, fx, phase, t ):
    x = A0 * np.sin(2 * np.pi * fx * t + phase)
    return x

def funcion_cuadrada (fcuad):
    x = signal.square(2 * np.pi * fcuad * t)
    return x

def pulsos(t, T):
    pulso = np.where(t < T, 1.0, 0.0)
    return pulso

#señal original
x_senoidal = mi_funcion_sen(A0 = 1, fx = fx, phase = 0, t = t)

#señal desfazada
x_desfazada = mi_funcion_sen(A0 = 1, fx = fx, phase = np.pi/2, t = t)

#señal modulada con una de la mitad de su frecuencia
x_aux = mi_funcion_sen(A0 = 1, fx = fx/2, phase = 0, t = t)
x_modulada = x_senoidal * x_aux

#señal recortada al 75%
x_recortada = np.clip(x_senoidal, -0.75, 0.75, out=None)

#señal cuadrada 4KHz
x_cuadrada = funcion_cuadrada(fcuad = 4000)

#pulso de 10ms
x_pulsos = pulsos(t, 0.01)

entradas = [x_senoidal, x_desfazada, x_modulada, x_recortada, x_cuadrada, x_pulsos]

salidas = []
potencia = []
energia = []
for i, x in enumerate(entradas):
    y = signal.lfilter(b, a, x)
    salidas.append(y)
    
    
    
#Graficos
fig, axs = plt.subplots(6, 1, figsize=(10, 12))  # 6 filas, 1 columna

# Señal senoidal
axs[0].plot(t, salidas[0], label = f"P = {np.mean(salidas[0]**2):.2f}")
axs[0].set_title("Señal senoidal 2 kHz")
axs[0].set_xlabel("Tiempo [s]")
axs[0].set_ylabel("Amplitud [V]")
axs[0].legend()
axs[0].grid(True)

# Amplificada y desfasada
axs[1].plot(t, salidas[1], label = f"P = {np.mean(salidas[1]**2):.2f}")
axs[1].set_title("Amplificada y desfasada")
axs[1].set_xlabel("Tiempo [s]")
axs[1].set_ylabel("Amplitud [V]")
axs[1].legend()
axs[1].grid(True)

# Modulacion en amplitud
axs[2].plot(t, salidas[2], label = f"P = {np.mean(salidas[2]**2):.2f}")
axs[2].set_title("Modulada con sen f/2")
axs[2].set_xlabel("Tiempo [s]")
axs[2].set_ylabel("Amplitud [V]")
axs[2].legend()
axs[2].grid(True)

# Señal recortada
axs[3].plot(t, salidas[3], label = f"P = {np.mean(salidas[3]**2):.2f}")
axs[3].set_title("Recorte al 75%")
axs[3].set_xlabel("Tiempo [s]")
axs[3].set_ylabel("Amplitud [V]")
axs[3].legend()
axs[3].grid(True)

# Señal cuadrada
axs[4].plot(t, salidas[4], label = f"P = {np.mean(salidas[4]**2):.2f}")
axs[4].set_title("Señal cuadrada")
axs[4].set_xlabel("Tiempo [s]")
axs[4].set_ylabel("Amplitud [V]")
axs[4].legend()
axs[4].grid(True)

# Pulso rectangular
axs[5].plot(t, salidas[5], label = f"E = {np.sum(salidas[5]**2):.2f}")
axs[5].set_title("Pulso 10 ms")
axs[5].set_xlabel("Tiempo [s]")
axs[5].set_ylabel("Amplitud [V]")
axs[5].legend()
axs[5].grid(True)

# Ajustar todo
plt.tight_layout()
plt.show()
    

##CALCULAR LA RESPUESTA AL IMPULSO
delta = np.zeros(len(x_senoidal))
delta[0] = 1

h = signal.lfilter(b, a, delta)  #respuesta al impulso

y_conv = np.convolve(x_senoidal, h)[:len(x_senoidal)] #salida

plt.figure()
plt.plot(t, y_conv, "o", color='orange' , label="Y convolucion")
plt.plot(t, salidas[0], label="Salida Y")
plt.title("Señal de salida senoidal")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend()
plt.grid(True)
plt.show()

##2) HALLAR RTA AL IMPULSO Y SALIDA A UNA SENAL SENOIDAL

#𝑦[𝑛]=𝑥[𝑛]+3⋅𝑥[𝑛−10]

a1 = np.array([1]) #coeficientes de y


b1 = np.zeros(11) #coeficientes de x
b1[0] = 1
b1[10] = 3

h1 = signal.lfilter(b1, a1, delta)  #respuesta al impulso
y1_conv= np.convolve(x_senoidal, h1)[:len(x_senoidal)]
y1 = signal.lfilter(b1, a1, x_senoidal)

plt.figure()
plt.plot(t, y1, label="Salida y1[n]")
plt.plot(t, y1_conv, "o",color='orange' , label="Y convolucion")
plt.title("Sistema 1")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)


#𝑦[𝑛]-3⋅𝑦[𝑛−10]=𝑥[𝑛]

a2 = np.zeros(11) #coeficientes de y
a2[0] = 1
a2[10] = -3

b2 = np.array([1]) #coeficientes de x

h2 = signal.lfilter(b2, a2, delta)  #respuesta al impulso
y2_conv = np.convolve(x_senoidal, h2)[:len(x_senoidal)]
y2 = signal.lfilter(b2, a2, x_senoidal)


plt.figure()
plt.plot(t, y2, label="Salida y2[n]")
plt.plot(t, y2_conv, "o",color='orange' , label="Y convolucion")
plt.title("Sistema 2")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Bonus Windkessel

# Parámetros fisiológicos (circulación sistémica)
C = 5.0     # mL/mmHg
R = 0.2     # mmHg·s/mL
Ts_wk = 0.01   # paso temporal [s]
T_wk = 2       # tiempo total [s]
N_wk = int(T_wk / Ts_wk)

# Señal de entrada: flujo Q(t)
t_wk = np.arange(N_wk) * Ts_wk
Q = 10 * (t_wk < 1)   # flujo constante de 10 mL/s durante 1 segundo

# Inicialización de presión
P = np.zeros(N_wk)
P[0] = 0  # condición inicial

# Coeficientes para Backward Euler
alpha = C / Ts_wk
den = alpha + 1 / R

# Iteración recursiva
for n in range(1, N_wk):
    P[n] = (alpha * P[n-1] + Q[n]) / den

# Graficar resultados
plt.figure(figsize=(10,5))
plt.plot(t_wk, Q, label="Flujo Q(t) [mL/s]", linestyle="--")
plt.plot(t_wk, P, label="Presión P(t) [mmHg]", linewidth=2)
plt.xlabel("Tiempo [s]")
plt.ylabel("Magnitud")
plt.title("Modelo Windkessel discretizado (Backward Euler)")
plt.legend()
plt.grid(True)
plt.show()

# Parámetros fisiológicos
C = 5.0       # mL/mmHg
R = 0.2       # mmHg·s/mL
Ts = 0.01     # paso temporal [s]
  # tiempo total [s] para varios latidos
N = int(T_total / Ts)

# Señal de entrada: flujo Q(t) pulsátil (simula 1 Hz, 60 latidos/min)
f_heart = 1.0  # Hz
t = np.arange(N) * Ts
Q = 10 * (np.sin(2 * np.pi * f_heart * t) > 0)  # tren de pulsos

# Inicialización de presión
P = np.zeros(N)
P[0] = 0  # condición inicial

# Coeficientes para Backward Euler
alpha = C / Ts
den = alpha + 1/R

# Iteración recursiva (Backward Euler)
for n in range(1, N):
    P[n] = (alpha * P[n-1] + Q[n]) / den

# Graficar resultados
plt.figure(figsize=(10,5))
plt.plot(t, Q, label="Flujo Q(t) [mL/s]", linestyle="--")
plt.plot(t, P, label="Presión P(t) [mmHg]", linewidth=2)
plt.xlabel("Tiempo [s]")
plt.ylabel("Magnitud")
plt.title("Modelo Windkessel con flujo pulsátil")
plt.legend()
plt.grid(True)
plt.show()





