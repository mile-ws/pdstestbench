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
    
delta = np.zeros(len(x_senoidal))
delta[0] = 1

h = signal.lfilter(b, a, delta)  #respuesta al impulso

y_conv = np.convolve(x_senoidal, h, mode='same')#[:len(x_senoidal)] #salida    
y_conv_full = np.convolve(x_senoidal, h, mode= 'full')
t_full = np.arange(len(y_conv_full)) / fs

plt.figure()
plt.plot(t_full*1000, y_conv_full, label='Convolución completa', color='C0')
plt.plot(t[:len(y_conv)]*1000, y_conv, label='Convolución (recortada)', color='C1')
plt.axvline(len(x_senoidal)/fs*1000, color='r', linestyle='--', label='Fin señal original')
plt.xlabel('Tiempo [ms]')
plt.ylabel('Amplitud')
plt.title('Efecto de borde en la convolución')
plt.xlim(0, 30) 
plt.legend()
plt.grid(True)

#Graficos
fig, axs = plt.subplots(7, 1, figsize=(10, 12))  # 6 filas, 1 columna

# Señal senoidal
axs[0].plot(t, salidas[0], label = f"P = {np.mean(salidas[0]**2):.2f}")
axs[0].plot(t, x_senoidal, label = "Entrada")
axs[0].set_title("Señal senoidal 2 kHz")
axs[0].set_xlabel("Tiempo [s]")
axs[0].set_ylabel("Amplitud [V]")
axs[0].set_xlim(0, 0.005)
axs[0].legend()
axs[0].grid(True)

# Amplificada y desfasada
axs[1].plot(t, salidas[1], label = f"P = {np.mean(salidas[1]**2):.2f}")
axs[1].plot(t, x_desfazada, label = "Entrada")
axs[1].set_title("Amplificada y desfasada")
axs[1].set_xlabel("Tiempo [s]")
axs[1].set_ylabel("Amplitud [V]")
axs[1].set_xlim(0, 0.005)
axs[1].legend()
axs[1].grid(True)

# Modulacion en amplitud
axs[2].plot(t, salidas[2], label = f"P = {np.mean(salidas[2]**2):.2f}")
axs[2].plot(t, x_modulada, label = "Entrada")
axs[2].set_title("Modulada con sen f/2")
axs[2].set_xlabel("Tiempo [s]")
axs[2].set_ylabel("Amplitud [V]")
axs[2].set_xlim(0, 0.005)
axs[2].legend()
axs[2].grid(True)

# Señal recortada
axs[3].plot(t, salidas[3], label = f"P = {np.mean(salidas[3]**2):.2f}")
axs[3].plot(t, x_recortada, label = "Entrada")
axs[3].set_title("Recorte al 75%")
axs[3].set_xlabel("Tiempo [s]")
axs[3].set_ylabel("Amplitud [V]")
axs[3].set_xlim(0, 0.005)
axs[3].legend()
axs[3].grid(True)

# Señal cuadrada
axs[4].plot(t, salidas[4], label = f"P = {np.mean(salidas[4]**2):.2f}")
axs[4].plot(t, x_cuadrada, label = "Entrada")
axs[4].set_title("Señal cuadrada")
axs[4].set_xlabel("Tiempo [s]")
axs[4].set_ylabel("Amplitud [V]")
axs[4].set_xlim(0, 0.005)
axs[4].legend()
axs[4].grid(True)

# Entrada (pulso)
axs[5].plot(t, x_pulsos, "orange", label="Entrada (Pulso 10 ms)")
axs[5].set_title("Entrada Pulso 10 ms")
axs[5].set_xlabel("Tiempo [s]")
axs[5].set_ylabel("Amplitud [V]")
axs[5].set_xlim(0, 0.02)
axs[5].legend()
axs[5].grid(True)

# Salida (respuesta al pulso)
axs[6].plot(t, salidas[5], label=f"Salida – E = {np.sum(salidas[5]**2):.2f}")
axs[6].set_title("Salida del sistema ante pulso de 10 ms")
axs[6].set_xlabel("Tiempo [s]")
axs[6].set_ylabel("Amplitud [V]")
axs[6].set_xlim(0, 0.02)   
axs[6].legend()
axs[6].grid(True)



# Ajustar todo
plt.tight_layout()
plt.show()
    
#CALCULAR LA RESPUESTA AL IMPULSO

#senoidal
delta_sen = np.zeros(len(x_senoidal))
delta_sen[0] = 1
h_sen = signal.lfilter(b, a, delta_sen)  #respuesta al impulso
y_conv_sen = np.convolve(x_senoidal, h_sen)[:len(x_senoidal)] #salida

#amplificada
delta_amp = np.zeros(len(x_desfazada))
delta_amp[0] = 1
h_amp = signal.lfilter(b, a, delta_amp)  #respuesta al impulso
y_conv_amp = np.convolve(x_desfazada, h_amp)[:len(x_desfazada)] #salida

#modulada
delta_mod = np.zeros(len(x_modulada))
delta_mod[0] = 1
h_mod = signal.lfilter(b, a, delta_mod)  #respuesta al impulso
y_conv_mod = np.convolve(x_modulada, h_mod)[:len(x_modulada)] #salida

#recortada
delta_rec = np.zeros(len(x_recortada))
delta_rec[0] = 1
h_rec = signal.lfilter(b, a, delta_rec)  #respuesta al impulso
y_conv_rec = np.convolve(x_recortada, h_rec)[:len(x_recortada)] #salida

#cuadrada
delta_cuad = np.zeros(len(x_cuadrada))
delta_cuad[0] = 1
h_cuad = signal.lfilter(b, a, delta_cuad)  #respuesta al impulso
y_conv_cuad = np.convolve(x_cuadrada, h_cuad)[:len(x_cuadrada)] #salida

#pulso
delta_pul = np.zeros(len(x_pulsos))
delta_pul[0] = 1
h_pul = signal.lfilter(b, a, delta_pul)  #respuesta al impulso
y_conv_pul = np.convolve(x_pulsos, h_pul)[:len(x_pulsos)] #sa"lida




#Graficos
fig, axs = plt.subplots(6, 1, figsize=(10, 12))  # 6 filas, 1 columna

# Señal senoidal
axs[0].plot(t, salidas[0], label ="Salida Y")
axs[0].plot(t, y_conv_sen, "o", color='orange' , label="Y convolucion")
axs[0].set_title("Señal de salida senoidal")
axs[0].set_xlabel("Tiempo [s]")
axs[0].set_ylabel("Amplitud")
axs[0].set_xlim(0, 0.005)
axs[0].legend()
axs[0].grid(True)

# Amplificada y desfasada
axs[1].plot(t, salidas[1], label = "Salida Y")
axs[1].plot(t, y_conv_amp, "o", color='orange' , label = "Y convolucion")
axs[1].set_title("Señal de salida Amplificada y desfasada")
axs[1].set_xlabel("Tiempo [s]")
axs[1].set_ylabel("Amplitud")
axs[1].set_xlim(0, 0.005)
axs[1].legend()
axs[1].grid(True)

# Modulacion en amplitud
axs[2].plot(t, salidas[2], label = "Salida Y")
axs[2].plot(t, y_conv_mod,"o", color='orange' , label = "Y convolucion")
axs[2].set_title("Señal de salida Modulada con sen f/2")
axs[2].set_xlabel("Tiempo [s]")
axs[2].set_ylabel("Amplitud")
axs[2].set_xlim(0, 0.005)
axs[2].legend()
axs[2].grid(True)

# Señal recortada
axs[3].plot(t, salidas[3], label ="Salida Y")
axs[3].plot(t, y_conv_rec,"o", color='orange' , label = "Y convolucion")
axs[3].set_title("Señal de salida Recorte al 75%")
axs[3].set_xlabel("Tiempo [s]")
axs[3].set_ylabel("Amplitud")
axs[3].set_xlim(0, 0.005)
axs[3].legend()
axs[3].grid(True)

# Señal cuadrada
axs[4].plot(t, salidas[4], label = "Salida Y")
axs[4].plot(t, y_conv_cuad,"o", color='orange' , label = "Y convolucion")
axs[4].set_title("Señal de salida cuadrada")
axs[4].set_xlabel("Tiempo [s]")
axs[4].set_ylabel("Amplitud")
axs[4].set_xlim(0, 0.005)
axs[4].legend()
axs[4].grid(True)

# Pulso rectangular
axs[5].plot(t, salidas[5], label = "Salida Y")
axs[5].plot(t, y_conv_pul, "o", color='orange' , label = "Y convolucion")
axs[5].set_title("Pulso 10 ms")
axs[5].set_xlabel("Tiempo [s]")
axs[5].set_ylabel("Amplitud")
axs[4].set_xlim(0, 0.005)
axs[5].legend()
axs[5].grid(True)

# Ajustar todo
plt.tight_layout()
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
plt.plot(t, h1, "-",color='red' , label="Respuesta al impulso")
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
plt.plot(t, h2, "-",color='red' , label="Respuesta al impulso")
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






