#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 22:46:57 2025

@author: milenawaichnan
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import TS1joaco

# =============================================================================
# PARAMETROS GLOBALES
# =============================================================================

fs = 50000
Ts = 1/fs
xx = TS1joaco.xx
N  = len(xx) ##es 500 por el ts1
tt = np.arange(N) * Ts      

# =============================================================================
# Ejercicio 1
# HECHO --> Graficar la señal de salida para cada una de las señales de entrada que generó en el TS1. Considere que las mismas son causales.
# HECHO --> Hallar la respuesta al impulso y usando la misma, repetir la generación de la señal de salida para alguna de las señales de entrada consideradas en el punto anterior.
# En cada caso indique la frecuencia de muestreo, el tiempo de simulación y la potencia o energía de la señal de salida.
# =============================================================================

def en_diferencias(N,x):
    y = np.zeros(N)
    for n in range (N):
        x0 = x[n]
        x1 = x[n-1] if n-1 >= 0 else 0
        x2 = x[n-2] if n-2 >= 0 else 0
        y1 = y[n-1] if n-1 >= 0 else 0
        y2 = y[n-2] if n-2 >= 0 else 0
        y[n] = 3* 10**(-2)*x0 + 5 * 10**(-2)*x1 +  3 * 10**(-2)*x2 + 1.5*y1-0.5*y2
    return y

entradas = [
    (TS1joaco.xx,   "Seno principal"),
    (TS1joaco.x1,   "Seno desf. π/2"),
    (TS1joaco.x2,   "AM (f/2)"),
    (TS1joaco.x3,   "AM clipeada 75%"),
    (TS1joaco.x4,   "Cuadrada 4 kHz"),
    (TS1joaco.pulso,"Pulso 10 ms"),
]
tconv = 0
# convolucion de la entrada xx con delta
delta = np.zeros(len(xx))
delta[0] = 1
h = en_diferencias(N = N, x = delta)
y_conv = np.convolve(xx, h,'valid')#[:N]

fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
axs = axs.ravel()  # aplanar para poder usar axs[i]

for i, (x, nombre) in enumerate(entradas):
    y = en_diferencias(len(x), x)
    t = np.arange(len(x)) * Ts

    ax = axs[i]
    ax.plot(t, y, label="Salida", linewidth=1.5)
    ax.plot(t, x, '--', label="Entrada", linewidth=1.0)
    if i == 0:  # primer subplot corresponde a xx
        ax.plot(tconv, y_conv, linestyle='none', marker='o', markersize=2.5,
                label="h*xx (convolución)")

    ax.set_title(nombre)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.9)

fig.supxlabel("Tiempo [s]")
fig.supylabel("Amplitud")
fig.tight_layout()
plt.show()


#%%
# =========================
# Ejercicio 2
# =========================
# Sistema A (FIR): y[n] = x[n] + 3 x[n-10]

b2 = np.zeros(11)
b2[0] = 1.0
b2[10] = 3.0
a2 = np.array([1.0])
y2  = lfilter(b2, a2, xx)

delta = np.zeros(len(xx))
delta[0] = 1
h2  = lfilter(b2, a2, delta)               
y2_conv = np.convolve(xx, h2, mode='full')[:N]  # causal
#y2_conv = y2_conv1[:len(xx)]  


plt.figure(figsize=(10,4))
plt.plot(tt, y2, '--', label="y (lfilter)", linewidth=1.5)
plt.plot(tt, y2_conv, linestyle='none', marker='o', markersize=2.5, label="y (conv con h2)")
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud")
plt.title("Sistema A (FIR): y[n] = x[n] + 3·x[n−10]")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.show()

# Sistema B (IIR): y[n] = x[n] + 3 y[n-10]  (INESTABLE)

b3 = np.array([1.0])
a3 = np.zeros(11)
a3[0]  = 1.0   # y[n]
a3[10] = -3.0  # -3 y[n-10]
y3  = lfilter(b3, a3, xx)

h3 = lfilter(b3, a3, delta)     # [1, 0.., 3, 0.., 9, ...] hasta len(xx)

y3_conv = np.convolve(xx, h3, mode='full')[:N]    # mismo largo que xx
#y3_conv = y3_conv1[:len(TS1joaco.xx)]  

# --- Plot comparación --------------------------------------------------------

plt.figure(figsize=(10,4))
plt.plot(tt, y3, '--', label="y (lfilter IIR)", linewidth=1.5)
plt.plot(tt, y3_conv, linestyle='none', marker='o', markersize=2.5, label="y (conv con h3 trunc.)")
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud")
plt.title("Sistema B (IIR, inestable): y[n] = x[n] + 3·y[n−10]")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.show()

print(len(xx))
print(len(h))
print(len(y_conv))








