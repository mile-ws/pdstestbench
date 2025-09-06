#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 12:07:41 2025

@author: milenawaichnan
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros fisiológicos
C = 5.0       # mL/mmHg
R = 0.2       # mmHg·s/mL
Ts = 0.01     # paso temporal [s]
T_total = 5   # tiempo total [s] para varios latidos
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
