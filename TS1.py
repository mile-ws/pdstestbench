#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 18:51:49 2025

@author: milenawaichnan
"""
import matplotlib.pyplot as plt
import numpy as np
#import scipy as sc
from scipy import signal
from scipy.io import wavfile


#todos estos parametros sirven si no defino mi funcion, uso np.sin(2 * np.pi * fx * tiempo) directamente
fs = 20000## si mi fs != a N estoy cambiando mi relacion de 1/deltaf = N.Ts = 1 y cambio el tiempo.
N = 500  ##cantidad de muestras 
#deltaF = fs/N
fx = 2000
Ts = 1/fs


t = np.arange(N) * Ts 

def mi_funcion_sen (A0, fx, phase, t ):
   
    x = A0 * np.sin(2 * np.pi * fx * t + phase)
    return x

def funcion_cuadrada (fcuad):
    #t_cuadrada = np.linspace(0, 1, 1000)
    x = signal.square(2 * np.pi * fcuad * t)
    return x

def pulsos(t, T):
    #t_pulsos = np.arange(0, 0.05, 1/fs)  # 50 ms de tiempo
    # Pulso rectangular de 10 ms que arranca en t=0
    pulso = np.where(t < T, 1.0, 0.0)
    return pulso

#señal original
x_senoidal = mi_funcion_sen(A0 = 1, fx = fx, phase = 0, t = t)
potencia_x1 = np.mean((x_senoidal) ** 2)
print(potencia_x1)

#señal desfazada
x_desfazada = mi_funcion_sen(A0 = 1, fx = fx, phase = np.pi/2, t = t)
potencia_x2 = np.mean((x_desfazada) ** 2)
print(potencia_x2)

#señal modulada con una de la mitad de su frecuencia
x_aux = mi_funcion_sen(A0 = 1, fx = fx/2, phase = 0, t = t)
x_modulada = x_senoidal * x_aux
potencia_x3 = np.mean((x_modulada) ** 2)
print(potencia_x3)

#señal recortada al 75%
x_recortada = np.clip(x_senoidal, -0.75, 0.75, out=None)
potencia_x4 = np.mean((x_recortada) ** 2)
print(potencia_x4)

#señal cuadrada 4KHz
x_cuadrada = funcion_cuadrada(fcuad = 4000)
potencia_x5 = np.mean((x_cuadrada) ** 2)
print(potencia_x5)

#pulso de 10ms
x_pulsos = pulsos(t, 0.01)
energia_x6 = np.sum((x_pulsos) ** 2)
print(energia_x6)
#Graficos

fig, axs = plt.subplots(2, 3, figsize=(12, 6))  # 2 filas, 3 columnas

# Señal senoidal
axs[0, 0].plot(t, x_senoidal,label = f"P = {potencia_x1:.2f}")
axs[0, 0].set_title("Señal senoidal 2 kHz")
axs[0, 0].set_xlabel("Tiempo [s]")
axs[0, 0].set_ylabel("Amplitud [V]")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Amplificada y desfasada
axs[0, 1].plot(t, x_desfazada,label = f"P = {potencia_x2:.2f}", color="orange")
axs[0, 1].set_title("Amplificada y desfasada")
axs[0, 1].set_xlabel("Tiempo [s]")
axs[0, 1].set_ylabel("Amplitud [V]")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Modulacion en amplitud
axs[0, 2].plot(t, x_modulada,label = f"P = {potencia_x3:.2f}", color="green")
axs[0, 2].set_title("Modulada con sen f/2")
axs[0, 2].set_xlabel("Tiempo [s]")
axs[0, 2].set_ylabel("Amplitud [V]")
axs[0, 2].legend()
axs[0, 2].grid(True)

# Señal recortada
axs[1, 0].plot(t, x_recortada,label = f"P = {potencia_x1:.2f}", color="red")
axs[1, 0].set_title("Recorte al 75%")
axs[1, 0].set_xlabel("Tiempo [s]")
axs[1, 0].set_ylabel("Amplitud [V]")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Señal cuadrada
axs[1, 1].plot(t, x_cuadrada,label = f"P = {potencia_x1:.2f}", color="purple")
axs[1, 1].set_title("Señal cuadrada")
axs[1, 1].set_xlabel("Tiempo [s]")
axs[1, 1].set_ylabel("Amplitud [V]")
axs[1, 1].legend()
axs[1, 1].grid(True)

# Pulso rectangular
axs[1, 2].scatter(t, x_pulsos,label = f"E = {energia_x6:.2f}", color="brown", marker="o")
axs[1, 2].set_title("Pulso 10 ms")
axs[1, 2].set_xlabel("Tiempo [s]")
axs[1, 2].set_ylabel("Amplitud [V]")
axs[1, 2].legend()
axs[1, 2].grid(True)

# Ajustar todo
plt.tight_layout()
plt.show()


#2) Ortogonalidad


def fun_ortogonalidad(x, y):
    numerador = np.sum(x*y) ##aca ya sabe que el vector tiene N elementos, entonces suma desde n=0 hasta N-1.
    denominador = np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2)) # aca normalizo cada señal
    return numerador/denominador


print("señal principal vs x1 (desfasada pi/2):", fun_ortogonalidad(x_senoidal, x_desfazada))
print("señal principal vs x2 (modulada):", fun_ortogonalidad(x_senoidal, x_modulada))
print("señal principal vs x3 (clipeada en amplitud):", fun_ortogonalidad(x_senoidal, x_recortada))
print("señal principal vs x4 (cuadrada de 4kHz):", fun_ortogonalidad(x_senoidal, x_cuadrada))
print("señal principal vs pulso:", fun_ortogonalidad(x_senoidal,x_pulsos))


#3) Correlacion

x0 = np.correlate(x_senoidal, x_senoidal,  mode="full")
x1 = np.correlate(x_senoidal, x_desfazada, mode="full")
x2 = np.correlate(x_senoidal, x_modulada, mode="full")
x3 = np.correlate(x_senoidal, x_recortada, mode="full")
x4 = np.correlate(x_senoidal, x_cuadrada, mode="full")
x5 = np.correlate(x_senoidal, x_pulsos, mode="full")

lags = np.arange(-N+1, N)         # retardos en muestras
lags_t = lags * Ts

fig, axs = plt.subplots(2, 3, figsize=(12, 6))  # 2 filas, 3 columnas

# Señal senoidal
axs[0, 0].plot(lags_t, x0)
axs[0, 0].set_title("Autocorrelacion")
axs[0, 0].set_xlabel("Lag")
axs[0, 0].set_ylabel("$R_{xy}$ [k]")
axs[0, 0].grid(True)

# Amplificada y desfasada
axs[0, 1].plot(lags_t, x1, color="orange")
axs[0, 1].set_title("Correlacion sen y desfazada")
axs[0, 1].set_xlabel("Lag k")
axs[0, 1].set_ylabel("$R_{xy}$ [k]")
axs[0, 1].grid(True)

# Modulacion en amplitud
axs[0, 2].plot(lags_t, x2, color="green")
axs[0, 2].set_title("Correlacion sen y modulada")
axs[0, 2].set_xlabel("Lag k")
axs[0, 2].set_ylabel("$R_{xy}$ [k]")
axs[0, 2].grid(True)

# Señal recortada
axs[1, 0].plot(lags_t, x3, color="red")
axs[1, 0].set_title("Correlacion sen y recortada")
axs[1, 0].set_xlabel("Lag k")
axs[1, 0].set_ylabel("$R_{xy}$ [k]")
axs[1, 0].grid(True)

# Señal cuadrada
axs[1, 1].plot(lags_t, x4, color="purple")
axs[1, 1].set_title("Correlacion sen y cuadrada")
axs[1, 1].set_xlabel("Lag k")
axs[1, 1].set_ylabel("$V")
axs[1, 1].grid(True)

# Pulso rectangular
axs[1, 2].plot(lags_t, x5, color="brown", marker="o")
axs[1, 2].set_title("Correlacion sen y pulso")
axs[1, 2].set_xlabel("Lag k")
axs[1, 2].set_ylabel("$R_{xy}$ [k]")
axs[1, 2].grid(True)

# Ajustar todo
plt.tight_layout()
plt.show()

# 4 identidad trigonometrica
w = 2 * np.pi * fs
identidad = 2 * np.sin(w * t / 2) - np.cos(w * t / 2) + np.cos(w * t * 3 / 2)
if np.allclose(identidad, 0, atol = 1e-12):
    print("La identidad se cumple para cualquier frecuencia w")
    print("La igualdad para cualquier f es: ", np.sum(identidad))
else:
    print("No se cumple la propiedad (el resultado es distinto de 0)")
    print(identidad)

#5 BONUS

fs, data = wavfile.read("sonido.wav")

# Vector de tiempo
tt = np.arange(len(data)) / fs

# Energía
energia_sonido = np.sum(data**2)
print("Energía del sonido:", energia_sonido)

# Graficar un fragmento (ej: primeros 1000 puntos)
plt.figure()  # sin número → nueva figura siempre
plt.plot(tt, data, label = f"E = {energia_sonido:.2f}")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Señal del audio")
plt.legend()
plt.grid(True)
plt.show()

