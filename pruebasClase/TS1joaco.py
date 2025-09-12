#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 22:48:09 2025

@author: milenawaichnan
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy 
from scipy import signal
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags

# Una señal sinusoidal de 2KHz.
# Misma señal amplificada y desfazada en π/2.
# Misma señal modulada en amplitud por otra señal sinusoidal de la mitad de la frecuencia.
# Señal anterior recortada al 75% de su amplitud.
# Una señal cuadrada de 4kHz.
# Un pulso rectangular de 10ms.
# En cada caso indique tiempo entre muestras, número de muestras y potencia.
#print("###### Ejercicio 1 ######")

fs = 100000
N = 500
f = 2000
Ts = 1/fs
tt = np.arange(N) * Ts           # vector de tiempo

deltaF = fs/N
def mi_funcion_sen(vmax, dc, f, fase, N, fs):
    Ts = 1/fs
    tt = np.arange(N) * Ts           # vector de tiempo
    xx = dc + vmax * np.sin(2* np.pi * f * tt + fase)  # señal senoidal
    return tt, xx

#Para hacer modulacion hay que multplicar una señal contra otra señal.
def modulacion(vmax, dc, f, fase, N, fs):
    tt, xx = mi_funcion_sen(vmax=1, dc=0, f=f, fase=0, N = N, fs=fs)
    tt, x1 = mi_funcion_sen(vmax=1, dc=0, f=f/2, fase=0, N = N, fs=fs)
    x2 = xx * x1
    return x2


tt, xx = mi_funcion_sen(vmax=1, dc=0, f=f, fase=0, N = N, fs=fs)
tt, x1 = mi_funcion_sen(vmax=1, dc=0, f=f, fase=np.pi/2, N = N, fs=fs)
x2 = modulacion(vmax=1, dc=0, f=f, fase=0, N = N, fs=fs)

#Forma de hacer el clipeo para recortar la amplitud de la señal. 
x3 = np.clip(xx,-0.75,0.75,out=None)

#x4 = sp.square(2 * np.pi * 2*f * tt) #multiplico f por dos porque me pide 4kHz. Esta es con la funcion de scipy
#x4 = np.sign(np.sin(2 * np.pi * 2*f * tt)) ##esta es haciendolo con numpy y viendo el signo de la senoidal
x4 = signal.square(2*np.pi*4000*tt, duty=0.5)
x4 = x4 - np.mean(x4)
print("mean x4:", np.mean(x4))      # ~ 0.0
print("sum x4:", np.sum(x4))        # ~ 0

# aca hago el del puslo. Como Npulso = Tpulso . fs. Voy a tener 200 muestras
# como mi N lo tengo fijo en 500. voy a tener 300 muestras que estan en 0. Si yo aumento N por ejemplo, siempre voy a tener fijas 200muestras que valen 1 y las N-200=0.
#Si yo cambio fs, me cambia el Npulso entonces ahi ya se modifican las muestras que valen 1. 
T_pulso = 0.01    # 10 ms
N_pulso = int(round(T_pulso * fs))
pulso = np.zeros(N)
pulso[:N_pulso] = 1  # primeras 200 muestras valen 1
#"""
## aca empiezan los graficos.
plt.figure(figsize=(10,12))

##plt.figure(1)
plt.subplot(6,1,1)
plt.plot(tt, xx)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("2000 Hz")

plt.subplot(6,1,2)
plt.plot(tt, x1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("2000 Hz + desfasaje")

plt.subplot(6,1,3)
plt.plot(tt, x2)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("modulacion")

plt.subplot(6,1,4)
plt.plot(tt, x3)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("recortada en el 75% de la amplitud ")

#plt.tight_layout()  # ajusta los títulos y ejes
#plt.show()


#plt.figure(2)
plt.subplot(6,1,5)
plt.plot(tt, x4)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Funcion cuadrada")

plt.subplot(6,1,6)
plt.scatter(tt, pulso)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Pulso rectangular de 10 ms")

plt.tight_layout()  # ajusta los títulos y ejes
plt.show()
#"""

## np.mean(xx**2  saca el promedio de esos valores, para xx,x1,x2,x3,x4 voy a usar la potencia promeido. Porque tienen duración infinita (si la extendiera en el tiempo), y su energía sería infinita.
# np.mean hace 1/N * sumatoria desde n=1 hasta N de x**2
potencia_xx = np.mean(xx**2)
potencia_x1 = np.mean(x1**2)
potencia_x2 = np.mean(x2**2)
potencia_x3 = np.mean(x3**2)
potencia_x4 = np.mean(x4**2)
energia_pulso = np.sum(pulso**2) * Ts #aca uso energia porque, Señales no periódicas o de duración finita. energía en unidades tiempo·amplitud^2
#aca imprimo esto que piden: En cada caso indique tiempo entre muestras, número de muestras y potencia o energía según corresponda.
"""
print("Señal principal, Ts: ",Ts, " N: ", N, "y potencia promedio:", potencia_xx)
print("Señal desfada, Ts: ",Ts, " N: ", N, "y potencia promedio:", potencia_x1)
print("Señal modulada, Ts: ",Ts, " N: ", N, "y potencia promedio:", potencia_x2)
print("Señal recortada, Ts: ",Ts, " N: ", N, "y potencia promedio:", potencia_x3)
print("Señal cuadrada, Ts: ",Ts, " N: ", N, "y potencia promedio:", potencia_x4)
print("Señal pulso, Ts: ",Ts, " N: ", N, "y energia:", energia_pulso)
print("\n")
"""


##2) ortogonalidad

def fun_ortogonalidad(x, y):
    numerador = np.sum(x*y) ##aca ya sabe que el vector tiene N elementos, entonces suma desde n=0 hasta N-1.
    denominador = np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2)) # aca normalizo cada señal
    return numerador/denominador
#"""
print("###### Ejercicio 2 ######")

print("señal principal vs x1 (desfasada pi/2):", fun_ortogonalidad(xx, x1))
print("señal principal vs x2 (modulada):", fun_ortogonalidad(xx, x2))
print("señal principal vs x3 (clipeada en amplitud):", fun_ortogonalidad(xx, x3))
print("señal principal vs x4 (cuadrada de 4kHz):", fun_ortogonalidad(xx, x4))
print("señal principal vs pulso:", fun_ortogonalidad(xx, pulso))
print("\n")
#"""
#3)  3) Graficar la autocorrelación de la primera señal y la correlación entre ésta y las demás.
    
Rxx = correlate(xx, xx, mode="full")
Rx1 = correlate(xx, x1, mode="full")
Rx2 = correlate(xx, x2, mode="full")
Rx3 = correlate(xx, x3, mode="full")
#Rx4 = np.correlate(xx, x4, mode="full")
#Rxpulso = np.correlate(xx, pulso, mode="full")
Rx4 = correlate(xx, x4, mode="full")
Rxpulso = correlate(xx, pulso, mode="full")


lags = correlation_lags(len(xx), len(xx), mode="full")
lags_time = lags * Ts
#"""
##plt.figure(2)
plt.figure(figsize=(18,22))

plt.subplot(6,1,1)
plt.plot(lags_time, Rxx)
plt.title("autocorrelacion")
plt.xlabel("Retardo [s]")
plt.ylabel("Rxx")

plt.subplot(6,1,2)
plt.plot(lags_time, Rx1)
plt.title("x vs x1")
plt.xlabel("Retardo [s]")
plt.ylabel("Rxx")


plt.subplot(6,1,3)
plt.plot(lags_time, Rx2)
plt.title("x vs x2")
plt.xlabel("Retardo [s]")
plt.ylabel("Rxx")

plt.subplot(6,1,4)
plt.plot(lags_time, Rx3)
plt.title("x vs x3")
plt.xlabel("Retardo [s]")
plt.ylabel("Rxx")

#plt.tight_layout()  # ajusta los títulos y ejes
#plt.show()

#plt.figure(4)

plt.subplot(6,1,5)
plt.plot(lags_time, Rx4)
plt.title("x vs x4")
plt.xlabel("Retardo [s]")
plt.ylabel("Rxx")

plt.subplot(6,1,6)
plt.plot(lags_time, Rxpulso)
plt.title("x vs pulso")
plt.xlabel("Retardo [s]")
plt.ylabel("Rxx")

plt.tight_layout()  # ajusta los títulos y ejes
plt.show()

#3 Igualdad
print("###### Ejercicio 3 ######")

w = 2 * np.pi * f
w1 = 2 * w
w2 = 0
igualdad = 2*np.sin(w*tt)*np.sin(w*tt/2)-np.cos(w*tt/2)+np.cos(w*tt*3/2)
igualdad1 = 2*np.sin(w1*tt)*np.sin(w1*tt/2)-np.cos(w1*tt/2)+np.cos(w1*tt*3/2)
igualdad2 = 2*np.sin(w2*tt)*np.sin(w2*tt/2)-np.cos(w2*tt/2)+np.cos(w2*tt*3/2)


print("Aca demuestro la identidad trigonometrica")

if np.allclose([igualdad1, igualdad2, igualdad], 0, atol=1e-12):
    print("Son (casi) todos cero")
    print("La propiedad se cumple para cualquier frecuencia")
else:
    print("No se cumple la propiedad")
    
