#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 10:02:05 2026

@author: milenawaichnan
"""

import numpy as np
import matplotlib.pyplot as plt
import wfdb as wfdb
from scipy import signal
import scipy.io as sio
from numpy.fft import fft

from pytc2.sistemas_lineales import plot_plantilla


## --- CREACION DE LA SENAL ----- ##
archivo = wfdb.rdrecord('datos/later_induced/icehg606')

senal = archivo.p_signal      # matriz (N muestras × canales)
channels = archivo.sig_name    # nombres de los canales
fs = archivo.fs             # frecuencia de muestreo = 20 --> nyquist = 10

ehg = senal[:,1]
t = np.arange(len(ehg)) / fs

N = len(ehg)
deltaF = fs / N
freqs = np.arange(N) * deltaF


# SENAL CRUDA
plt.figure(figsize=(10,4))
plt.plot(t, ehg)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('EHG - canal 0')
plt.grid()
plt.show()

#%%
## --- ANALISIS ESPECTRAL ---
## uso welch para estimar la PSD de la señal y con esa info armar el filtro

cant_promedio = 25 #chequear
nperseg = N // cant_promedio

#print(nperseg)
nfft = 2 * nperseg
win = "flattop" #consultar con mariano que ventana es mejor, flattop queda mas limpio, hann y hamming son bastante mas ruidosas

f_ehg, PSD_EHG_W = signal.welch(ehg, fs = fs, window=win, nperseg = nperseg, nfft = nfft )
PSD_EHG_dB = 10 * np.log10(PSD_EHG_W)

plt.figure(figsize=(10,5))
plt.plot(f_ehg, PSD_EHG_dB)
plt.title('PSD del EHG (Método de Welch)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PSD en Decibeles (dB/Hz)')
plt.grid(True)

## COMENTARIO PARA CORREGIR: la info psd queda concentrada entre 0.1 y 4.5, respeto la plantilla original o la modifico? pruebo varias señales y promedio?
# %%
## --- FILTRADO --- ##
wp = (0.3, 2.5) #freq de corte/paso (Hz)
ws = (0.1, 3.5) #freq de stop/detenida (Hz)

#si alpha_p es =3 -> max atenuacion, butter

alpha_p = 3 #atenuacion de corte/paso, alfa_max, perdida en banda de paso 
alpha_s = 40 #atenuacion de stop/detenida, alfa_min, minima atenuacion requerida en banda de paso 


#Aprox de modulo
f_aprox = 'butter'
mi_sos_butter = signal.iirdesign(wp = wp, ws = ws, gpass = alpha_p, gstop = alpha_s, analog = False, ftype = f_aprox, output ='sos', fs=fs)
w, h= signal.freqz_sos(mi_sos_butter, worN = np.logspace(-2, 1.9, 1000), fs = fs) #calcula rta en frq del filtro, devuelve w y vector de salida (h es numero complejo)

ehg_filt = signal.sosfiltfilt(mi_sos_butter, ehg)

fase = np.unwrap(np.angle(h)) #unwrap hace grafico continuo

w_rad = w / (fs / 2) * np.pi
gd = -np.diff(fase) / np.diff(w_rad) #retardo de grupo [rad/rad]


#RAW VS FILTRADA
plt.figure()
plt.plot(ehg, label = 'EHG raw')
plt.plot(ehg_filt, label = 'Filtrada', color = 'orange')
plt.legend()

#Rta en magnitud
plt.figure()
plt.plot(w, 20*np.log10(np.maximum(abs(h), 1e-10)), label = f_aprox)
plot_plantilla(filter_type = 'bandpass' , fpass = wp, ripple = alpha_p*2 , fstop = ws, attenuation = alpha_s*2, fs = fs)
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('|H(jω)| [dB]')
plt.xlim([0, 10])
plt.ylim([-50, 1])
plt.grid(True, which='both', ls=':')
plt.legend()

#Fase
plt.figure()
plt.plot()
plt.plot(w, fase, label = f_aprox)
plt.title('Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True, which='both', ls=':')

# Retardo de grupo
plt.figure()
plt.plot()
plt.plot(w[:-1], gd, label = f_aprox)
plt.title('Retardo de Grupo ')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')

#Filtrado unidireccional vs bidireccional
ehg_lf = signal.sosfilt(mi_sos_butter, ehg)

plt.figure()
plt.plot(t, ehg, label='Raw', alpha=0.5)
plt.plot(t, ehg_lf, label='sosfilt')
plt.plot(t, ehg_filt, label='sosfiltfilt')
plt.legend()
plt.title('Comparación filtrado unidireccional vs bidireccional')


# --- NORMALIZACION ---

media = np.mean(ehg_filt)
desvio = np.std(ehg_filt)

ehg_norm = (ehg_filt - media) / desvio


# --- RECORTE EN VENTANAS ---
window_length = 120 # segundos
overlap = 0.5

L = int(window_length * fs)
step = int(L * (1 - overlap))

ventanas = []   #quedan 28 ventanas
for start in range(0, len(ehg_norm) - L + 1, step):
    ventanas.append(ehg_norm[start:start+L])

#chequeo de ventanas!!
# print("L:", L)
# print("Step:", step)
# print("Cantidad de ventanas:", len(ventanas))

# n_teorico = (len(ehg_filt) - L) // step + 1
# print("Ventanas teóricas:", n_teorico)

# --- EXTRACCION DE CARACTERISTICAS ---

#DOMINIO TEMPORAL --> ENERGIA

energias = []

for i in ventanas:
    E = np.sum(i ** 2)
    energias.append(E)
    
tiempo_ventanas = np.arange(len(energias)) * (step/fs)

plt.figure(figsize=(10,4))
plt.plot(tiempo_ventanas, energias)
plt.xlabel("Tiempo [s]")
plt.ylabel("Energía")
plt.title("Energía del EHG por ventanas")
plt.grid()

rms = []

for i in ventanas:
    RMS = np.sqrt(np.mean(i ** 2))
    rms.append(RMS)
    
tiempo_ventanas_1 = np.arange(len(rms)) * (step/fs)

plt.figure(figsize=(10,4))
plt.plot(tiempo_ventanas_1, rms)
plt.xlabel("Tiempo [s]")
plt.ylabel("Energía")
plt.title("RMS del EHG por ventanas")
plt.grid()

#DOMINIO ESPECTRAL --> FRECUENCIA PICO Y MEDIANA

frec_pico = []
frec_mediana = []

for i in ventanas:
    f, psd = signal.welch(i, fs = fs, window = win, nperseg = nperseg, nfft = nfft)
    
    fp = f[np.argmax(psd)]
    frec_pico.append(fp)
    
    psd_acum = np.cumsum(psd)
    mitad = psd_acum[-1] / 2
    idx = np.where(psd_acum >= mitad)[0][0]
    frec_mediana.append(f[idx])

plt.figure(figsize=(10,4))
plt.plot(tiempo_ventanas, frec_pico)
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.title("Frecuencia pico por ventanas")
plt.grid()

plt.figure(figsize=(10,4))
plt.plot(tiempo_ventanas, frec_mediana)
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.title("Frecuencia mediana por ventanas")
plt.grid()










