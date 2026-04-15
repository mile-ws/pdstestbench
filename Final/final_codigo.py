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
archivo = wfdb.rdrecord('data/tpehgdb/tpehg639')

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
retardo = -np.diff(fase) / np.diff(w_rad) #retardo de grupo [rad/rad]


#RAW VS FILTRADA
plt.figure()
plt.plot(t, ehg, label = 'EHG raw')
plt.plot(t, ehg_filt, label = 'Filtrada', color = 'orange')
plt.title('PX_639 - Señal cruda vs Filtrada (Seccion)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.xlim([15000 / fs, 16000 / fs])
plt.grid(True, which='both', ls=':')
plt.legend()

#Rta en magnitud
plt.figure()
plt.plot(w, 20*np.log10(np.maximum(abs(h), 1e-10)), label = f_aprox)
plot_plantilla(filter_type = 'bandpass' , fpass = wp, ripple = alpha_p*2 , fstop = ws, attenuation = alpha_s*2, fs = fs)
plt.title('PX_639 - Respuesta en Magnitud de filtro')
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
plt.plot(w[:-1], retardo, label = f_aprox)
plt.title('Retardo de Grupo ')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')

#Filtrado unidireccional vs bidireccional
ehg_lf = signal.sosfilt(mi_sos_butter, ehg)


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

#SAMPLE ENTROPY

def sample_entropy(signal, m=2, r=None):
    x = np.array(signal) #convierte la senal en array
    N = len(x)
    
    if r is None:
        r = 0.2 * np.std(x) #define el umbral máximo de diferencia para considerar dos segmentos similares
    
    def count_similar(m): #va comparando cuantos segmentos se parecen entre si
        count = 0
        for i in range(N - m):
            for j in range(i + 1, N - m):
                if np.max(np.abs(x[i:i+m] - x[j:j+m])) < r: #los segmentos son similares si en TODOS sus puntos la diferencia es menor que r
                    count += 1
        return count
    
    B = count_similar(m) #coincidencias en patrones cortos
    A = count_similar(m + 1) #coincidencias en patrones largos
    
    if B == 0 or A == 0:
        return np.nan
    
    return -np.log(A / B) #si A/B≈1 -> entropia baja, si A/B<<1 -> entropia alta, el logaritmo mejora la visualizacion de la info


#DOMINIO ESPECTRAL --> FRECUENCIA PICO Y MEDIANA

frec_pico = []
frec_mediana = []
sample_entropy_all = []

for i in ventanas:
    nperseg_win = len(i) // 4
    nfft_win = 2 * nperseg_win
    f, psd = signal.welch(i, fs = fs, window = win, nperseg = nperseg_win, nfft = nfft_win)
    
    
    #frecuencia mediana
    psd_norm = psd / np.sum(psd)
    psd_acum = np.cumsum(psd_norm) #suma la energia acumulada
    mitad = psd_acum[-1] / 2
    idx = np.where(psd_acum >= mitad)[0][0]
    frec_mediana.append(f[idx])
    
    
    #sample entropy
    se = sample_entropy(i)
    if not np.isnan(se):
       sample_entropy_all.append(se)

#DATOS DEL PACIENTE OBTENIDOS DEL ANALISIS
mean_d5 = np.nanmean(frec_mediana) #prom de frecuencia mediana
mean_sampen = np.nanmean(sample_entropy_all) #prom sample entropy













