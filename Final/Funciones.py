#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:08:28 2026

@author: milenawaichnan
"""

import numpy as np
import matplotlib.pyplot as plt
import wfdb as wfdb
from scipy import signal
import scipy.io as sio

from pytc2.sistemas_lineales import plot_plantilla

def cargar_senal(path):
    archivo = wfdb.rdrecord(path)
    
    senal = archivo.p_signal
    fs = archivo.fs
    
    ehg = senal[:, 1]
    t = np.arange(len(ehg)) / fs
    
    return ehg, t, fs

def calcular_psd(ehg, fs, cant_promedio=25, win="flattop"):
    N = len(ehg)
    nperseg = N // cant_promedio
    nfft = 2 * nperseg
    
    f, psd = signal.welch(ehg, fs=fs, window=win, nperseg=nperseg, nfft=nfft)
    psd_db = 10 * np.log10(psd)
    
    return f, psd_db

def filtrar_senal(ehg, fs, wp=(0.3, 2.5), ws=(0.1, 3.5), alpha_p=3, alpha_s=40):
    
    sos = signal.iirdesign(
        wp=wp, ws=ws,
        gpass=alpha_p,
        gstop=alpha_s,
        ftype='butter',
        output='sos',
        fs=fs
    )
    
    ehg_filt = signal.sosfiltfilt(sos, ehg)
    
    return ehg_filt, sos

def normalizar_senal(x):
    media = np.mean(x)
    desvio = np.std(x)
    return (x - media) / desvio

def segmentar_senal(x, fs, window_length=120, overlap=0.5):
    
    L = int(window_length * fs)
    step = int(L * (1 - overlap))
    
    ventanas = []
    
    for start in range(0, len(x) - L + 1, step):
        ventanas.append(x[start:start+L])
    
    tiempo = np.arange(len(ventanas)) * (step / fs)
    
    return ventanas, tiempo

def calcular_energia_rms(ventanas):
    energias = []
    rms = []
    
    for v in ventanas:
        energias.append(np.sum(v**2))
        rms.append(np.sqrt(np.mean(v**2)))
    
    return np.array(energias), np.array(rms)

def sample_entropy(signal, m=2, r=None):
    x = np.array(signal)
    N = len(x)
    
    if r is None:
        r = 0.2 * np.std(x)
    
    def count_similar(m):
        count = 0
        for i in range(N - m):
            for j in range(i + 1, N - m):
                if np.max(np.abs(x[i:i+m] - x[j:j+m])) < r:
                    count += 1
        return count
    
    B = count_similar(m)
    A = count_similar(m + 1)
    
    if B == 0 or A == 0:
        return np.nan
    
    return -np.log(A / B)


def calcular_features(ventanas, fs, win="flattop"):
    
    frec_mediana = []
    sample_entropy_all = []
    
    for v in ventanas:
        
        nperseg = len(v) // 4
        nfft = 2 * nperseg
        
        f, psd = signal.welch(v, fs=fs, window=win, nperseg=nperseg, nfft=nfft)
        
        # frecuencia mediana
        psd_norm = psd / np.sum(psd)
        psd_acum = np.cumsum(psd_norm)
        
        mitad = psd_acum[-1] / 2
        idx = np.where(psd_acum >= mitad)[0][0]
        frec_mediana.append(f[idx])
        
        # sample entropy
        se = sample_entropy(v)
        if not np.isnan(se):
            sample_entropy_all.append(se)
    
    return np.array(frec_mediana), np.array(sample_entropy_all)

def analizar_paciente(path):
    
    # carga
    ehg, t, fs = cargar_senal(path)
    
    # PSD
    f_psd, psd_db = calcular_psd(ehg, fs)
    
    # filtrado
    ehg_filt, sos = filtrar_senal(ehg, fs)
    
    # normalización
    ehg_norm = normalizar_senal(ehg_filt)
    
    # segmentación
    ventanas, t_vent = segmentar_senal(ehg_norm, fs)
    
    # energía y rms
    energia, rms = calcular_energia_rms(ventanas)
    
    # features
    frec_mediana, sampen = calcular_features(ventanas, fs)
    
    return {
        "ehg": ehg,
        "ehg_filt": ehg_filt,
        "t": t,
        "fs": fs,
        "energia": energia,
        "rms": rms,
        "t_vent": t_vent,
        "frec_mediana": frec_mediana,
        "sampen": sampen,
        "mean_frec": np.nanmean(frec_mediana),
        "mean_sampen": np.nanmean(sampen)
    }


#----- GRAFICOS -------

def plot_senal_cruda(t, ehg):
    plt.figure(figsize=(10,4))
    plt.plot(t, ehg)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.title('EHG - señal cruda')
    plt.grid()
    
def plot_psd(f, psd_db):
    plt.figure(figsize=(10,5))
    plt.plot(f, psd_db)
    plt.title('PSD del EHG (Método de Welch)')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('PSD (dB/Hz)')
    plt.grid(True)
    
def plot_senal_filtrada(t, ehg, ehg_filt):
    plt.figure(figsize=(10,4))
    plt.plot(t, ehg, label='EHG raw')
    plt.plot(t, ehg_filt, label='Filtrada', color='orange')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.title('Señal cruda vs filtrada')
    plt.legend()
    plt.grid()

def plot_respuesta_magnitud(w, h, wp, ws, alpha_p, alpha_s, fs, f_aprox='butter'):
    plt.figure()
    plt.plot(w, 20*np.log10(np.maximum(abs(h), 1e-10)), label=f_aprox)
    
    plot_plantilla(
        filter_type='bandpass',
        fpass=wp,
        ripple=alpha_p*2,
        fstop=ws,
        attenuation=alpha_s*2,
        fs=fs
    )
    
    plt.title('Respuesta en Magnitud')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('|H(jω)| [dB]')
    plt.xlim([0, 10])
    plt.ylim([-50, 1])
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
def plot_fase(w, fase, f_aprox='butter'):
    plt.figure()
    plt.plot(w, fase, label=f_aprox)
    plt.title('Fase')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Fase [rad]')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
def plot_retardo_grupo(w, gd, f_aprox='butter'):
    plt.figure()
    plt.plot(w[:-1], gd, label=f_aprox)
    plt.title('Retardo de Grupo')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('τg [muestras]')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    
def plot_energia(t_vent, energia):
    plt.figure(figsize=(10,4))
    plt.plot(t_vent, energia)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Energía")
    plt.title("Energía del EHG por ventanas")
    plt.grid()
    
def plot_rms(t_vent, rms):
    plt.figure(figsize=(10,4))
    plt.plot(t_vent, rms)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("RMS")
    plt.title("RMS del EHG por ventanas")
    plt.grid()
    
def plot_frec_mediana(t_vent, frec_mediana):
    plt.figure(figsize=(10,4))
    plt.plot(t_vent[:len(frec_mediana)], frec_mediana)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.title("Frecuencia mediana por ventanas")
    plt.grid()
    
def plot_sample_entropy(t_vent, sampen):
    plt.figure(figsize=(10,4))
    plt.plot(t_vent[:len(sampen)], sampen)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Sample Entropy")
    plt.title("Sample Entropy por ventanas")
    plt.grid()
    

