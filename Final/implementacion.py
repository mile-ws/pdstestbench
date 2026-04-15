#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:14:33 2026

@author: milenawaichnan
"""

import Funciones as fun
import numpy as np
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla

# ---- CARGA Y ANÁLISIS DE PACIENTES ----

px_877 = fun.analizar_paciente('data/tpehgdb/tpehg877')
px_914 = fun.analizar_paciente('data/tpehgdb/tpehg914')
px_718 = fun.analizar_paciente('data/tpehgdb/tpehg718')
px_639 = fun.analizar_paciente('data/tpehgdb/tpehg639')

pacientes = [
    {'data': px_877,  'nombre': 'PX_877',  'grupo': 'pretermino'},
    {'data': px_914, 'nombre': 'PX_914', 'grupo': 'pretermino'},
    {'data': px_718,  'nombre': 'PX_718',  'grupo': 'termino'},
    {'data': px_639,  'nombre': 'PX_639',  'grupo': 'termino'},
]


# ---- PREPROCESAMIENTO (una sola vez, paciente representativo) ----

wp = (0.3, 2.5)
ws = (0.1, 3.5)
alpha_p = 3
alpha_s = 40

fun.plot_psd(f=px_639['f_psd'], psd_db=px_639['psd_db'])
fun.plot_filtro(t=px_639['t'], ehg=px_639['ehg'], ehg_filt=px_639['ehg_filt'],
                w=px_639['w'], h=px_639['h'],
                wp=wp, ws=ws, alpha_p=alpha_p, alpha_s=alpha_s,
                fs=px_639['fs'], f_aprox='butter')

# ---- SEÑAL RAW VS FILTRADA POR PACIENTE ----

for px in pacientes:
    fun.plot_senal_filtrada(t=px['data']['t'],
                            ehg=px['data']['ehg'],
                            ehg_filt=px['data']['ehg_filt'])
    plt.suptitle(px['nombre'])

# ---- COMPARATIVO RMS ----

colores  = {'pretermino': 'steelblue', 'termino': 'tomato'}
conteo   = {'pretermino': 0, 'termino': 0}
rms_pre, rms_ter = [], []

plt.figure(figsize=(12, 4))
for px in pacientes:
    grupo = px['grupo']
    conteo[grupo] += 1
    ls = '-' if conteo[grupo] == 1 else '--'
    plt.plot(px['data']['t_vent'], px['data']['rms'],
             color=colores[grupo], linestyle=ls,
             label=f"{px['nombre']} ({grupo})")
    if grupo == 'pretermino':
        rms_pre.append(px['data']['rms'])
    else:
        rms_ter.append(px['data']['rms'])

prom_pre = np.mean(np.concatenate(rms_pre))
prom_ter = np.mean(np.concatenate(rms_ter))
plt.axhline(prom_pre, color=colores['pretermino'],  linestyle=':', linewidth=1.5,
            label=f'Promedio pretermino ({prom_pre:.4f} mV)')
plt.axhline(prom_ter, color=colores['termino'], linestyle=':', linewidth=1.5,
            label=f'Promedio termino({prom_ter:.4f} mV)')
plt.xlabel('Tiempo [s]')
plt.ylabel('RMS [mV]')
plt.title('RMS por ventana — Pretermino vs Termino')
plt.legend()
plt.grid(True)
plt.tight_layout()

# ---- COMPARATIVO FRECUENCIA MEDIANA ----

conteo = {'pretermino': 0, 'termino': 0}
frec_pre, frec_ter = [], []

plt.figure(figsize=(12, 4))
for px in pacientes:
    grupo = px['grupo']
    conteo[grupo] += 1
    ls = '-' if conteo[grupo] == 1 else '--'
    fm = px['data']['frec_mediana']
    plt.plot(px['data']['t_vent'][:len(fm)], fm,
             color=colores[grupo], linestyle=ls,
             label=f"{px['nombre']} ({grupo})")
    if grupo == 'pretermino':
        frec_pre.append(fm)
    else:
        frec_ter.append(fm)

prom_pre = np.nanmean(np.concatenate(frec_pre))
prom_ter = np.nanmean(np.concatenate(frec_ter))
plt.axhline(prom_pre, color=colores['pretermino'],  linestyle=':', linewidth=1.5,
            label=f'Promedio pretermino ({prom_pre:.3f} Hz)')
plt.axhline(prom_ter, color=colores['termino'], linestyle=':', linewidth=1.5,
            label=f'Promedio termino ({prom_ter:.3f} Hz)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia mediana [Hz]')
plt.title('Frecuencia mediana por ventana — Pretermino vs Termino')
plt.legend()
plt.grid(True)
plt.tight_layout()

# ---- COMPARATIVO SAMPLE ENTROPY ----

conteo = {'pretermino': 0, 'termino': 0}
se_pre, se_ter = [], []

plt.figure(figsize=(12, 4))
for px in pacientes:
    grupo = px['grupo']
    conteo[grupo] += 1
    ls = '-' if conteo[grupo] == 1 else '--'
    se = px['data']['sampen']
    plt.plot(px['data']['t_vent'][:len(se)], se,
             color=colores[grupo], linestyle=ls,
             label=f"{px['nombre']} ({grupo})")
    if grupo == 'pretermino':
        se_pre.append(se)
    else:
        se_ter.append(se)

prom_pre = np.nanmean(np.concatenate(se_pre))
prom_ter = np.nanmean(np.concatenate(se_ter))
plt.axhline(prom_pre, color=colores['pretermino'],  linestyle=':', linewidth=1.5,
            label=f'Promedio pretermino ({prom_pre:.3f})')
plt.axhline(prom_ter, color=colores['termino'], linestyle=':', linewidth=1.5,
            label=f'Promedio termino ({prom_ter:.3f})')
plt.xlabel('Tiempo [s]')
plt.ylabel('Sample Entropy')
plt.title('Sample Entropy por ventana — Pretermino vs Termino')
plt.legend()
plt.grid(True)
plt.tight_layout()

# ---- RESUMEN COMPARATIVO (barras) ----

grupos   = ['induced', 'cesarean']
data_res = {g: {'rms': [], 'frec_mediana': [], 'sampen': []} for g in grupos}

for px in pacientes:
    g = px['grupo']
    data_res[g]['rms'].append(np.mean(px['data']['rms']))
    data_res[g]['frec_mediana'].append(np.nanmean(px['data']['frec_mediana']))
    data_res[g]['sampen'].append(np.nanmean(px['data']['sampen']))

features = ['rms',     'frec_mediana',    'sampen']
labels   = ['RMS [mV]','Frec. Mediana [Hz]','Sample Entropy']

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, feat, label in zip(axes, features, labels):
    valores = [np.mean(data_res[g][feat]) for g in grupos]
    barras  = ax.bar(grupos, valores,
                     color=[colores[g] for g in grupos],
                     width=0.4, edgecolor='black', linewidth=0.8)
    for barra, val in zip(barras, valores):
        ax.text(barra.get_x() + barra.get_width()/2, barra.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_title(label)
    ax.set_ylabel(label)
    ax.set_xlabel('Grupo')
    ax.grid(True, axis='y', linestyle=':')

fig.suptitle('Comparación de features — Induced vs Cesarean', fontsize=13)
plt.tight_layout()

plt.show()

def plot_senal_filtrada(t, ehg, ehg_filt, nombre=''):
    plt.figure(figsize=(10,4))
    plt.plot(t, ehg, label='EHG raw')
    plt.plot(t, ehg_filt, label='Filtrada', color='orange')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [mV]')
    plt.title(f'{nombre} — Señal cruda vs filtrada')
    plt.legend()
    plt.grid()
    plt.tight_layout()

def plot_respuesta_filtro(w, h, wp, ws, alpha_p, alpha_s, fs, f_aprox='butter'):
    
    plt.figure(figsize=(6,4))
    
    plt.plot(w, 20*np.log10(np.maximum(abs(h), 1e-10)), label=f_aprox)
    
    plot_plantilla(filter_type='bandpass', fpass=wp, ripple=alpha_p*2, fstop=ws, attenuation=alpha_s*2,fs=fs)
    
    plt.title('Respuesta en magnitud del filtro')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud [dB]')
    
    plt.xlim([0, 10])
    plt.ylim([-50, 1])
    
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    

mi_sos_butter, ehg_filt, w, h, fase, retardo = fun.filtrar_senal(px_639['ehg'], px_639['fs'])
plot_respuesta_filtro(w = w, h = h, wp = wp, ws = ws, alpha_p = alpha_p, alpha_s = alpha_s, fs = px_639['ehg'])
plot_senal_filtrada(t = px_639['t'], ehg = px_639['ehg'], ehg_filt = px_639['ehg_filt'], nombre='PX_639')
   
   
   
   
   
   