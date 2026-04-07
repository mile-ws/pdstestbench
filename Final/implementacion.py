#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:14:33 2026

@author: milenawaichnan
"""

import Funciones as fun

#res["ehg"]            # señal original
#res["ehg_filt"]       # señal filtrada
#res["t"]              # tiempo
#res["fs"]             # frecuencia de muestreo

#res["energia"]        # energía por ventanas
#res["rms"]            # RMS por ventanas
#res["t_vent"]         # tiempo de ventanas

#res["frec_mediana"]   # frecuencia mediana por ventana
#res["sampen"]         # sample entropy por ventana

#res["mean_frec"]      # promedio frecuencia
#res["mean_sampen"]    # promedio entropy


px_606 = fun.analizar_paciente('datos/later_induced/icehg606')

fun.plot_senal_filtrada(px_606["t"], px_606["ehg"], px_606["ehg_filt"])