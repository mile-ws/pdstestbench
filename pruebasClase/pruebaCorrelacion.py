#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 21:02:12 2025

@author: milenawaichnan
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

M = 8

x = np.zeros(M)
y = np.zeros(M)

x[:2] = 1
y[5] = 1

rxx = sig.correlate(x, y)
convxy = sig.convolve(x, y)

plt.figure(1)
plt.clf()
plt.plot(x, 'x:', label = 'x')
plt.plot(y, 'x:', label = 'y')
plt.plot(rxx, 'o:', label = 'rxy')
plt.plot(convxy, 'o:', label = 'conv')
plt.legend()
plt.show()
