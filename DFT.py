import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

N = 20
fs = 10 
f0 = 1
deltaF = (2 * np.pi) / N

n = np.arange(N)
x = np.sin(2 * np.pi * f0 * n/N)

y = np.zeros (N, dtype= np.complex128)
for k in range  (N):
    for nn in range (N):
        y[k] += x[nn] * np.exp(-1j * 2 * np.pi * k * nn / N)
    
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.stem(np.abs(y))
plt.title("Magnitud de la DFT")
plt.xlabel("Índice k")
plt.ylabel("|X[k]|")
plt.grid()

plt.subplot(1,2,2)
plt.stem(np.angle(y))
plt.title("Fase de la DFT")
plt.xlabel("Índice k")
plt.ylabel("Fase [rad]")
plt.grid()

plt.tight_layout()
plt.show()



