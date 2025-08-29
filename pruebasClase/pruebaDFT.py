import numpy as np
import matplotlib.pyplot as plt

# Parámetros
N = 1000  # cantidad de muestras
n = np.arange(N)
fx = 3   # frecuencia discreta
fs = 1000
x = np.sin(2 * np.pi * fx * n/N)
deltaF = fs / N

# ---------- DFT MANUAL ----------
X_manual = []
for k in range(N):
    suma = 0
    for n_ in range(N):
        suma += x[n_] * np.exp(-1j*2*np.pi*k*n_/N)
    X_manual.append(suma)
X_manual = np.array(X_manual)

# ---------- FFT (NumPy) ----------
X_fft = np.fft.fft(x)

# ---------- Comparación ----------
plt.figure(1)

plt.stem(np.abs(X_manual), basefmt=" ")
plt.title("DFT manual |X[k]|")
plt.xlabel("k")


plt.figure(2)

plt.stem(np.abs(X_fft), basefmt=" ")
plt.title("FFT NumPy |X[k]|")
plt.xlabel("k")

plt.tight_layout()
plt.show()

# Verifico numéricamente
print("¿Son iguales?:", np.allclose(X_manual, X_fft))
