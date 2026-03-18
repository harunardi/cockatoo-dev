import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
Fs = 50        # Sampling frequency [Hz]
T = 1 / Fs     # Sampling period [s]
N = 128         # Number of samples
f0 = 5         # Signal frequency [Hz]

# --- Time vector ---
t = np.arange(N) * T  # 0, T, 2T, ..., (N-1)*T

# --- Signal ---
x = np.sin(2 * np.pi * f0 * t)

# --- FFT computation ---
X = np.fft.fft(x)
freq = np.fft.fftfreq(N, d=T)  # Frequency bins

# --- Take only the positive half (single-sided) ---
half = N // 2
X_mag = 2 * np.abs(X[:half]) / N  # Normalize amplitude to 1
freq_half = freq[:half]

# --- Find the FFT peak ---
peak_index = np.argmax(X_mag)
peak_freq = freq_half[peak_index]
peak_amp = X_mag[peak_index]

print(f"Peak frequency: {peak_freq:.3f} Hz")
print(f"Peak amplitude: {peak_amp:.3f}")

# --- Plot ---
plt.figure(figsize=(7,4))
plt.plot(freq_half, X_mag, 'o-')
plt.title('FFT of sin(2π·2t)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.savefig('fft_example.png')
