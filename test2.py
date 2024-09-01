import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft

import yfinance as yf

# Download historical data for NG=F (Natural Gas Futures) within the last 3 months with an hourly interval
symbol = 'NG=F'
data = yf.download(symbol, period="3mo", interval="1h")

# Calculate moving averages
data['50_MA'] = data['Close'].rolling(window=50).mean()
data['200_MA'] = data['Close'].rolling(window=200).mean()

# Volume Oscillator Calculation
def volume_oscillator(data, short_period=14, long_period=28):
    data['Short_Vol_MA'] = data['Volume'].rolling(window=short_period).mean()
    data['Long_Vol_MA'] = data['Volume'].rolling(window=long_period).mean()
    data['Vol_Osc'] = ((data['Short_Vol_MA'] - data['Long_Vol_MA']) / data['Long_Vol_MA']) * 100
    return data

# Assuming you already have the data loaded and processed as before:
# Replace 'volume_oscillator' with your actual volume oscillator data
volume_oscillator = data['Vol_Osc'].fillna(0).to_numpy(dtype=float)

# Perform Fourier Transform
fft_values = fft(volume_oscillator)
frequencies = np.fft.fftfreq(len(fft_values))

# Calculate amplitudes
amplitudes = np.abs(fft_values)

# Identify high amplitude frequencies
threshold = np.percentile(amplitudes, 90)  # Top 10% of amplitudes
high_amp_indices = np.where(amplitudes > threshold)[0]
high_amplitude_frequencies = frequencies[high_amp_indices]
high_amplitude_values = amplitudes[high_amp_indices]

# Plot the high amplitude frequencies
plt.figure(figsize=(10, 6))
plt.stem(high_amplitude_frequencies, high_amplitude_values, use_line_collection=True, basefmt=" ")
plt.title('High Amplitude Frequencies in Volume Oscillator')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()
