import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import find_peaks
import yfinance as yf

# Download historical data
symbol = 'CL=F'  # Crude Oil Futures
data = yf.download(symbol, start="2020-01-01", end="2024-01-01", interval="1d")

# Calculate moving averages
data['50_MA'] = data['Close'].rolling(window=50).mean()
data['200_MA'] = data['Close'].rolling(window=200).mean()

# Volume Oscillator Calculation
def volume_oscillator(data, short_period=14, long_period=28):
    data['Short_Vol_MA'] = data['Volume'].rolling(window=short_period).mean()
    data['Long_Vol_MA'] = data['Volume'].rolling(window=long_period).mean()
    data['Vol_Osc'] = ((data['Short_Vol_MA'] - data['Long_Vol_MA']) / data['Long_Vol_MA']) * 100
    return data

data = volume_oscillator(data)

# Fourier Transform on Volume Oscillator
volume_oscillator = data['Vol_Osc'].fillna(0).to_numpy(dtype=float)
fft_values = fft(volume_oscillator)
frequencies = np.fft.fftfreq(len(fft_values))

# Identify high amplitude frequencies
amplitudes = np.abs(fft_values)
threshold = np.percentile(amplitudes, 90)
high_amp_indices = np.where(amplitudes > threshold)[0]

# Map to Time Domain
significant_time_series = np.zeros_like(volume_oscillator, dtype=float)
for idx in high_amp_indices:
    significant_time_series += np.real(ifft(fft_values * (np.arange(len(fft_values)) == idx)))

# Generate Trading Signals
high_amplitude_threshold = np.percentile(significant_time_series, 90)
data['Signal'] = np.where(
    (significant_time_series > high_amplitude_threshold) & (data['Close'] > data['Close'].rolling(20).mean()), 'Buy',
    np.where((significant_time_series < -high_amplitude_threshold) & (data['Close'] < data['Close'].rolling(20).mean()), 'Sell', None)
)

# Apply Wyckoff Method logic
def wyckoff_method(data):
    for i in range(len(data)):
        if data['50_MA'].iloc[i] > data['200_MA'].iloc[i]:
            if data['Close'].iloc[i] < data['50_MA'].iloc[i] and data['Close'].iloc[i] > data['200_MA'].iloc[i]:
                data.loc[data.index[i], 'Phase'] = 'Distribution'
            elif data['Close'].iloc[i] > data['50_MA'].iloc[i]:
                data.loc[data.index[i], 'Phase'] = 'Markup'
            elif data['Close'].iloc[i] < data['200_MA'].iloc[i]:
                data.loc[data.index[i], 'Phase'] = 'Markdown'
        elif data['50_MA'].iloc[i] < data['200_MA'].iloc[i]:
            if data['Close'].iloc[i] < data['50_MA'].iloc[i] and data['Close'].iloc[i] > data['200_MA'].iloc[i]:
                data.loc[data.index[i], 'Phase'] = 'Accumulation'
            elif data['Close'].iloc[i] > data['50_MA'].iloc[i]:
                data.loc[data.index[i], 'Phase'] = 'Markup'
            elif data['Close'].iloc[i] < data['200_MA'].iloc[i]:
                data.loc[data.index[i], 'Phase'] = 'Markdown'
    return data

data = wyckoff_method(data)

# Peak Detection
data['VolOsc_Peak'] = np.nan
data['Price_Movement'] = np.nan

peaks_top, _ = find_peaks(data['Vol_Osc'], distance=10)
peaks_bottom, _ = find_peaks(-data['Vol_Osc'], distance=10)

peaks_top_indices = data.index[peaks_top]
peaks_bottom_indices = data.index[peaks_bottom]

data.loc[peaks_top_indices, 'VolOsc_Peak'] = 'Top'
data.loc[peaks_bottom_indices, 'VolOsc_Peak'] = 'Bottom'

data.loc[peaks_bottom_indices, 'Price_Movement'] = data['Close'].shift(-1).loc[peaks_bottom_indices].sub(data['Close'].loc[peaks_bottom_indices]).apply(lambda x: 'Up' if x > 0 else 'Down')
data.loc[peaks_top_indices, 'Price_Movement'] = data['Close'].shift(-1).loc[peaks_top_indices].sub(data['Close'].loc[peaks_top_indices]).apply(lambda x: 'Down' if x < 0 else 'Up')

# Save data to CSV
data.to_csv('trading_signals_with_fourier_and_wyckoff.csv')

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Vol_Osc'], label='Volume Oscillator', color='gray')
plt.scatter(peaks_top_indices, data['Vol_Osc'][peaks_top_indices], color='red', label='Top Peaks')
plt.scatter(peaks_bottom_indices, data['Vol_Osc'][peaks_bottom_indices], color='blue', label='Bottom Peaks')
plt.title('Volume Oscillator with Peak Detection and Fourier Analysis')
plt.legend()
plt.show()
