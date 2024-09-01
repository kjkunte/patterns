import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import yfinance as yf

# Download historical data for CL=F (Crude Oil Futures) within the last 3 months with an hourly interval
symbol = 'CL=F'
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

# Apply Volume Oscillator
data = volume_oscillator(data)

# Fourier Transform on Volume Oscillator
volume_oscillator = data['Vol_Osc'].fillna(0).to_numpy(dtype=float)  # Ensure no NaN values and convert to NumPy array
fft_values = fft(volume_oscillator)
frequencies = np.fft.fftfreq(len(fft_values))

# Identify high amplitude frequencies
amplitudes = np.abs(fft_values)
threshold = np.percentile(amplitudes, 90)  # Top 10% of amplitudes
high_amp_indices = np.where(amplitudes > threshold)[0]

# Map to Time Domain
significant_time_series = np.zeros_like(volume_oscillator, dtype=float)
for idx in high_amp_indices:
    significant_time_series += np.real(ifft(fft_values * (np.arange(len(fft_values)) == idx)))

# Generate Trading Signals only when the amplitude of the volume oscillator is high
high_amplitude_threshold = np.percentile(significant_time_series, 90)  # Top 10% of amplitude in time series
data['Signal'] = np.where(
    (significant_time_series > high_amplitude_threshold) & (data['Close'] > data['Close'].rolling(20).mean()), 'Buy', 
    np.where((significant_time_series < -high_amplitude_threshold) & (data['Close'] < data['Close'].rolling(20).mean()), 'Sell', None)
)

# Save data to a file
data.to_csv('trading_signals_with_high_amplitude.csv')

# Filter out the non-signal data points before plotting
filtered_data = data[data['Signal'].notna()]

# Plotting
fig, ax1 = plt.subplots(figsize=(14, 7))

ax1.plot(data.index, data['Close'], label='Price', color='blue')
ax1.set_ylabel('Price', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(data.index, significant_time_series, label='High Amplitude Frequencies', color='orange')
ax2.set_ylabel('High Amplitude Frequencies', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Plot Trading Signals - only filtered signals
colors = np.where(filtered_data['Signal'] == 'Buy', 'green', np.where(filtered_data['Signal'] == 'Sell', 'red', 'gray'))
ax1.scatter(filtered_data.index, filtered_data['Close'], c=colors, label='Trading Signals', marker='o')

fig.tight_layout()
plt.title('Price and High Amplitude Frequency Patterns with Trading Signals')
plt.show()
