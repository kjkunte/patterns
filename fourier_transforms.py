import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft

# Download historical data for CL=F (Crude Oil Futures) within the last 3 months with an hourly interval
symbol = 'CL=F'
data = yf.download(symbol, period="3mo", interval="1h")

# Calculate moving averages
data['50_MA'] = data['Close'].rolling(window=50).mean()
data['200_MA'] = data['Close'].rolling(window=200).mean()

# Initialize the Phase column as an object (string) type
data['Phase'] = np.nan
data['Phase'] = data['Phase'].astype('object')

# Wyckoff Method using Simplified Logic
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

# Apply the Wyckoff Method logic
data = wyckoff_method(data)

# Fill NaN values in the 'Phase' column with a placeholder, avoiding chained assignment
data['Phase'] = data['Phase'].fillna('Unknown')

# Volume Oscillator Calculation (Simplified for Demonstration)
def volume_oscillator(data, short_period=14, long_period=28):
    data['Short_Vol_MA'] = data['Volume'].rolling(window=short_period).mean()
    data['Long_Vol_MA'] = data['Volume'].rolling(window=long_period).mean()
    data['Vol_Osc'] = ((data['Short_Vol_MA'] - data['Long_Vol_MA']) / data['Long_Vol_MA']) * 100
    return data

# Apply Volume Oscillator
data = volume_oscillator(data)

# Save the data to a CSV file
data.to_csv('wyckoff_analysis.csv')

# Convert the Volume Oscillator to a NumPy array for FFT, ensuring proper alignment
vol_osc_np = data['Vol_Osc'].dropna().to_numpy()

# Apply FFT to the Volume Oscillator data
vol_osc_fft = fft(vol_osc_np)

# Frequency domain representation
frequencies = np.fft.fftfreq(len(vol_osc_fft))

# Amplitudes of frequencies
amplitudes = np.abs(vol_osc_fft)

# Plot the frequency spectrum
plt.figure(figsize=(10, 6))
plt.plot(frequencies, amplitudes)
plt.title('Volume Oscillator - Frequency Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

# Filter: Identify significant frequencies (example: top 3)
significant_freq_indices = np.argsort(amplitudes)[-3:]

# Create a filtered signal focusing on the significant frequencies
filtered_vol_osc_fft = np.zeros_like(vol_osc_fft)
filtered_vol_osc_fft[significant_freq_indices] = vol_osc_fft[significant_freq_indices]

# Inverse FFT to get the filtered time-domain signal
filtered_vol_osc = ifft(filtered_vol_osc_fft)

# Add the filtered oscillator to the dataframe
data['Filtered_Vol_Osc'] = np.nan
data.loc[data['Vol_Osc'].dropna().index, 'Filtered_Vol_Osc'] = np.real(filtered_vol_osc)

# Plot the original and filtered volume oscillator for comparison
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Vol_Osc'], label='Original Volume Oscillator')
plt.plot(data.index, data['Filtered_Vol_Osc'], label='Filtered Volume Oscillator', linestyle='--')
plt.title('Original vs Filtered Volume Oscillator')
plt.legend()
plt.show()
