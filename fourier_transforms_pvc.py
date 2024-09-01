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



# Calculate volume oscillator
data = volume_oscillator(data)

# Apply FFT to volume oscillator and price data
vol_osc_fft = fft(data['Vol_Osc'].dropna().to_numpy())
price_fft = fft(data['Close'].dropna().to_numpy())

# Frequency domain representation
frequencies = np.fft.fftfreq(len(vol_osc_fft))

# Calculate cross-correlation
cross_correlation = np.correlate(vol_osc_fft, price_fft, mode='full')

# Plot cross-correlation
plt.figure(figsize=(10, 6))
plt.plot(frequencies, cross_correlation)
plt.title('Cross-Correlation between Volume Oscillator and Price')
plt.xlabel('Frequency')
plt.ylabel('Cross-Correlation')
plt.show()