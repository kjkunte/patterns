import yfinance as yf
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Download data using daily interval
data = yf.download("CL=F", start="2020-01-01", end="2024-01-01", interval="1d")

# Calculate the Volume Oscillator
short_window = 10
long_window = 50
data['VolOsc'] = data['Volume'].rolling(window=short_window).mean() - data['Volume'].rolling(window=long_window).mean()
data['VolOscNorm'] = data['VolOsc'] / data['Volume'].rolling(window=long_window).mean()
data.dropna(inplace=True)

# Initialize columns with appropriate data types
data['VolOsc_Peak'] = np.nan
data['Price_Movement'] = np.nan
data['VolOsc_Peak'] = data['VolOsc_Peak'].astype(object)
data['Price_Movement'] = data['Price_Movement'].astype(object)

# Find peaks (tops) and troughs (bottoms)
peaks_top, _ = find_peaks(data['VolOscNorm'], distance=10)
peaks_bottom, _ = find_peaks(-data['VolOscNorm'], distance=10)

# Convert the indices to match the DataFrame's datetime index
peaks_top_indices = data.index[peaks_top]
peaks_bottom_indices = data.index[peaks_bottom]

# Mark the peaks in the DataFrame
data.loc[peaks_top_indices, 'VolOsc_Peak'] = 'Top'
data.loc[peaks_bottom_indices, 'VolOsc_Peak'] = 'Bottom'

# Determine price movement after each peak
data.loc[peaks_bottom_indices, 'Price_Movement'] = data['Adj Close'].shift(-1).loc[peaks_bottom_indices].sub(data['Adj Close'].loc[peaks_bottom_indices]).apply(lambda x: 'Up' if x > 0 else 'Down')
data.loc[peaks_top_indices, 'Price_Movement'] = data['Adj Close'].shift(-1).loc[peaks_top_indices].sub(data['Adj Close'].loc[peaks_top_indices]).apply(lambda x: 'Down' if x < 0 else 'Up')

# Save data to a CSV file for further analysis
data.to_csv('volume_oscillation_analysis.csv')

# Display the results for inspection
print(data[['VolOscNorm', 'VolOsc_Peak', 'Price_Movement']])

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['VolOscNorm'], label='Volume Oscillator')
plt.scatter(peaks_top_indices, data['VolOscNorm'][peaks_top_indices], color='red', label='Top Peaks')
plt.scatter(peaks_bottom_indices, data['VolOscNorm'][peaks_bottom_indices], color='blue', label='Bottom Peaks')
plt.title('Volume Oscillator with Peak Detection and Price Movement')
plt.legend()
plt.show()
