import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import pymc3 as pm
from scipy.fftpack import fft
import matplotlib.pyplot as plt




# Download historical data for Crude Oil futures (symbol: CL=F)
data = yf.download("CL=F", start="2010-01-01", end="2024-01-01", interval="1h")
data['Returns'] = data['Adj Close'].pct_change().dropna()

# Volume Oscillator Calculation
short_window = 10
long_window = 50
data['VolOsc'] = data['Volume'].rolling(window=short_window).mean() - data['Volume'].rolling(window=long_window).mean()
data['VolOscNorm'] = data['VolOsc'] / data['Volume'].rolling(window=long_window).mean()
data.dropna(inplace=True)


# Fit a GARCH(1,1) model to the returns
model = arch_model(data['Returns'], vol='Garch', p=1, q=1)
garch_result = model.fit(disp="off")
data['GARCH_Volatility'] = np.sqrt(garch_result.conditional_volatility)



with pm.Model() as model:
    # Priors for the parameters
    sigma = pm.HalfNormal('sigma', sd=1)
    phi = pm.Beta('phi', alpha=2, beta=2)
    
    # Latent volatility process
    volatility = pm.GaussianRandomWalk('volatility', sd=sigma, shape=len(data['Returns']))
    
    # Observation equation
    epsilon = pm.Normal('epsilon', mu=0, sd=pm.math.exp(volatility / 2), observed=data['Returns'])
    
    # Inference
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# Extract the latent volatility estimates
volatility_estimates = trace.posterior['volatility'].mean(dim=['chain', 'draw'])
data['Bayesian_Volatility'] = volatility_estimates.values


# Apply FFT to the normalized Volume Oscillator
vol_osc_fft = fft(data['VolOscNorm'])
frequencies = np.fft.fftfreq(len(vol_osc_fft))

# Plot the amplitude spectrum
plt.figure(figsize=(10, 5))
plt.plot(frequencies, np.abs(vol_osc_fft))
plt.title('Volume Oscillator FFT Amplitude Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

# Detect sinusoidal pattern
sinusoidal_component = np.argmax(np.abs(vol_osc_fft))
sinusoidal_freq = frequencies[sinusoidal_component]
print(f"Detected Sinusoidal Frequency: {sinusoidal_freq}")


# Combine GARCH and Bayesian Volatility to predict peaks
data['Volatility'] = data[['GARCH_Volatility', 'Bayesian_Volatility']].mean(axis=1)

# Detect peaks in volatility using the volume oscillator's sinusoidal pattern
from scipy.signal import find_peaks

peaks, _ = find_peaks(data['Volatility'], distance=10)
data['Peaks'] = np.nan
data['Peaks'][peaks] = data['Volatility'][peaks]

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Volatility'], label='Combined Volatility')
plt.plot(data.index, data['Peaks'], "x", label='Peaks')
plt.plot(data.index, data['VolOscNorm'], label='Volume Oscillator')
plt.title('Volatility and Volume Oscillator with Peak Detection')
plt.legend()
plt.show()
