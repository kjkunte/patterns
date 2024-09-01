import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Download historical data for ICICI Bank within the last 730 days
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

# Fill NaN values in the 'Phase' column with a placeholder
data['Phase'].fillna('Unknown', inplace=True)

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

# Plot the data in separate windows
plt.figure(figsize=(14, 6))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.title(f'{symbol} Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(data['50_MA'], label='50-Day Moving Average', color='orange')
plt.plot(data['200_MA'], label='200-Day Moving Average', color='green')
plt.title(f'{symbol} Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize=(14, 6))
for phase, color in zip(['Accumulation', 'Markup', 'Distribution', 'Markdown'], ['darkblue', 'darkgreen', 'darkred', 'purple']):
    plt.fill_between(data.index, data['Close'], where=data['Phase'] == phase, color=color, alpha=0.2, label=phase)
plt.plot(data['Close'], label='Close Price', color='blue')
plt.title(f'{symbol} Wyckoff Phases')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(data['Vol_Osc'], label='Volume Oscillator', color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.title(f'{symbol} Volume Oscillator')
plt.xlabel('Date')
plt.ylabel('Volume Oscillator (%)')
plt.legend()
plt.show()

# Calculate RSI
data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()

# Calculate MACD
# data['MACD'] = ta.trend.MACD(data['Close']).macd()
# data['Signal_Line'] = ta.trend.MACD(data['Close']).macd_signal()

# Example Trading Signal Logic
data['Signal'] = np.nan
for i in range(1, len(data)):
    if data['RSI'].iloc[i] < 30 and data['Vol_Osc'].iloc[i] > 0:
        data.loc[data.index[i], 'Signal'] = 'Buy'
    elif data['RSI'].iloc[i] > 70 and data['Vol_Osc'].iloc[i] < 0:
        data.loc[data.index[i], 'Signal'] = 'Sell'

# Plot RSI
plt.figure(figsize=(14, 6))
plt.plot(data['RSI'], label='RSI', color='orange')
plt.axhline(30, color='red', linestyle='--')
plt.axhline(70, color='red', linestyle='--')
plt.title(f'{symbol} RSI')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.show()


# Save the data to a CSV file with signals
# data.to_csv('wyckoff_with_signals.csv')


# Calculate average volume during each phase
average_volumes = data.groupby('Phase')['Volume'].mean()

# Print the average volumes
print("Average Volumes During Each Phase:")
print(average_volumes)

# You can also visualize this
plt.figure(figsize=(10, 6))
average_volumes.plot(kind='bar', color=['darkblue', 'darkgreen', 'darkred', 'purple'])
plt.title(f'Average Volume During Wyckoff Phases - {symbol}')
plt.xlabel('Wyckoff Phases')
plt.ylabel('Average Volume')
plt.show()

# Calculate RSI
data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()

# Calculate MACD
# data['MACD'] = ta.trend.MACD(data['Close']).macd()
# data['Signal_Line'] = ta.trend.MACD(data['Close']).macd_signal()

# Example Trading Signal Logic
data['Signal'] = np.nan
for i in range(1, len(data)):
    if data['RSI'].iloc[i] < 30 and data['Vol_Osc'].iloc[i] > 0:
        data.loc[data.index[i], 'Signal'] = 'Buy'
    elif data['RSI'].iloc[i] > 70 and data['Vol_Osc'].iloc[i] < 0:
        data.loc[data.index[i], 'Signal'] = 'Sell'

# Plot RSI
plt.figure(figsize=(14, 6))
plt.plot(data['RSI'], label='RSI', color='orange')
plt.axhline(30, color='red', linestyle='--')
plt.axhline(70, color='red', linestyle='--')
plt.title(f'{symbol} RSI')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.show()


# Save the data to a CSV file with signals
data.to_csv('wyckoff_with_signals.csv')

