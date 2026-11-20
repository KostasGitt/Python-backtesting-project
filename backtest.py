print("My Python Backtesting Project")
print("--------------------------------")

initial_capital = 10000

print("Starting capital: £", initial_capital)
print("My backtesting project is officially started!")

# ==========================================
# US STOCK MARKET BACKTESTING PROJECT
# Moving Average Crossover Strategy
# ==========================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# 1. PROJECT SETTINGS

TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"

INITIAL_CAPITAL = 10000
SHORT_WINDOW = 20
LONG_WINDOW = 50

print("Starting backtest...")
print(f"Stock: {TICKER}")


# 2. DOWNLOAD HISTORICAL STOCK DATA

data = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False
)

if data.empty:
    raise ValueError("No stock data downloaded. Check your internet connection.")

# Handle possible multi-level column names
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data[["Close"]].copy()
data.dropna(inplace=True)

print(f"Downloaded {len(data)} trading days.")


# 3. CALCULATE MOVING AVERAGES

data["SMA_20"] = data["Close"].rolling(
    window=SHORT_WINDOW
).mean()

data["SMA_50"] = data["Close"].rolling(
    window=LONG_WINDOW
).mean()


# 4. GENERATE TRADING SIGNALS

data["Signal"] = 0

data.loc[
    data["SMA_20"] > data["SMA_50"],
    "Signal"
] = 1

# Shift signals by one day to avoid look-ahead bias
data["Position"] = data["Signal"].shift(1).fillna(0)


# 5. CALCULATE DAILY RETURNS

data["Market_Return"] = data["Close"].pct_change().fillna(0)

data["Strategy_Return"] = (
    data["Position"] * data["Market_Return"]
)


# 6. CALCULATE PORTFOLIO VALUE

data["Buy_Hold"] = (
    INITIAL_CAPITAL * (1 + data["Market_Return"]).cumprod()
)

data["Portfolio_Value"] = (
    INITIAL_CAPITAL * (1 + data["Strategy_Return"]).cumprod()
)


# 7. CALCULATE PERFORMANCE METRICS

total_return = (
    data["Portfolio_Value"].iloc[-1] / INITIAL_CAPITAL - 1
) * 100

buy_hold_return = (
    data["Buy_Hold"].iloc[-1] / INITIAL_CAPITAL - 1
) * 100

daily_returns = data["Strategy_Return"]

annual_volatility = daily_returns.std() * np.sqrt(252)

sharpe_ratio = (
    (daily_returns.mean() * 252) /
    (daily_returns.std() * np.sqrt(252))
    if daily_returns.std() != 0 else 0
)

rolling_peak = data["Portfolio_Value"].cummax()

drawdown = (
    data["Portfolio_Value"] / rolling_peak - 1
)

max_drawdown = drawdown.min() * 100


# 8. DISPLAY RESULTS

print("\n========== BACKTEST RESULTS ==========")

print(f"Stock: {TICKER}")
print(f"Starting Capital: ${INITIAL_CAPITAL:,.2f}")

print(f"Final Portfolio Value: ${data['Portfolio_Value'].iloc[-1]:,.2f}")

print(f"Strategy Return: {total_return:.2f}%")

print(f"Buy & Hold Return: {buy_hold_return:.2f}%")

print(f"Annualised Volatility: {annual_volatility:.2%}")

print(f"Sharpe Ratio (0% risk-free rate): {sharpe_ratio:.2f}")

print(f"Maximum Drawdown: {max_drawdown:.2f}%")


# 9. PLOT PORTFOLIO PERFORMANCE

plt.figure(figsize=(14, 7))

plt.plot(
    data.index,
    data["Portfolio_Value"],
    label="Moving Average Strategy"
)

plt.plot(
    data.index,
    data["Buy_Hold"],
    label="Buy and Hold",
    alpha=0.7
)

plt.title("AAPL Moving Average Crossover Backtest")

plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("backtest_results.png")

plt.show()

print("\nBacktest completed successfully!")
print("Chart saved as backtest_results.png")
