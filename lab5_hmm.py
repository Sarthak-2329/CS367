# lab5_hmm.py
# CS367 - Lab 5: Gaussian Hidden Markov Model on stock returns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import yfinance as yf

# 1. Data download + returns

ticker = "AAPL"
data = yf.download(ticker, period="10y")

prices = data["Close"].dropna()

returns = np.log(prices / prices.shift(1)).dropna()
X = returns.values.reshape(-1, 1)

# 2. Fit Gaussian HMM (2 states)

model = GaussianHMM(
    n_components=2,
    covariance_type="full",
    n_iter=200,
    random_state=42
)
model.fit(X)

hidden_states = model.predict(X)

means = model.means_.flatten()
variances = np.array([model.covars_[i][0][0] for i in range(2)])
transmat = model.transmat_

print("State means:", means)
print("State variances:", variances)
print("Transition matrix:\n", transmat)

# Expected duration of each state
durations = 1 / (1 - np.diag(transmat))
print("Expected durations (in days):", durations)

# 3. Visualization

plt.figure(figsize=(10, 4))
plt.plot(returns.index, returns.values)
plt.title(f"{ticker} Daily Log Returns")
plt.xlabel("Date")
plt.ylabel("Log return")
plt.tight_layout()
plt.savefig("returns_plot.png")
plt.close()

plt.figure(figsize=(10, 4))
for state in range(2):
    idx = (hidden_states == state)
    plt.scatter(
        returns.index[idx],
        returns.values[idx],
        s=5,
        label=f"State {state}"
    )
plt.title(f"{ticker} Returns with HMM Hidden States")
plt.xlabel("Date")
plt.ylabel("Log return")
plt.legend()
plt.tight_layout()
plt.savefig("hmm_states.png")
plt.close()
