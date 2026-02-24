"""
regime_engine.py — HMM-based Market Regime Detection.

Uses hmmlearn.GaussianHMM with 7 hidden components trained on three features:
  1. Log Returns
  2. Normalised Price Range  (High − Low) / Close
  3. Volume Volatility       rolling std of log-volume changes

After fitting, the engine automatically labels states:
  • Bull Run    — component with the highest mean log-return
  • Bear/Crash  — component with the lowest mean log-return
  • Neutral     — all other components
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# Canonical regime labels
REGIME_BULL = "Bull Run"
REGIME_BEAR = "Bear/Crash"
REGIME_NEUTRAL = "Neutral"


class RegimeEngine:
    """Wraps a GaussianHMM and exposes simple fit / predict helpers."""

    def __init__(self, n_components: int = 7, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: hmm.GaussianHMM | None = None
        self.bull_state: int | None = None
        self.bear_state: int | None = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(self, data: pd.DataFrame) -> np.ndarray:
        close = data["Close"].values.astype(float)
        high = data["High"].values.astype(float)
        low = data["Low"].values.astype(float)
        volume = data["Volume"].values.astype(float)

        # Feature 1: log returns
        log_c = np.log(np.clip(close, 1e-10, None))
        returns = np.diff(log_c, prepend=log_c[0])

        # Feature 2: normalised intrabar range
        price_range = (high - low) / np.clip(close, 1e-10, None)

        # Feature 3: rolling std of log-volume changes (window = 5)
        log_vol = np.log(volume + 1.0)
        vol_diff = np.diff(log_vol, prepend=log_vol[0])
        vol_vol = (
            pd.Series(vol_diff)
            .rolling(window=5, min_periods=1)
            .std()
            .fillna(0.0)
            .values
        )

        return np.column_stack([returns, price_range, vol_vol])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the HMM on historical OHLCV data."""
        features = self._build_features(data)
        scaled = self.scaler.fit_transform(features)

        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=200,
            random_state=self.random_state,
            tol=1e-4,
        )
        self.model.fit(scaled)

        # Identify bull/bear states from the unscaled mean returns
        unscaled_means = self.scaler.inverse_transform(self.model.means_)
        mean_returns = unscaled_means[:, 0]
        self.bull_state = int(np.argmax(mean_returns))
        self.bear_state = int(np.argmin(mean_returns))
        self.is_fitted = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_states(self, data: pd.DataFrame) -> np.ndarray:
        """Return the Viterbi-decoded hidden state sequence."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_states().")
        features = self._build_features(data)
        scaled = self.scaler.transform(features)
        return self.model.predict(scaled)

    def state_to_label(self, state: int) -> str:
        if state == self.bull_state:
            return REGIME_BULL
        if state == self.bear_state:
            return REGIME_BEAR
        return REGIME_NEUTRAL

    def get_regime_series(self, data: pd.DataFrame) -> pd.Series:
        """Return a Series of regime labels aligned with `data`."""
        states = self.predict_states(data)
        labels = [self.state_to_label(s) for s in states]
        return pd.Series(labels, index=data.index, name="Regime")

    def current_regime(self, data: pd.DataFrame) -> str:
        """Return the regime label for the most recent bar."""
        return self.get_regime_series(data).iloc[-1]
