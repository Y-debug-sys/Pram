import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def arima_predict(x, horizon=1, order=(2,1,2)):
    """
    x: torch.Tensor of shape [B, S, F]
    horizon: steps ahead to predict
    order: ARIMA order (p,d,q)
    return: torch.Tensor of shape [B, horizon, F]
    """
    B, S, F = x.shape
    preds = np.zeros((B, horizon, F))
    
    for b in range(B):
        for f in range(F):
            series = x[b, :, f].cpu().numpy()
            try:
                model = ARIMA(series, order=order)
                fitted = model.fit()
                forecast = fitted.forecast(steps=horizon)
            except Exception:
                forecast = np.repeat(series[-1], horizon)
            preds[b, :, f] = forecast
    
    return torch.tensor(preds, dtype=x.dtype)


def last_value(x, horizon=1):
    """
    Repeat the last observed value for all future steps.
    Non-autoregressive: every forecast step gets the same value.
    Args:
        x: input tensor [B, S, F]
        horizon: number of steps to predict
    Returns:
        [B, horizon, F]
    """
    last = x[:, -1:, :]          # (B,1,F)
    return last.repeat(1, horizon, 1)


def mean_forecast(x, horizon=1):
    """
    Predict the average of the entire history for all future steps.
    Args:
        x: [B, S, F]
        horizon: number of steps to predict
    Returns:
        [B, horizon, F]
    """
    mean = x.mean(dim=1, keepdim=True)  # (B,1,F)
    return mean.repeat(1, horizon, 1)


def moving_average(x, horizon=1, window=5):
    """
    Predict the average of the last `window` steps for all future steps.
    Args:
        x: [B, S, F]
        horizon: number of steps to predict
        window: how many past steps to average
    Returns:
        [B, horizon, F]
    """
    B, S, F = x.shape
    w = min(window, S)
    ma = x[:, -w:, :].mean(dim=1, keepdim=True)  # (B,1,F)
    return ma.repeat(1, horizon, 1)


def seasonal_naive(x, horizon=1, season_len=12):
    """
    Seasonal naive: repeat the last observed season into the future.
    If the sequence is shorter than `season_len`, use as many steps as possible.
    Args:
        x: [B, S, F]
        horizon: number of steps to predict
        season_len: assumed periodicity
    Returns:
        [B, horizon, F]
    """
    B, S, F = x.shape
    if season_len <= 0:
        raise ValueError("season_len must be > 0")
    L = min(season_len, S)
    frag = x[:, -L:, :]  # last season fragment (B,L,F)
    reps = (horizon + L - 1) // L
    out = frag.repeat(1, reps, 1)[:, :horizon, :]  # repeat and trim
    return out


def linear_trend(x, horizon=1):
    """
    Linear extrapolation based on the last two points.
    Each future step is last + k * delta, where delta = last - prev.
    If the sequence has length 1, fall back to last_value.
    Args:
        x: [B, S, F]
        horizon: number of steps to predict
    Returns:
        [B, horizon, F]
    """
    B, S, F = x.shape
    if S == 1:
        return last_value(x, horizon)
    last = x[:, -1, :]    # (B, F)
    prev = x[:, -2, :]    # (B, F)
    delta = last - prev   # increment
    steps = torch.arange(1, horizon + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)  # (1,h,1)
    out = last.view(B, 1, F) + steps * delta.view(B, 1, F)
    return out


# --------- example usage ----------
if __name__ == "__main__":
    B, S, F = 2, 10, 3
    x = torch.randn(B, S, F)

    h = 5
    print("last_value:", last_value(x, h).shape)
    print("mean_forecast:", mean_forecast(x, h).shape)
    print("moving_average:", moving_average(x, h, window=3).shape)
    print("seasonal_naive:", seasonal_naive(x, h, season_len=4).shape)
    print("linear_trend:", linear_trend(x, h).shape)


