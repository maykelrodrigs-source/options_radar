"""
Parâmetros de horizonte temporal para análise técnica.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class HorizonParams:
    """Parâmetros de análise técnica por horizonte temporal."""
    history_days: int
    sma_short: int
    sma_long: int
    rsi_len: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    vol_window: int
    weights: Dict[str, float]  # {'trend':..,'rsi':..,'macd':..,'volume':..}


HORIZON_PRESETS = {
    "1-4 semanas": HorizonParams(
        history_days=90,
        sma_short=10,
        sma_long=50,
        rsi_len=7,  # Mais sensível
        macd_fast=8,
        macd_slow=21,
        macd_signal=5,
        vol_window=5,  # Volume 5d vs 20d
        weights={"trend": 0.30, "momentum": 0.50, "volume": 0.20},
    ),
    "3-6 meses": HorizonParams(
        history_days=260,
        sma_short=20,
        sma_long=100,
        rsi_len=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        vol_window=20,
        weights={"trend": 0.45, "momentum": 0.40, "volume": 0.15},
    ),
    "6-12 meses": HorizonParams(
        history_days=400,
        sma_short=50,
        sma_long=200,
        rsi_len=21,  # Mais lento
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        vol_window=20,
        weights={"trend": 0.65, "momentum": 0.25, "volume": 0.10},
    ),
}
