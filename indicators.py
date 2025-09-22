"""
Módulo para cálculo de indicadores técnicos com janelas dinâmicas.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def compute_indicators(close: pd.Series, p, volume: pd.Series = None) -> Dict[str, Optional[float]]:
    """
    Calcula indicadores técnicos usando janelas dinâmicas baseadas no horizonte.
    
    Args:
        close: Série de preços de fechamento
        p: Parâmetros do horizonte (HorizonParams)
        volume: Série de volume (opcional)
        
    Returns:
        Dicionário com indicadores calculados
    """
    print(f"DEBUG: compute_indicators called with close type={type(close)}, volume type={type(volume)}")
    print(f"DEBUG: close values: {close.head() if hasattr(close, 'head') else 'No head method'}")
    
    df = pd.DataFrame({"close": close})
    if volume is not None:
        df["volume"] = volume
    
    # Médias móveis
    print(f"DEBUG: Computing SMAs with windows: short={p.sma_short}, long={p.sma_long}")
    df["sma_short"] = df["close"].rolling(window=p.sma_short, min_periods=1).mean()
    df["sma_long"] = df["close"].rolling(window=p.sma_long, min_periods=1).mean()
    print(f"DEBUG: SMAs computed successfully")
    
    # RSI
    print(f"DEBUG: Computing RSI with period: {p.rsi_len}")
    rsi = _calculate_rsi(df["close"], p.rsi_len)
    print(f"DEBUG: RSI computed successfully")
    
    # MACD
    print(f"DEBUG: Computing MACD with params: fast={p.macd_fast}, slow={p.macd_slow}, signal={p.macd_signal}")
    macd_line, macd_signal, macd_hist = _calculate_macd(
        df["close"], p.macd_fast, p.macd_slow, p.macd_signal
    )
    print(f"DEBUG: MACD computed successfully")
    
    # Volume ratio (diferente por horizonte)
    print(f"DEBUG: Computing volume ratio")
    vol_ratio = _calculate_volume_ratio(df, p)
    print(f"DEBUG: Volume ratio computed: {vol_ratio}")
    
    # ADX (apenas para médio prazo)
    adx = None
    if p.history_days >= 260:  # Médio e longo prazo
        print(f"DEBUG: Computing ADX for medium/long term")
        adx = _calculate_adx(df)
        print(f"DEBUG: ADX computed: {adx}")
    
    # OBV (apenas para longo prazo)
    obv = None
    if p.history_days >= 400 and volume is not None:  # Longo prazo
        print(f"DEBUG: Computing OBV for long term")
        obv = _calculate_obv(df)
        print(f"DEBUG: OBV computed: {obv}")
    
    # Função auxiliar para pegar último valor válido
    def last_valid(series):
        if series is None:
            return None
        if hasattr(series, 'empty') and series.empty:
            return None
        if hasattr(series, 'dropna'):
            clean_series = series.dropna()
            if clean_series.empty:
                return None
            return float(clean_series.iloc[-1])
        # Se não é uma Series, tenta converter diretamente
        try:
            return float(series) if series is not None else None
        except (TypeError, ValueError):
            return None
    
    return {
        "price": float(df["close"].iloc[-1]),
        "sma_short": last_valid(df["sma_short"]),
        "sma_long": last_valid(df["sma_long"]),
        "rsi": last_valid(rsi),
        "macd_hist": last_valid(macd_hist),
        "vol_ratio": vol_ratio,
        "adx": last_valid(adx) if adx is not None else None,  # ADX para médio/longo prazo
        "obv": last_valid(obv) if obv is not None else None,  # OBV para longo prazo
    }


def _calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
    """Calcula RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    
    # Evita divisão por zero
    rs = gain / loss.where(loss != 0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_macd(prices: pd.Series, fast: int, slow: int, signal: int) -> tuple:
    """Calcula MACD."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    
    return macd_line, macd_signal, macd_hist


def _calculate_volume_ratio(df: pd.DataFrame, p) -> Optional[float]:
    """Calcula ratio de volume baseado no horizonte."""
    if "volume" not in df.columns:
        # Usa volatilidade como proxy para volume
        vol = df["close"].rolling(window=p.vol_window).std()
        if len(vol.dropna()) > p.vol_window * 2:
            recent_vol = vol.iloc[-p.vol_window:].mean()
            historical_vol = vol.iloc[-p.vol_window*2:-p.vol_window].mean()
            historical_vol_val = float(historical_vol) if historical_vol is not None else 0
            if historical_vol_val > 0:
                return float(recent_vol / historical_vol_val)
        return None
    else:
        # Volume real: 5d vs 20d para curto prazo, 20d vs 40d para outros
        if p.vol_window == 5:  # Curto prazo
            recent_vol = df["volume"].iloc[-5:].mean()
            historical_vol = df["volume"].iloc[-20:-5].mean()
        else:  # Médio e longo prazo
            recent_vol = df["volume"].iloc[-20:].mean()
            historical_vol = df["volume"].iloc[-40:-20].mean()
        
        historical_vol_val = float(historical_vol) if historical_vol is not None else 0
        if historical_vol_val > 0:
            return float(recent_vol / historical_vol_val)
        return None


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Calcula ADX (Average Directional Index)."""
    high = df["close"]  # Usando close como proxy para high/low
    low = df["close"] * 0.98  # Proxy simplificado
    
    # True Range
    tr1 = high.diff().abs()
    tr2 = (high - low.shift()).abs()
    tr3 = (low - low.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = high.diff()
    dm_minus = -low.diff()
    
    # Evita erro de ambiguidade com comparações de Series
    dm_plus_mask = (dm_plus > dm_minus) & (dm_plus > 0)
    dm_minus_mask = (dm_minus > dm_plus) & (dm_minus > 0)
    
    dm_plus = dm_plus.where(dm_plus_mask, 0)
    dm_minus = dm_minus.where(dm_minus_mask, 0)
    
    # Smoothed values
    atr = tr.rolling(window=period).mean()
    di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
    di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)
    
    # ADX
    di_sum = di_plus + di_minus
    # Evita divisão por zero
    di_sum = di_sum.where(di_sum != 0, 0.01)  # Substitui zeros por valor pequeno
    dx = 100 * ((di_plus - di_minus).abs() / di_sum)
    adx = dx.rolling(window=period).mean()
    
    return float(adx.dropna().iloc[-1]) if not adx.dropna().empty else None


def _calculate_obv(df: pd.DataFrame) -> Optional[float]:
    """Calcula OBV (On-Balance Volume)."""
    if "volume" not in df.columns:
        return None
    
    price_change = df["close"].diff()
    # Evita erro de ambiguidade com comparações de Series
    up_mask = price_change > 0
    down_mask = price_change < 0
    
    obv = np.where(up_mask, df["volume"], 
                   np.where(down_mask, -df["volume"], 0))
    
    obv_series = pd.Series(obv, index=df.index).cumsum()
    
    # Retorna a tendência do OBV (últimos 20 vs 40 períodos)
    if len(obv_series) > 40:
        recent_obv = obv_series.iloc[-20:].mean()
        historical_obv = obv_series.iloc[-40:-20].mean()
        return float(recent_obv - historical_obv)
    
    return None
