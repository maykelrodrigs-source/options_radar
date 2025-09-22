"""
Versão simplificada dos indicadores para debug.
"""
import pandas as pd
from typing import Dict, Optional

def compute_indicators_simple(close: pd.Series, p, volume: pd.Series = None) -> Dict[str, Optional[float]]:
    """
    VERSÃO SIMPLIFICADA PARA DEBUG - Calcula indicadores técnicos básicos.
    """
    print(f"DEBUG: compute_indicators_simple called")
    
    try:
        # Apenas indicadores básicos para testar
        current_price = float(close.iloc[-1])
        print(f"DEBUG: Current price: {current_price}")
        
        # SMAs básicas
        sma_short = float(close.rolling(window=p.sma_short, min_periods=1).mean().iloc[-1])
        sma_long = float(close.rolling(window=p.sma_long, min_periods=1).mean().iloc[-1])
        print(f"DEBUG: SMAs computed: short={sma_short}, long={sma_long}")
        
        # RSI simplificado
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=p.rsi_len, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=p.rsi_len, min_periods=1).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi_val = float(rsi_series.iloc[-1])
        print(f"DEBUG: RSI computed: {rsi_val}")
        
        # MACD simplificado
        ema_fast = close.ewm(span=p.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=p.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=p.macd_signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        macd_val = float(macd_hist.iloc[-1])
        print(f"DEBUG: MACD computed: {macd_val}")
        
        print(f"DEBUG: All indicators computed successfully!")
        
        return {
            "price": current_price,
            "sma_short": sma_short,
            "sma_long": sma_long,
            "rsi": rsi_val,
            "macd_hist": macd_val,
            "vol_ratio": 1.0,  # Valor fixo para teste
            "adx": None,
            "obv": None,
        }
        
    except Exception as e:
        print(f"DEBUG: ERROR in compute_indicators_simple: {e}")
        import traceback
        traceback.print_exc()
        raise
