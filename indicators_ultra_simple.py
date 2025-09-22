"""
Versão ULTRA simplificada para debug - apenas SMAs.
"""
import pandas as pd
from typing import Dict, Optional

def compute_indicators_ultra_simple(close: pd.Series, p, volume: pd.Series = None) -> Dict[str, Optional[float]]:
    """
    VERSÃO ULTRA SIMPLIFICADA - apenas SMAs para identificar o problema.
    """
    print(f"DEBUG: compute_indicators_ultra_simple called")
    
    try:
        # Apenas o essencial
        current_price = float(close.iloc[-1])
        print(f"DEBUG: Current price: {current_price}")
        
        # SMAs básicas - usando conversão explícita para float
        sma_short_series = close.rolling(window=p.sma_short, min_periods=1).mean()
        sma_long_series = close.rolling(window=p.sma_long, min_periods=1).mean()
        
        sma_short = float(sma_short_series.iloc[-1])
        sma_long = float(sma_long_series.iloc[-1])
        
        print(f"DEBUG: SMAs computed: short={sma_short}, long={sma_long}")
        print(f"DEBUG: All indicators computed successfully!")
        
        return {
            "price": current_price,
            "sma_short": sma_short,
            "sma_long": sma_long,
            "rsi": 50.0,  # Valor fixo
            "macd_hist": 0.0,  # Valor fixo
            "vol_ratio": 1.0,  # Valor fixo
            "adx": None,
            "obv": None,
        }
        
    except Exception as e:
        print(f"DEBUG: ERROR in compute_indicators_ultra_simple: {e}")
        import traceback
        traceback.print_exc()
        raise
