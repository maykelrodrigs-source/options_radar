"""
Módulo para decisão de direção baseada em indicadores técnicos.
"""

from typing import Tuple
from enum import Enum


class Direction(Enum):
    CALL = "CALL"
    PUT = "PUT"
    NEUTRAL = "NEUTRAL"


def direction_signal(ind: dict, weights: dict) -> Tuple[str, int, float, str]:
    """
    Calcula sinal de direção baseado em indicadores e pesos com filtros robustos.
    
    Args:
        ind: Dicionário com indicadores calculados
        weights: Pesos para cada categoria de indicador
        
    Returns:
        Tupla (direção, confiança, score, motivo)
    """
    print(f"DEBUG: direction_signal called with ind={ind}")
    print(f"DEBUG: weights={weights}")
    
    score = 0.0
    signals = []
    
    # 1. TENDÊNCIA (peso maior no longo prazo)
    trend_score = 0
    if ind["price"] is not None and ind["sma_long"] is not None:
        price_val = float(ind["price"])
        sma_long_val = float(ind["sma_long"])
        trend_score += (1 if price_val > sma_long_val else -1)
        if price_val > sma_long_val:
            signals.append(f"Preço > SMA{ind.get('sma_long_period', 'longa')}")
        else:
            signals.append(f"Preço < SMA{ind.get('sma_long_period', 'longa')}")
    
    # Tendência secundária (SMA curta vs SMA longa)
    if ind["sma_short"] is not None and ind["sma_long"] is not None:
        sma_short_val = float(ind["sma_short"])
        sma_long_val = float(ind["sma_long"])
        trend_score += 0.5 * (1 if sma_short_val > sma_long_val else -1)
        if sma_short_val > sma_long_val:
            signals.append(f"SMA{ind.get('sma_short_period', 'curta')} > SMA{ind.get('sma_long_period', 'longa')}")
    
    score += weights["trend"] * trend_score
    
    # 2. MOMENTUM (RSI + MACD)
    momentum_score = 0
    momentum_signals = []
    
    # RSI
    if ind["rsi"] is not None:
        rsi_val = float(ind["rsi"])
        if rsi_val >= 65:  # Sobrecompra
            momentum_score += 1
            momentum_signals.append(f"RSI sobrecomprado ({rsi_val:.1f})")
        elif rsi_val <= 35:  # Sobrevenda
            momentum_score += -1
            momentum_signals.append(f"RSI sobrevendido ({rsi_val:.1f})")
        elif 45 <= rsi_val <= 55:  # Neutro
            momentum_score += 0
        else:  # Levemente sobrecomprado/vendido
            momentum_score += 0.5 if rsi_val > 55 else -0.5
    
    # MACD
    if ind["macd_hist"] is not None:
        macd_val = float(ind["macd_hist"])
        if macd_val > 0.001:  # Histograma positivo forte
            momentum_score += 1
            momentum_signals.append("MACD positivo")
        elif macd_val < -0.001:  # Histograma negativo forte
            momentum_score += -1
            momentum_signals.append("MACD negativo")
        else:  # Neutro
            momentum_score += 0.5 if macd_val > 0 else -0.5
    
    score += weights["momentum"] * momentum_score
    signals.extend(momentum_signals)
    
    # 3. VOLUME/CONFIRMAÇÃO
    volume_score = 0
    if ind["vol_ratio"] is not None:
        vol_val = float(ind["vol_ratio"])
        if vol_val > 1.2:  # Volume alto
            volume_score += 0.5  # Confirma movimento
            signals.append(f"Volume alto ({vol_val:.1f}x)")
        elif vol_val < 0.8:  # Volume baixo
            volume_score += -0.2  # Indica fraqueza
            signals.append(f"Volume baixo ({vol_val:.1f}x)")
    
    # ADX para médio/longo prazo
    if ind.get("adx") is not None:
        adx_val = float(ind["adx"])
        if adx_val > 25:  # Tendência forte
            volume_score += 0.3
            signals.append(f"ADX forte ({adx_val:.1f})")
        elif adx_val < 15:  # Tendência fraca
            volume_score += -0.3
            signals.append(f"ADX fraco ({adx_val:.1f})")
    
    # OBV para longo prazo
    if ind.get("obv") is not None:
        obv_val = float(ind["obv"])
        if obv_val > 0:  # Acumulação
            volume_score += 0.2
            signals.append("OBV acumulação")
        else:  # Distribuição
            volume_score += -0.2
            signals.append("OBV distribuição")
    
    score += weights["volume"] * volume_score
    
    # 4. FILTROS DE CONFIRMAÇÃO
    confirmation_count = 0
    if trend_score != 0:
        confirmation_count += 1
    if momentum_score != 0:
        confirmation_count += 1
    if volume_score != 0:
        confirmation_count += 1
    
    # 5. ZONA NEUTRA EXPANDIDA
    if abs(score) < 0.2:  # Zona neutra expandida
        direction = Direction.NEUTRAL.value
        confidence = int(abs(score) * 100)
        reason = "Sinais fracos - " + ", ".join(signals[:2]) if signals else "Sinais inconclusivos"
    elif confirmation_count < 2:  # Filtro de confirmação
        direction = Direction.NEUTRAL.value
        confidence = int(abs(score) * 100)
        reason = "Confirmação insuficiente - apenas " + str(confirmation_count) + " indicador(es)"
    elif score > 0.2:
        direction = Direction.CALL.value
        confidence = min(85, int(abs(score) * 100))  # Limita em 85%
        reason = "Tendência de alta - " + ", ".join(signals[:3])
    else:  # score < -0.2
        direction = Direction.PUT.value
        confidence = min(85, int(abs(score) * 100))  # Limita em 85%
        reason = "Tendência de baixa - " + ", ".join(signals[:3])
    
    return direction, confidence, score, reason
