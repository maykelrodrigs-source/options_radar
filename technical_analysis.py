"""
Módulo para análise técnica e decisão de direção (CALL/PUT).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    CALL = "CALL"
    PUT = "PUT"
    NEUTRAL = "NEUTRAL"


class RiskProfile(Enum):
    SHORT_TERM = "0-3 meses"
    MEDIUM_TERM = "3-6 meses"
    LONG_TERM = "6-12 meses"


@dataclass
class TechnicalSignal:
    """Sinal técnico com indicadores calculados."""
    current_price: float
    sma_50: float
    sma_200: float
    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    volume_ratio_20d: float
    direction: Direction
    confidence: float
    reasoning: str
    price_history: Optional[pd.DataFrame] = None


class TechnicalAnalyzer:
    """Analisador técnico para decisão de direção em opções."""
    
    def __init__(self):
        self.rsi_period = 14
        self.sma_periods = [50, 200]
        self.macd_params = (12, 26, 9)
        self.volume_period = 20
    
    def calculate_technical_indicators(self, price_data: pd.DataFrame) -> TechnicalSignal:
        """
        Calcula indicadores técnicos e determina direção.
        
        Args:
            price_data: DataFrame com colunas ['date', 'close', 'volume']
        """
        if len(price_data) < 50:
            raise ValueError("Dados insuficientes. Necessário pelo menos 50 períodos para análise básica.")
        
        # Ajusta períodos baseado nos dados disponíveis
        if len(price_data) < 200:
            self.sma_periods = [min(50, len(price_data)//4), min(200, len(price_data)//2)]
            self.rsi_period = min(14, len(price_data)//10)
            # Ajusta MACD para dados menores
            if len(price_data) < 100:
                self.macd_params = (8, 17, 6)  # MACD mais rápido
        
        # Garante que está ordenado por data
        price_data = price_data.sort_values('date').reset_index(drop=True)
        
        # Calcula indicadores
        sma_50 = self._calculate_sma(price_data['close'], self.sma_periods[0])
        sma_200 = self._calculate_sma(price_data['close'], self.sma_periods[1])
        rsi_14 = self._calculate_rsi(price_data['close'], self.rsi_period)
        macd_line, macd_signal, macd_histogram = self._calculate_macd(
            price_data['close'], *self.macd_params
        )
        volume_ratio = self._calculate_volume_ratio(price_data['volume'], self.volume_period)
        
        # Dados atuais (último período)
        current_price = price_data['close'].iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        current_sma_200 = sma_200.iloc[-1]
        current_rsi = rsi_14.iloc[-1]
        current_macd_line = macd_line.iloc[-1]
        current_macd_signal = macd_signal.iloc[-1]
        current_macd_histogram = macd_histogram.iloc[-1]
        current_volume_ratio = volume_ratio.iloc[-1]
        
        
        # Determina direção e confiança
        direction, confidence, reasoning = self._determine_direction(
            current_price, current_sma_50, current_sma_200, current_rsi,
            current_macd_line, current_macd_signal, current_macd_histogram,
            current_volume_ratio
        )
        
        return TechnicalSignal(
            current_price=current_price,
            sma_50=current_sma_50,
            sma_200=current_sma_200,
            rsi_14=current_rsi,
            macd_line=current_macd_line,
            macd_signal=current_macd_signal,
            macd_histogram=current_macd_histogram,
            volume_ratio_20d=current_volume_ratio,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            price_history=price_data
        )
    
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcula média móvel simples."""
        return prices.rolling(window=period, min_periods=period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcula RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_volume_ratio(self, volume: pd.Series, period: int) -> pd.Series:
        """Calcula ratio de volume relativo."""
        avg_volume = volume.rolling(window=period, min_periods=period).mean()
        return volume / avg_volume
    
    def _determine_direction(self, price: float, sma_50: float, sma_200: float,
                           rsi: float, macd_line: float, macd_signal: float,
                           macd_histogram: float, volume_ratio: float) -> Tuple[Direction, float, str]:
        """
        Determina direção baseada em indicadores técnicos.
        
        Regras:
        - CALL: Preço > SMA200, RSI 45-65, MACD > 0
        - PUT: Preço < SMA200, RSI > 70 ou MACD < 0
        - NEUTRAL: Casos contrários
        """
        bullish_signals = 0
        bearish_signals = 0
        reasoning_parts = []
        
        # Análise de tendência (SMA) - mais balanceada
        if price > sma_200:
            bullish_signals += 1
            reasoning_parts.append(f"Preço (R${price:.2f}) acima da SMA200 (R${sma_200:.2f})")
        elif price < sma_200:
            bearish_signals += 1
            reasoning_parts.append(f"Preço (R${price:.2f}) abaixo da SMA200 (R${sma_200:.2f})")
        
        # Análise de momentum (RSI) - mais balanceada
        if 30 <= rsi <= 70:
            # RSI neutro - não adiciona sinal
            reasoning_parts.append(f"RSI neutro ({rsi:.1f}) - sem sinal claro")
        elif rsi > 70:
            bearish_signals += 1
            reasoning_parts.append(f"RSI sobrecomprado ({rsi:.1f}) - risco de correção")
        elif rsi < 30:
            bullish_signals += 1
            reasoning_parts.append(f"RSI sobrevendido ({rsi:.1f}) - possível reversão")
        
        # Análise de momentum (MACD) - mais balanceada
        if macd_line > 0:
            bullish_signals += 1
            reasoning_parts.append("MACD positivo")
        elif macd_line < 0:
            bearish_signals += 1
            reasoning_parts.append("MACD negativo")
        
        # Análise de volume - simplificada
        if volume_ratio > 1.5:
            reasoning_parts.append(f"Volume alto ({volume_ratio:.1f}x) - confirma movimento")
        
        # Determina direção
        total_signals = bullish_signals + bearish_signals
        
        
        if total_signals == 0:
            direction = Direction.NEUTRAL
            confidence = 0.0
            reasoning = "Sinais técnicos inconclusivos"
        elif bullish_signals > bearish_signals:
            direction = Direction.CALL
            # Confiança mais realista: baseada na força do sinal, limitada a 85%
            raw_confidence = bullish_signals / total_signals
            confidence = min(0.85, 0.50 + (raw_confidence - 0.5) * 0.7)  # 50% a 85%
            reasoning = "Tendência de alta: " + " | ".join(reasoning_parts)
        elif bearish_signals > bullish_signals:
            direction = Direction.PUT
            # Confiança mais realista: baseada na força do sinal, limitada a 85%
            raw_confidence = bearish_signals / total_signals
            confidence = min(0.85, 0.50 + (raw_confidence - 0.5) * 0.7)  # 50% a 85%
            reasoning = "Tendência de baixa: " + " | ".join(reasoning_parts)
        else:
            direction = Direction.NEUTRAL
            confidence = 0.45  # Ligeiramente abaixo de 50%
            reasoning = "Sinais equilibrados: " + " | ".join(reasoning_parts)
        
        
        return direction, confidence, reasoning


def get_option_parameters_by_direction(direction: Direction, current_price: float, 
                                     risk_profile: RiskProfile) -> Dict[str, float]:
    """
    Retorna parâmetros de opções baseados na direção e perfil de risco.
    
    Args:
        direction: Direção identificada (CALL/PUT/NEUTRAL)
        current_price: Preço atual da ação
        risk_profile: Perfil de risco do investidor
    
    Returns:
        Dicionário com parâmetros para busca de opções
    """
    if direction == Direction.NEUTRAL:
        return {}
    
    # Define parâmetros baseados no perfil de prazo
    risk_params = {
        RiskProfile.SHORT_TERM: {
            "distance_pct": 0.05,  # 5% - strikes próximos para curto prazo
            "delta_min": 0.05,     # Delta mais flexível
            "delta_max": 0.50,     # Delta mais flexível
            "min_days": 7,         # Mínimo 7 dias (mais flexível)
            "max_days": 35         # Máximo 35 dias (5 semanas)
        },
        RiskProfile.MEDIUM_TERM: {
            "distance_pct": 0.10,  # 10% - strikes moderados
            "delta_min": 0.08,     # Delta mais flexível
            "delta_max": 0.35,     # Delta mais flexível
            "min_days": 30,        # Mínimo 30 dias
            "max_days": 120        # Máximo 4 meses
        },
        RiskProfile.LONG_TERM: {
            "distance_pct": 0.12,  # 12% - strikes mais distantes
            "delta_min": 0.05,     # Delta mais baixo
            "delta_max": 0.25,     # Delta mais baixo
            "min_days": 90,        # Mínimo 3 meses
            "max_days": 365        # Máximo 12 meses
        }
    }
    
    params = risk_params[risk_profile]
    
    if direction == Direction.CALL:
        return {
            "min_days": params["min_days"],
            "max_days": params["max_days"],
            "min_volume": 10,
            "max_exercise_prob": abs(params["delta_max"]) * 100,  # Converte delta para probabilidade
            "option_types": "Apenas CALL"
        }
    else:  # PUT
        return {
            "min_days": params["min_days"],
            "max_days": params["max_days"],
            "min_volume": 10,
            "max_exercise_prob": abs(params["delta_min"]) * 100,  # Converte delta para probabilidade
            "option_types": "Apenas PUT"
        }


def generate_direction_justification(direction: Direction, option_type: str, 
                                   strike: float, premium: float, 
                                   current_price: float, days_to_expiry: int,
                                   confidence: float) -> str:
    """
    Gera justificativa específica para opção baseada na direção.
    
    Args:
        direction: Direção identificada
        option_type: Tipo da opção (CALL/PUT)
        strike: Strike da opção
        premium: Prêmio da opção
        current_price: Preço atual da ação
        days_to_expiry: Dias até vencimento
        confidence: Confiança do sinal técnico
    """
    distance_pct = abs(strike - current_price) / current_price * 100
    premium_pct = premium / current_price * 100
    
    if direction == Direction.CALL and option_type == "CALL":
        return f"Aproveita tendência de alta (confiança {confidence:.0%}). Strike {distance_pct:.0f}% OTM, custo {premium_pct:.2f}% em {days_to_expiry}d"
    elif direction == Direction.PUT and option_type == "PUT":
        return f"Protege contra queda (confiança {confidence:.0%}). Strike {distance_pct:.0f}% OTM, custo {premium_pct:.2f}% em {days_to_expiry}d"
    else:
        return f"Opção {option_type} - Strike {distance_pct:.0f}% OTM, custo {premium_pct:.2f}% em {days_to_expiry}d"
