"""
Sistema de análise profissional para Radar de Direção.
Implementa análise em 6 camadas: tendência, momentum, volume, sentimento, macro e decisão.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math
import json
from datetime import datetime, timedelta

from src.core.data.oplab_client import OpLabClient


class Direction(Enum):
    CALL = "CALL"
    PUT = "PUT"
    NEUTRAL = "NEUTRAL"


class TrendStrength(Enum):
    STRONG_UP = "FORTE ALTA"
    WEAK_UP = "ALTA FRACA"
    LATERAL = "LATERAL"
    WEAK_DOWN = "BAIXA FRACA"
    STRONG_DOWN = "FORTE BAIXA"


@dataclass
class TrendAnalysis:
    """Análise de tendência estrutural."""
    sma_10: float
    sma_50: float
    sma_100: float
    sma_200: float
    golden_cross: bool
    death_cross: bool
    volatility_regime: str  # "ALTA", "NORMAL", "BAIXA"
    atr_ratio: float
    trend_score: float
    trend_strength: TrendStrength


@dataclass
class MomentumAnalysis:
    """Análise de momentum e força."""
    rsi_7: float
    rsi_14: float
    rsi_21: float
    macd_histogram: float
    macd_slope: float
    macd_signal: float
    adx: float
    roc_10: float
    roc_20: float
    
    # Indicadores de reversão para médio prazo
    bb_position: float  # Posição nas Bollinger Bands (-1 a +1)
    bb_squeeze: bool    # Se está em squeeze (baixa volatilidade)
    stoch_k: float      # %K do Stochastic
    stoch_d: float      # %D do Stochastic
    stoch_signal: str   # "OVERSOLD", "OVERBOUGHT", "NEUTRAL"
    # EMAs rápidas
    ema_9: float
    ema_21: float
    # Money Flow Index
    mfi_14: float
    
    momentum_score: float
    momentum_strength: str


@dataclass
class VolumeFlowAnalysis:
    """Análise de volume e fluxo."""
    volume_ratio_5d: float
    volume_ratio_20d: float
    obv_trend: float
    accumulation_distribution: float
    volume_score: float
    flow_direction: str  # "ENTRADA", "SAÍDA", "NEUTRO"


@dataclass
class OptionsSentimentAnalysis:
    """Análise de sentimento via opções."""
    put_call_ratio: float
    volatility_skew: float
    call_volume_ratio: float
    put_volume_ratio: float
    sentiment_score: float
    market_bias: str  # "BULLISH", "BEARISH", "NEUTRAL"


@dataclass
class MacroContextAnalysis:
    """Análise de contexto macro e setorial."""
    sector_score: float
    macro_score: float
    commodity_score: float
    overall_context_score: float
    context_bias: str


@dataclass
class DynamicHorizonConfig:
    """Configuração de horizonte dinâmico baseado em regime de mercado."""
    evaluation_days: int      # Dias para avaliação do sinal
    position_size: float      # Tamanho da posição (0.0 a 1.0)
    stop_loss_pct: float      # Stop loss em %
    take_profit_pct: float    # Take profit em %
    regime: str               # "LATERAL", "TRENDING", "BREAKOUT"


@dataclass
class ProfessionalAnalysis:
    """Análise profissional completa."""
    ticker: str
    current_price: float
    analysis_date: datetime
    
    # Análises por camada
    trend: TrendAnalysis
    momentum: MomentumAnalysis
    volume_flow: VolumeFlowAnalysis
    options_sentiment: OptionsSentimentAnalysis
    macro_context: MacroContextAnalysis
    
    # Resultado final
    final_score: float
    direction: Direction
    confidence: float
    key_drivers: List[str]
    strategy_recommendation: str
    
    # Horizonte dinâmico
    dynamic_horizon: Optional[DynamicHorizonConfig] = None
    # Pesos efetivos usados na decisão (após redistribuições)
    effective_weights: Optional[Dict[str, float]] = None
    # Contadores de rejeição para diagnóstico
    gate_counters: Optional[Dict[str, int]] = None
    # Razões de rejeição (debug)
    rejection_reasons: Optional[List[str]] = None
    # Scores para debug
    raw_final_score: Optional[float] = None
    adjusted_final_score: Optional[float] = None
    # Gates e flags para debug
    gates_passed: Optional[bool] = None
    gates_relaxed: Optional[bool] = None
    # Debug específico para PUTs
    put_threshold_triggered: Optional[bool] = None
    put_rejected_low_conf: Optional[bool] = None
    bearish_penalty_value: Optional[float] = None
    bearish_penalty_trigger: Optional[str] = None
    # Meta-labeling de PUT
    put_meta_label_passed: Optional[bool] = None
    put_meta_label_reason: Optional[str] = None
    # Filtros seletivos para PUTs
    prefilter_reject: Optional[bool] = None
    prefilter_reason: Optional[str] = None


class ProfessionalAnalyzer:
    """Analisador profissional para Radar de Direção."""
    
    def __init__(self, client: Optional[OpLabClient] = None, horizon: str = "médio", 
                 decision_threshold: float = 0.20, layer_weights: Optional[Dict[str, float]] = None):
        self.client = client or OpLabClient()
        self.horizon = horizon
        self.params = self._get_horizon_parameters(horizon)
        self.decision_threshold = decision_threshold or 0.15  # Threshold mais baixo para gerar mais sinais
        
        # Pesos configuráveis das camadas - mais equilibrados
        self.layer_weights = layer_weights or {
            'trend': 0.30,
            'momentum': 0.25,
            'volume': 0.25,
            'sentiment': 0.15,
            'macro': 0.05
        }
    
    def _get_horizon_parameters(self, horizon: str) -> dict:
        """Define parâmetros adaptativos baseados no horizonte temporal."""
        horizon_configs = {
            "curto": {  # 1-3 meses
                "sma_short": 5,
                "sma_medium": 20,
                "sma_long": 50,
                "sma_trend": 100,
                "rsi_short": 7,
                "rsi_medium": 14,
                "rsi_long": 21,
                "macd_fast": 8,
                "macd_slow": 17,
                "macd_signal": 9,
                "atr_period": 10,
                "adx_period": 10
            },
            "médio": {  # 3-6 meses
                "sma_short": 10,
                "sma_medium": 50,
                "sma_long": 100,
                "sma_trend": 200,
                "rsi_short": 7,
                "rsi_medium": 14,
                "rsi_long": 21,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 14,
                "adx_period": 14
            },
            "longo": {  # 6+ meses
                "sma_short": 20,
                "sma_medium": 50,
                "sma_long": 100,
                "sma_trend": 200,
                "rsi_short": 14,
                "rsi_medium": 21,
                "rsi_long": 28,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "atr_period": 20,
                "adx_period": 20
            }
        }
        return horizon_configs.get(horizon, horizon_configs["médio"])
    
    def analyze_trend(self, price_data: pd.DataFrame) -> TrendAnalysis:
        """Camada 1: Análise de tendência estrutural."""
        close = price_data['close']
        
        # Validação: dados suficientes para análise
        if len(close) < max(self.params["sma_trend"], 50):
            raise ValueError(f"Dados insuficientes para análise de tendência. "
                           f"Necessário: {max(self.params['sma_trend'], 50)}, recebido: {len(close)}")
        
        # Médias móveis adaptativas baseadas no horizonte
        sma_short_series = close.rolling(self.params["sma_short"]).mean()
        sma_medium_series = close.rolling(self.params["sma_medium"]).mean()
        sma_long_series = close.rolling(self.params["sma_long"]).mean()
        sma_trend_series = close.rolling(self.params["sma_trend"]).mean()

        sma_short = sma_short_series.iloc[-1]
        sma_medium = sma_medium_series.iloc[-1]
        sma_long = sma_long_series.iloc[-1]
        sma_trend = sma_trend_series.iloc[-1]

        # Golden/Death Cross por cruzamento real (t-1 -> t)
        golden_cross = False
        death_cross = False
        if len(sma_medium_series) >= 2 and len(sma_trend_series) >= 2:
            prev_medium = sma_medium_series.iloc[-2]
            prev_trend = sma_trend_series.iloc[-2]
            golden_cross = bool(prev_medium <= prev_trend and sma_medium > sma_trend)
            death_cross = bool(prev_medium >= prev_trend and sma_medium < sma_trend)
        
        # Validação e imputação de dados críticos
        high = price_data.get('high')
        low = price_data.get('low')
        volume = price_data.get('volume')
        
        # Imputação robusta para high/low se ausentes
        if high is None or high.isna().all():
            high = close * 1.02  # Aproximação: +2% do close
        if low is None or low.isna().all():
            low = close * 0.98   # Aproximação: -2% do close
        if volume is None or volume.isna().all():
            volume = pd.Series([1.0] * len(close), index=close.index)
        
        # Validação: dados suficientes após imputação
        if len(high) != len(close) or len(low) != len(close):
            raise ValueError(f"Erro de consistência: high={len(high)}, low={len(low)}, close={len(close)}")
        
        # Regime de volatilidade (ATR)
        atr = self._calculate_atr(high, low, close, self.params["atr_period"])
        atr_current = atr.iloc[-1]
        atr_avg = atr.rolling(20).mean().iloc[-1]
        atr_ratio = atr_current / atr_avg if atr_avg > 0 else 1.0
        
        # Classifica regime de volatilidade
        if atr_ratio > 1.5:
            volatility_regime = "ALTA"
        elif atr_ratio < 0.7:
            volatility_regime = "BAIXA"
        else:
            volatility_regime = "NORMAL"
        
        # Score de tendência normalizado com foco em médio prazo
        trend_score = 0.0
        
        # Peso das médias móveis adaptativas (maior peso para médio/longo prazo)
        if sma_short > sma_medium:
            trend_score += 0.15  # Reduzido para dar menos peso ao curto prazo
        if sma_medium > sma_long:
            trend_score += 0.25  # Mantido - importante para médio prazo
        if sma_long > sma_trend:
            trend_score += 0.35  # Aumentado - fundamental para médio prazo
            
        # Golden/Death Cross com peso maior (SMA50/100 vs SMA200)
        if golden_cross:
            trend_score += 0.25  # Aumentado - sinal forte para médio prazo
        elif death_cross:
            trend_score -= 0.25
            
        # Análise adicional para médio prazo: distância das SMAs
        sma_distance_score = self._calculate_sma_distance_score(sma_medium, sma_long, sma_trend)
        trend_score += sma_distance_score
        
        # Normalização com tanh para manter em -1..+1
        trend_score = float(np.tanh(trend_score))
        
        # Força da tendência (ajustado para valores normalizados com tanh)
        if trend_score > 0.4:
            trend_strength = TrendStrength.STRONG_UP
        elif trend_score > 0.1:
            trend_strength = TrendStrength.WEAK_UP
        elif trend_score < -0.4:
            trend_strength = TrendStrength.STRONG_DOWN
        elif trend_score < -0.1:
            trend_strength = TrendStrength.WEAK_DOWN
        else:
            trend_strength = TrendStrength.LATERAL
        
        return TrendAnalysis(
            sma_10=sma_short,
            sma_50=sma_medium,
            sma_100=sma_long,
            sma_200=sma_trend,
            golden_cross=golden_cross,
            death_cross=death_cross,
            volatility_regime=volatility_regime,
            atr_ratio=atr_ratio,
            trend_score=trend_score,
            trend_strength=trend_strength
        )
    
    def analyze_momentum(self, price_data: pd.DataFrame) -> MomentumAnalysis:
        """Camada 2: Análise de momentum e força."""
        close = price_data['close']
        
        # Validação: dados suficientes
        if len(close) < max(self.params["rsi_long"], 30):
            raise ValueError(f"Dados insuficientes para análise de momentum. "
                           f"Necessário: {max(self.params['rsi_long'], 30)}, recebido: {len(close)}")
        
        # RSI múltiplos períodos adaptativos
        rsi_short = self._calculate_rsi(close, self.params["rsi_short"])
        rsi_medium = self._calculate_rsi(close, self.params["rsi_medium"])
        rsi_long = self._calculate_rsi(close, self.params["rsi_long"])
        
        # MACD adaptativo
        macd_line, macd_signal, macd_hist = self._calculate_macd_adaptive(close)
        
        # ADX adaptativo
        adx = self._calculate_adx_adaptive(price_data)
        
        # ROC (Rate of Change)
        roc_10 = ((close / close.shift(10)) - 1) * 100
        roc_20 = ((close / close.shift(20)) - 1) * 100
        
        # Score de momentum normalizado
        momentum_score = 0.0
        
        # RSI adaptativo
        rsi_short_val = float(rsi_short.iloc[-1])
        rsi_medium_val = float(rsi_medium.iloc[-1])
        rsi_long_val = float(rsi_long.iloc[-1])
        rsi_avg = (rsi_short_val + rsi_medium_val + rsi_long_val) / 3
        if rsi_avg > 60:
            momentum_score += 0.3
        elif rsi_avg < 40:
            momentum_score -= 0.3
        
        # Bollinger Bands para médio prazo
        bb_position, bb_squeeze = self._calculate_bollinger_bands(close)
        
        # Stochastic Oscillator
        stoch_k, stoch_d, stoch_signal = self._calculate_stochastic(price_data)
        
        # MACD
        macd_hist_val = float(macd_hist.iloc[-1])
        # Inclinação do MACD hist: diferença entre último e de 3 barras atrás
        if len(macd_hist) >= 4 and not pd.isna(macd_hist.iloc[-4]):
            macd_slope_val = float(macd_hist.iloc[-1] - macd_hist.iloc[-4])
        else:
            macd_slope_val = 0.0
        if macd_hist_val > 0:
            momentum_score += 0.3
        elif macd_hist_val < 0:
            momentum_score -= 0.3
        
        # ADX
        if adx > 25:
            momentum_score += 0.2 if macd_hist_val > 0 else -0.2
        elif adx < 15:
            momentum_score *= 0.5  # Reduz força se tendência fraca
        
        # ROC
        roc_10_val = float(roc_10.iloc[-1])
        if roc_10_val > 5:
            momentum_score += 0.2
        elif roc_10_val < -5:
            momentum_score -= 0.2
        
        # Bollinger Bands - indicador de reversão
        if bb_position > 0.8:  # Próximo da banda superior
            momentum_score -= 0.2  # Possível reversão de baixa
        elif bb_position < -0.8:  # Próximo da banda inferior
            momentum_score += 0.2  # Possível reversão de alta
        
        if bb_squeeze:  # Baixa volatilidade - possível breakout
            momentum_score *= 0.8  # Reduz confiança até definir direção
        
        # Stochastic - identificação de extremos
        if stoch_signal == "OVERSOLD":
            momentum_score += 0.15
        elif stoch_signal == "OVERBOUGHT":
            momentum_score -= 0.15

        # EMAs rápidas: EMA9 acima de EMA21 reforça momentum positivo
        ema_9 = float(close.ewm(span=9).mean().iloc[-1])
        ema_21 = float(close.ewm(span=21).mean().iloc[-1])
        if ema_9 > ema_21:
            momentum_score += 0.1
        else:
            momentum_score -= 0.1

        # Money Flow Index (MFI 14) simplificado - usa dados já validados
        high = price_data.get('high', close * 1.01)
        low = price_data.get('low', close * 0.99)
        volume = price_data.get('volume', pd.Series([1.0] * len(close), index=close.index))
        tp = (high + low + close) / 3
        pmf = ((tp > tp.shift(1)) * (tp * volume)).fillna(0)
        nmf = ((tp < tp.shift(1)) * (tp * volume)).fillna(0)
        mfr = (pmf.rolling(14).sum()) / (nmf.rolling(14).sum().replace(0, 1e-10))
        mfi = 100 - (100 / (1 + mfr))
        mfi_val = float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0
        # MFI > 60 reforça, < 40 penaliza
        if mfi_val > 60:
            momentum_score += 0.1
        elif mfi_val < 40:
            momentum_score -= 0.1
        
        # Normalização com tanh para manter em -1..+1
        momentum_score = float(np.tanh(momentum_score))
        
        # Força do momentum (ajustado para valores normalizados com tanh)
        if momentum_score > 0.4:
            momentum_strength = "FORTE POSITIVO"
        elif momentum_score > 0.1:
            momentum_strength = "POSITIVO"
        elif momentum_score < -0.4:
            momentum_strength = "FORTE NEGATIVO"
        elif momentum_score < -0.1:
            momentum_strength = "NEGATIVO"
        else:
            momentum_strength = "NEUTRO"
        
        return MomentumAnalysis(
            rsi_7=rsi_short_val,
            rsi_14=rsi_medium_val,
            rsi_21=rsi_long_val,
            macd_histogram=macd_hist_val,
            macd_slope=macd_slope_val,
            macd_signal=float(macd_signal.iloc[-1]),
            adx=adx,
            roc_10=roc_10_val,
            roc_20=float(roc_20.iloc[-1]),
            bb_position=bb_position,
            bb_squeeze=bb_squeeze,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            stoch_signal=stoch_signal,
            ema_9=ema_9,
            ema_21=ema_21,
            mfi_14=mfi_val,
            momentum_score=momentum_score,
            momentum_strength=momentum_strength
        )
    
    def analyze_volume_flow(self, price_data: pd.DataFrame) -> VolumeFlowAnalysis:
        """Camada 3: Análise de volume e fluxo."""
        close = price_data['close']
        
        # Validação: dados suficientes
        if len(close) < 30:
            raise ValueError(f"Dados insuficientes para análise de volume. "
                           f"Necessário: 30, recebido: {len(close)}")
        
        volume = price_data.get('volume', pd.Series([1.0] * len(close), index=close.index))
        
        # Validação de volume
        if volume is None or volume.isna().all():
            volume = pd.Series([1.0] * len(close), index=close.index)
        
        # Volume ratios (compara médias recentes vs históricas)
        volume_5d = volume.rolling(5).mean()
        volume_20d = volume.rolling(20).mean()
        # Evita erro de ambiguidade com comparações de Series
        vol_5d_val = float(volume_5d.iloc[-1])
        vol_20d_val = float(volume_20d.iloc[-1])
        vol_val = float(volume.iloc[-1])
        
        # Compara volume atual vs médias e médias entre si
        volume_ratio_5d = vol_5d_val / vol_20d_val if vol_20d_val > 0 else 1.0  # Média 5d vs 20d
        volume_ratio_20d = vol_val / vol_20d_val if vol_20d_val > 0 else 1.0     # Volume atual vs 20d
        
        # OBV
        obv = self._calculate_obv(close, volume)
        # Evita erro de ambiguidade com comparações de Series
        obv_curr = float(obv.iloc[-1])
        obv_prev = float(obv.iloc[-10])
        obv_trend = (obv_curr - obv_prev) / obv_prev if obv_prev != 0 else 0
        
        # Accumulation/Distribution - usa dados validados
        ad = self._calculate_accumulation_distribution(price_data)
        # Evita erro de ambiguidade com comparações de Series
        ad_curr = float(ad.iloc[-1])
        ad_prev = float(ad.iloc[-10])
        ad_trend = (ad_curr - ad_prev) / abs(ad_prev) if ad_prev != 0 else 0
        
        # Score de volume
        volume_score = 0.0
        
        if volume_ratio_5d > 1.5:
            volume_score += 0.4
        elif volume_ratio_5d > 1.2:
            volume_score += 0.2
        elif volume_ratio_5d < 0.8:
            volume_score -= 0.2
        
        if obv_trend > 0.05:
            volume_score += 0.3
        elif obv_trend < -0.05:
            volume_score -= 0.3
        
        if ad_trend > 0.02:
            volume_score += 0.3
        elif ad_trend < -0.02:
            volume_score -= 0.3
        
        # Normalização com tanh
        volume_score = float(np.tanh(volume_score))
        
        # Direção do fluxo (ajustado para valores normalizados com tanh)
        if volume_score > 0.2:
            flow_direction = "ENTRADA"
        elif volume_score < -0.2:
            flow_direction = "SAÍDA"
        else:
            flow_direction = "NEUTRO"
        
        return VolumeFlowAnalysis(
            volume_ratio_5d=volume_ratio_5d,
            volume_ratio_20d=volume_ratio_20d,
            obv_trend=obv_trend,
            accumulation_distribution=ad_trend,
            volume_score=volume_score,
            flow_direction=flow_direction
        )
    
    def analyze_options_sentiment(self, ticker: str, price_data: pd.DataFrame) -> OptionsSentimentAnalysis:
        """Camada 4: Análise de sentimento via opções."""
        try:
            # Busca dados de opções
            option_chain = self.client.get_option_chain(ticker)
            
            if option_chain.empty:
                return OptionsSentimentAnalysis(
                    put_call_ratio=1.0,
                    volatility_skew=0.0,
                    call_volume_ratio=0.5,
                    put_volume_ratio=0.5,
                    sentiment_score=0.0,
                    market_bias="NEUTRAL"
                )
            
            # Separa CALLs e PUTs
            calls = option_chain[option_chain['option_type'] == 'CALL']
            puts = option_chain[option_chain['option_type'] == 'PUT']
            
            # Volume e Open Interest - fallbacks robustos
            # Volume: usa 0 se não disponível, não filtra
            call_volume = calls['volume'].fillna(0).sum() if 'volume' in calls.columns else 0
            put_volume = puts['volume'].fillna(0).sum() if 'volume' in puts.columns else 0
            total_volume = call_volume + put_volume
            
            # Open Interest - fallback para volume se não disponível
            if 'open_interest' in calls.columns and 'open_interest' in puts.columns:
                call_oi = calls['open_interest'].fillna(0).sum()
                put_oi = puts['open_interest'].fillna(0).sum()
            else:
                # Fallback: usa volume como proxy para OI
                call_oi = call_volume
                put_oi = put_volume
            total_oi = call_oi + put_oi
            
            # Calcula ratios com fallbacks robustos
            if total_volume > 0:
                call_volume_ratio = call_volume / total_volume
                put_volume_ratio = put_volume / total_volume
            else:
                # Fallback: usa quantidade de contratos
                call_count = len(calls)
                put_count = len(puts)
                total_count = call_count + put_count
                if total_count > 0:
                    call_volume_ratio = call_count / total_count
                    put_volume_ratio = put_count / total_count
                else:
                    call_volume_ratio = 0.5
                    put_volume_ratio = 0.5
            
            # OI Ratio (fluxo novo de posicionamento)
            if total_oi > 0:
                call_oi_ratio = call_oi / total_oi
                put_oi_ratio = put_oi / total_oi
            else:
                call_oi_ratio = call_volume_ratio  # Usa volume como proxy
                put_oi_ratio = put_volume_ratio
            
            # Put/Call Ratio
            put_call_ratio = put_volume / call_volume if call_volume > 0 else 1.0
            
            # Volatility Skew com fallbacks robustos
            current_price = float(price_data['close'].iloc[-1])
            lower_bound = current_price * 0.95
            upper_bound = current_price * 1.05
            
            # Evita erro de ambiguidade com comparações de Series
            atm_calls = calls[(calls['strike'] >= lower_bound) & 
                            (calls['strike'] <= upper_bound)]
            atm_puts = puts[(puts['strike'] >= lower_bound) & 
                           (puts['strike'] <= upper_bound)]
            
            # Calcula volatility skew com fallbacks
            volatility_skew = 0.0
            if not atm_calls.empty and not atm_puts.empty:
                # Prioridade 1: IV se disponível
                if 'implied_volatility' in calls.columns and 'implied_volatility' in puts.columns:
                    call_iv = atm_calls['implied_volatility'].fillna(0).mean()
                    put_iv = atm_puts['implied_volatility'].fillna(0).mean()
                    if call_iv > 0 and put_iv > 0:
                        volatility_skew = put_iv - call_iv
                # Prioridade 2: preços de opções
                elif 'last' in calls.columns or 'bid' in calls.columns:
                    call_price = atm_calls.get('last', atm_calls.get('bid', 0)).fillna(0).mean()
                    put_price = atm_puts.get('last', atm_puts.get('bid', 0)).fillna(0).mean()
                    if call_price > 0 and put_price > 0:
                        call_premium_ratio = call_price / current_price
                        put_premium_ratio = put_price / current_price
                        volatility_skew = put_premium_ratio - call_premium_ratio
            
            # Score de sentimento aprimorado
            sentiment_score = 0.0
            
            # Put/Call Ratio (volume)
            if put_call_ratio > 1.2:  # Mais puts = bearish
                sentiment_score -= 0.3
            elif put_call_ratio < 0.8:  # Mais calls = bullish
                sentiment_score += 0.3
            
            # Volume ratio
            if call_volume_ratio > 0.6:
                sentiment_score += 0.2
            elif put_volume_ratio > 0.6:
                sentiment_score -= 0.2
            
            # Open Interest ratio (fluxo novo)
            if call_oi_ratio > 0.6:  # Novo posicionamento em calls
                sentiment_score += 0.2
            elif put_oi_ratio > 0.6:  # Novo posicionamento em puts
                sentiment_score -= 0.2
            
            # Volatility skew
            if volatility_skew > 0.05:  # Puts mais caros = bearish
                sentiment_score -= 0.2
            elif volatility_skew < -0.05:  # Calls mais caros = bullish
                sentiment_score += 0.2
            
            # Normalização com tanh
            sentiment_score = float(np.tanh(sentiment_score))
            
            # Bias do mercado
            if sentiment_score > 0.3:
                market_bias = "BULLISH"
            elif sentiment_score < -0.3:
                market_bias = "BEARISH"
            else:
                market_bias = "NEUTRAL"
            
            return OptionsSentimentAnalysis(
                put_call_ratio=put_call_ratio,
                volatility_skew=volatility_skew,
                call_volume_ratio=call_volume_ratio,
                put_volume_ratio=put_volume_ratio,
                sentiment_score=sentiment_score,
                market_bias=market_bias
            )
            
        except Exception as e:
            print(f"Erro na análise de sentimento: {e}")
            return OptionsSentimentAnalysis(
                put_call_ratio=1.0,
                volatility_skew=0.0,
                call_volume_ratio=0.5,
                put_volume_ratio=0.5,
                sentiment_score=0.0,
                market_bias="NEUTRAL"
            )
    
    def analyze_macro_context(self, ticker: str, price_data: pd.DataFrame) -> MacroContextAnalysis:
        """Camada 5: Análise de contexto macro e setorial."""
        # Análise setorial simplificada baseada no ticker
        sector_scores = {
            'PETR': 0.2,  # Petróleo - positivo por preços altos
            'VALE': 0.1,  # Minério - moderadamente positivo
            'ITUB': -0.1, # Bancos - pressão de juros
            'BBAS': -0.1, # Bancos - pressão de juros
            'BBDC': -0.1, # Bancos - pressão de juros
            'WEGE': 0.0,  # Industrial - neutro
            'MGLU': -0.2, # Varejo - pressão consumo
        }
        
        # Identifica setor pelo ticker
        sector_score = 0.0
        for sector, score in sector_scores.items():
            if sector in ticker:
                sector_score = score
                break
        
        # Contexto macro simplificado (pode ser expandido com APIs reais)
        macro_score = 0.0  # Neutro por enquanto
        
        # Commodities (simplificado)
        commodity_score = 0.0
        if 'PETR' in ticker:
            commodity_score = 0.2  # Petróleo em alta
        elif 'VALE' in ticker:
            commodity_score = 0.1  # Minério moderado
        
        # Score geral
        overall_context_score = (sector_score * 0.5 + macro_score * 0.3 + commodity_score * 0.2)
        
        # Bias do contexto
        if overall_context_score > 0.1:
            context_bias = "POSITIVO"
        elif overall_context_score < -0.1:
            context_bias = "NEGATIVO"
        else:
            context_bias = "NEUTRAL"
        
        return MacroContextAnalysis(
            sector_score=sector_score,
            macro_score=macro_score,
            commodity_score=commodity_score,
            overall_context_score=overall_context_score,
            context_bias=context_bias
        )
    
    def calculate_final_decision(self, analysis: 'ProfessionalAnalysis') -> Tuple[Direction, float, List[str]]:
        """Camada 6: Modelo de decisão com gates, pesos dinâmicos e threshold adaptativo."""
        adx_value = float(analysis.momentum.adx)

        # Pesos dinâmicos por regime
        dyn_weights = self._dynamic_weights(adx_value)
        # Se sentimento é neutro/ausente, redistribui seu peso entre trend/momentum/volume
        eff_weights = dict(dyn_weights)
        if float(analysis.options_sentiment.sentiment_score) == 0.0:
            sentiment_w = eff_weights.get('sentiment', 0.0)
            if sentiment_w > 0:
                # soma base dos três alvos
                base_sum = eff_weights['trend'] + eff_weights['momentum'] + eff_weights['volume']
                if base_sum > 0:
                    eff_weights['trend'] += sentiment_w * (eff_weights['trend'] / base_sum)
                    eff_weights['momentum'] += sentiment_w * (eff_weights['momentum'] / base_sum)
                    eff_weights['volume'] += sentiment_w * (eff_weights['volume'] / base_sum)
                    eff_weights['sentiment'] = 0.0  # removido
                # mantém 'macro' inalterado
        analysis.effective_weights = eff_weights

        # Score final ponderado (antes da penalidade)
        raw_final_score = (
            analysis.trend.trend_score * eff_weights['trend'] +
            analysis.momentum.momentum_score * eff_weights['momentum'] +
            analysis.volume_flow.volume_score * eff_weights['volume'] +
            analysis.options_sentiment.sentiment_score * eff_weights['sentiment'] +
            analysis.macro_context.overall_context_score * eff_weights['macro']
        )

        # Normaliza para -1 a +1
        raw_final_score = max(-1.0, min(1.0, raw_final_score))
        
        # Penalidade bearish: se pelo menos 1 condição bearish for verdadeira
        final_score = raw_final_score
        bearish_signals = 0
        bearish_triggers = []
        
        # Drivers bearish originais
        if analysis.momentum.ema_9 < analysis.momentum.ema_21:
            bearish_signals += 1
            bearish_triggers.append("ema9<ema21")
        if analysis.momentum.macd_slope < 0:
            bearish_signals += 1
            bearish_triggers.append("macd_slope<0")
        if analysis.volume_flow.obv_trend < 0:
            bearish_signals += 1
            bearish_triggers.append("obv_trend<0")
        
        # Novos drivers bearish
        # 1. RSI < 45
        rsi_value = analysis.momentum.rsi_14
        if rsi_value < 45:
            bearish_signals += 1
            bearish_triggers.append(f"RSI<45({rsi_value:.1f})")
            analysis.rejection_reasons.append(f"bearish_trigger: RSI<45 (valor={rsi_value:.2f})")
        
        # 2. Padrões de candle de reversão com volume
        # Busca dados históricos para análise de padrões
        try:
            # Pega os últimos 3 candles para análise de padrões
            recent_data = self._get_recent_candles_for_patterns(analysis.ticker, analysis.analysis_date)
            if recent_data is not None and len(recent_data) >= 3:
                current_candle = recent_data.iloc[-1]
                prev_candle = recent_data.iloc[-2]
                prev2_candle = recent_data.iloc[-3]
                
                # Volume médio dos últimos 20 dias
                vol_avg_20 = recent_data['volume'].rolling(20).mean().iloc[-1] if len(recent_data) >= 20 else current_candle['volume']
                vol_ratio = current_candle['volume'] / vol_avg_20 if vol_avg_20 > 0 else 1.0
                
                # Engolfo de baixa (bearish engulfing)
                if (prev_candle['close'] > prev_candle['open'] and  # Candle anterior de alta
                    current_candle['close'] < current_candle['open'] and  # Candle atual de baixa
                    current_candle['open'] > prev_candle['close'] and  # Abertura atual > fechamento anterior
                    current_candle['close'] < prev_candle['open'] and  # Fechamento atual < abertura anterior
                    vol_ratio > 1.5):  # Volume > 1.5x média
                    bearish_signals += 1
                    bearish_triggers.append("engolfo_baixa")
                    analysis.rejection_reasons.append(f"bearish_trigger: engolfo_baixa (volume={vol_ratio:.2f})")
                
                # Estrela cadente (shooting star)
                if (current_candle['close'] < current_candle['open'] and  # Candle de baixa
                    (current_candle['high'] - max(current_candle['open'], current_candle['close'])) > 
                    2 * (max(current_candle['open'], current_candle['close']) - current_candle['low']) and  # Sombra superior > 2x corpo
                    vol_ratio > 1.5):  # Volume > 1.5x média
                    bearish_signals += 1
                    bearish_triggers.append("shooting_star")
                    analysis.rejection_reasons.append(f"bearish_trigger: shooting_star (volume={vol_ratio:.2f})")
                
                # 3. Sequência de queda (2 dias consecutivos de fechamento em queda e volume crescente)
                if (len(recent_data) >= 2 and
                    recent_data.iloc[-1]['close'] < recent_data.iloc[-2]['close'] and  # Fechamento atual < fechamento anterior
                    recent_data.iloc[-2]['close'] < recent_data.iloc[-3]['close'] and  # Fechamento anterior < fechamento anterior-2
                    recent_data.iloc[-1]['volume'] > recent_data.iloc[-2]['volume']):  # Volume crescente
                    bearish_signals += 1
                    vol_dia1 = recent_data.iloc[-2]['volume']
                    vol_dia2 = recent_data.iloc[-1]['volume']
                    bearish_triggers.append("2dias_queda_volume")
                    analysis.rejection_reasons.append(f"bearish_trigger: 2dias_queda_volume (vol_dia1={vol_dia1:.0f}, vol_dia2={vol_dia2:.0f})")
                    
        except Exception as e:
            # Se houver erro na análise de padrões, continua sem os novos drivers
            pass
            
        if bearish_signals >= 1:  # Pelo menos 1 sinal de baixa
            # Penalidade bearish escalada pelo número de triggers ativos
            if bearish_signals == 1:
                base_penalty = -0.05
            elif bearish_signals == 2:
                base_penalty = -0.10
            elif bearish_signals == 3:
                base_penalty = -0.15
            else:  # 4+ triggers
                base_penalty = -0.20
            
            # Ajusta penalidade baseada no ADX
            if adx_value >= 20 and adx_value < 25:
                penalty = base_penalty + (-0.02)  # -0.07 para 1 trigger, -0.12 para 2, etc.
            elif adx_value >= 25 and adx_value < 35:
                penalty = base_penalty + (-0.05)  # -0.10 para 1 trigger, -0.15 para 2, etc.
            elif adx_value >= 35:
                penalty = base_penalty + (-0.10)  # -0.15 para 1 trigger, -0.20 para 2, etc.
            else:
                penalty = base_penalty  # ADX < 20, usa penalidade base
            
            final_score += penalty  # penalty já é negativo
            trigger_str = ", ".join(bearish_triggers)
            analysis.rejection_reasons.append(f"bearish_penalty_applied: {penalty:.2f} (ADX={adx_value:.0f}, triggers={trigger_str})")
            analysis.bearish_penalty_value = penalty
            analysis.bearish_penalty_trigger = trigger_str
        
        # Normaliza novamente após penalidade
        final_score = max(-1.0, min(1.0, final_score))

        # Gatilhos alinhados: exigem ADX >= 18 e concordância EMA/MACD
        trigger_call = (
            adx_value >= 18 and 
            analysis.momentum.ema_9 > analysis.momentum.ema_21 and 
            analysis.momentum.macd_slope > 0
        )
        trigger_put = (
            adx_value >= 18 and 
            analysis.momentum.ema_9 < analysis.momentum.ema_21 and 
            analysis.momentum.macd_slope < 0
        )

        # Thresholds ajustados para forçar PUTs
        thr_base = self._adaptive_threshold(adx_value, base=self.decision_threshold)
        thr_call = thr_base  # Ex: 0.15
        
        # thr_put ajustado conforme especificação
        if adx_value >= 25 and analysis.trend.death_cross:
            thr_put = 0.00  # Se adx>=25 e ema200 descendente, thr_put = 0.00
        else:
            thr_put = 0.02  # thr_put padrão = 0.02

        passes_gates = self._passes_quality_gates(analysis, final_score, thr_put)

        # Regras de decisão com forçar PUTs muito negativos
        if final_score <= -0.10:  # Score muito negativo - força PUT
            direction = Direction.PUT
            analysis.rejection_reasons.append("forced_put_very_negative")
        elif passes_gates:
            if final_score >= thr_call:
                direction = Direction.CALL
            elif final_score <= -thr_put:
                direction = Direction.PUT
                analysis.put_threshold_triggered = True
                print(f"put_threshold_triggered: score={final_score:.4f}, thr_put={thr_put:.4f}")
                
                # Prefilter seletivo para PUTs
                prefilter_passed = self._put_prefilter(analysis)
                if not prefilter_passed:
                    # Se falhou no prefilter, converte para NEUTRAL
                    direction = Direction.NEUTRAL
                    analysis.rejection_reasons.append("put_prefilter_failed")
                    print(f"PUT rejeitado por prefilter: {analysis.prefilter_reason}")
                else:
                    # Meta-labeling de PUT: filtro extra no D+1
                    meta_label_passed = self._meta_label_put_filter(analysis)
                    analysis.put_meta_label_passed = meta_label_passed
                    
                    if not meta_label_passed:
                        # Se falhou no meta-labeling, converte para NEUTRAL
                        direction = Direction.NEUTRAL
                        analysis.rejection_reasons.append("put_meta_label_failed")
                        analysis.put_meta_label_reason = "failed"
                        print(f"PUT rejeitado por meta-labeling: {analysis.rejection_reasons[-1]}")
                    else:
                        analysis.put_meta_label_reason = "passed"
            else:
                # Se não passou pelo score, mas há gatilho forte, reduz levemente o threshold
                if trigger_call and final_score >= (thr_call * 0.9):
                    direction = Direction.CALL
                elif trigger_put and final_score <= -(thr_put * 0.9):
                    direction = Direction.PUT
                else:
                    direction = Direction.NEUTRAL
        else:
            # Gates falharam: só permitem CALL/PUT se gatilho estiver presente e score próximo do corte
            if trigger_call and final_score >= (thr_call * 0.95):
                direction = Direction.CALL
            elif trigger_put and final_score <= -(thr_put * 0.95):
                direction = Direction.PUT
            else:
                direction = Direction.NEUTRAL
                analysis.rejection_reasons.append("gate_rejected")

        # Confiança (fallback calibrado logístico)
        confidence = self._calculate_calibrated_confidence(final_score, direction)
        # Rejeição de baixa confiança (mais rígida no curto prazo)
        min_conf = 70 if self.horizon == 'curto' else 55
        if confidence < min_conf:
            # Para PUTs, aceita confiança mais baixa (50)
            if direction == Direction.PUT and confidence >= 50:
                pass  # Mantém PUT com confiança >= 50
            else:
                if direction == Direction.PUT:
                    analysis.put_rejected_low_conf = True
                    print(f"put_rejected_low_conf: conf={confidence:.1f}")
                direction = Direction.NEUTRAL

        # Meta-label de risco: descarta operações com alto risco estrutural
        if direction != Direction.NEUTRAL:
            if not self._meta_label_filter(analysis, direction, confidence):
                direction = Direction.NEUTRAL
                analysis.rejection_reasons.append("meta_rejected")

        # Registro de rejeição por trigger (quando nenhum trigger ajuda e ficou neutro)
        if direction == Direction.NEUTRAL and (not trigger_call and not trigger_put):
            analysis.rejection_reasons.append("trigger_rejected")

        # Armazena scores e flags para debug
        analysis.raw_final_score = raw_final_score
        analysis.adjusted_final_score = final_score
        analysis.gates_passed = passes_gates
        analysis.gates_relaxed = False  # Será definido em _passes_quality_gates se aplicável
        # Inicializa campos de debug PUT
        analysis.put_threshold_triggered = False
        analysis.put_rejected_low_conf = False
        analysis.bearish_penalty_value = 0.0
        analysis.bearish_penalty_trigger = None
        analysis.put_meta_label_passed = None
        analysis.put_meta_label_reason = None
        analysis.prefilter_reject = None
        analysis.prefilter_reason = None
        
        # Log JSON para explicabilidade (usa pesos efetivos)
        self._log_decision_json(analysis, final_score, eff_weights, thr_call, thr_put, 
                               trigger_call, trigger_put, passes_gates, direction, confidence)
        
        # Identifica drivers principais
        drivers = []
        
        # Driver de tendência
        if analysis.trend.trend_strength in [TrendStrength.STRONG_UP, TrendStrength.STRONG_DOWN]:
            drivers.append(f"Tendência {analysis.trend.trend_strength.value.lower()}")
        
        # Driver de momentum
        if analysis.momentum.momentum_strength in ["FORTE POSITIVO", "FORTE NEGATIVO"]:
            drivers.append(f"Momentum {analysis.momentum.momentum_strength.lower()}")
        
        # Driver de volume
        if analysis.volume_flow.flow_direction != "NEUTRO":
            drivers.append(f"Fluxo de {analysis.volume_flow.flow_direction.lower()}")
        
        # Driver de sentimento
        if analysis.options_sentiment.market_bias != "NEUTRAL":
            drivers.append(f"Sentimento {analysis.options_sentiment.market_bias.lower()}")
        
        # Driver macro
        if analysis.macro_context.context_bias != "NEUTRO":
            drivers.append(f"Contexto {analysis.macro_context.context_bias.lower()}")
        
        return direction, confidence, drivers

    def _get_recent_candles_for_patterns(self, ticker: str, analysis_date: datetime) -> Optional[pd.DataFrame]:
        """
        Busca dados históricos recentes para análise de padrões de candle.
        Retorna DataFrame com OHLCV dos últimos 25 dias.
        """
        try:
            from src.core.data.data import get_historical_data
            
            # Busca dados dos últimos 30 dias para garantir pelo menos 25 dias úteis
            start_date = analysis_date - timedelta(days=40)
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = analysis_date.strftime('%Y-%m-%d')
            
            df = get_historical_data(ticker, start_str, end_str)
            if df.empty or len(df) < 3:
                return None
            
            # Filtra até a data de análise (não inclui dados futuros)
            df = df[df['date'] <= analysis_date]
            
            # Retorna os últimos 25 dias ou todos se menos que 25
            return df.tail(25) if len(df) >= 25 else df
            
        except Exception as e:
            # Se houver erro, retorna None
            return None

    def _log_decision_json(self, analysis: 'ProfessionalAnalysis', final_score: float, 
                          dyn_weights: Dict[str, float], thr_base: float, thr_put: float,
                          trigger_call: bool, trigger_put: bool, passes_gates: bool,
                          direction: Direction, confidence: float) -> None:
        """Gera log JSON para explicabilidade da decisão."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "ticker": analysis.ticker,
            "horizon": self.horizon,
            "decision": {
                "direction": direction.value,
                "confidence": round(confidence, 2),
                "final_score": round(final_score, 4)
            },
            "thresholds": {
                "base": round(thr_base, 4),
                "put": round(thr_put, 4),
                "decision_threshold": self.decision_threshold
            },
            "triggers": {
                "call": trigger_call,
                "put": trigger_put,
                "passes_gates": passes_gates
            },
            "layer_scores": {
                "trend": round(analysis.trend.trend_score, 4),
                "momentum": round(analysis.momentum.momentum_score, 4),
                "volume": round(analysis.volume_flow.volume_score, 4),
                "sentiment": round(analysis.options_sentiment.sentiment_score, 4),
                "macro": round(analysis.macro_context.overall_context_score, 4)
            },
            "dynamic_weights": {k: round(v, 3) for k, v in dyn_weights.items()},
            "key_indicators": {
                "adx": round(analysis.momentum.adx, 2),
                "ema_9": round(analysis.momentum.ema_9, 2),
                "ema_21": round(analysis.momentum.ema_21, 2),
                "macd_slope": round(analysis.momentum.macd_slope, 4),
                "mfi_14": round(analysis.momentum.mfi_14, 2),
                "obv_trend": round(analysis.volume_flow.obv_trend, 4),
                "ad_trend": round(analysis.volume_flow.accumulation_distribution, 4)
            }
        }
        
        # Log apenas para debug (pode ser removido em produção)
        if self.horizon == 'curto':  # Log apenas para curto prazo para não poluir
            print(f"🔍 DECISION_LOG: {json.dumps(log_data, indent=2)}")

    def _meta_label_filter(self, analysis: 'ProfessionalAnalysis', direction: Direction, confidence: float) -> bool:
        """Filtro de risco (meta-label) para decidir operar vs descartar.
        Regras simples baseadas em regime, fluxo e extremos de osciladores.
        Retorna True se deve operar; False se deve descartar (virar NEUTRAL).
        """
        
        # Exceção: scores muito altos (>= 0.35) sempre passam no meta-label
        if abs(analysis.final_score) >= 0.35:
            return True
        adx = float(analysis.momentum.adx)
        mfi = float(analysis.momentum.mfi_14)
        bb_pos = float(analysis.momentum.bb_position)
        squeeze = bool(analysis.momentum.bb_squeeze)
        vol_score = float(analysis.volume_flow.volume_score)
        obv_trend = float(analysis.volume_flow.obv_trend)
        ad_trend = float(analysis.volume_flow.accumulation_distribution)
        ema9 = float(analysis.momentum.ema_9)
        ema21 = float(analysis.momentum.ema_21)
        macd_slope = float(analysis.momentum.macd_slope)
        stoch_signal = analysis.momentum.stoch_signal if hasattr(analysis.momentum, 'stoch_signal') else "NEUTRAL"

        # Permissão extra em squeeze se houver trigger forte e confiança suficiente
        # Obs: triggers são reavaliados aqui por segurança
        trigger_call = (adx >= 18 and ema9 > ema21 and macd_slope > 0)
        trigger_put = (adx >= 18 and ema9 < ema21 and macd_slope < 0)

        if squeeze and abs(vol_score) < 0.1:
            if not ((trigger_call or trigger_put) and confidence >= 65):
                analysis.gate_counters['meta_squeeze_vol'] = analysis.gate_counters.get('meta_squeeze_vol', 0) + 1
                return False

        # Regime LATERAL (reversão)
        if adx < 18:
            if direction == Direction.PUT:
                # Aceita PUT em reversão de topo - regra relaxada
                if not (((mfi > 65) or (stoch_signal == "OVERBOUGHT")) and bb_pos > 0.4):  # bb_position > 0.4 (antes exigia >0.6)
                    analysis.gate_counters['meta_lateral'] = analysis.gate_counters.get('meta_lateral', 0) + 1
                    return False
            elif direction == Direction.CALL:
                # Aceita CALL em reversão de fundo - condições muito relaxadas
                if not (((mfi < 60) or (stoch_signal == "OVERSOLD")) and bb_pos < 0.2):
                    analysis.gate_counters['meta_lateral'] = analysis.gate_counters.get('meta_lateral', 0) + 1
                    return False
            return True

        # Regime TENDÊNCIA (continuação)
        else:
            # Regras 2-de-4 (relaxadas de 3-de-4)
            if direction == Direction.PUT:
                conds = [ema9 < ema21, macd_slope < 0, obv_trend < 0, mfi < 50]
                if sum(1 for c in conds if c) < 2:
                    analysis.gate_counters['meta_trend'] = analysis.gate_counters.get('meta_trend', 0) + 1
                    return False
                return True
            elif direction == Direction.CALL:
                conds = [ema9 > ema21, macd_slope > 0, obv_trend > 0, mfi > 50]
                if sum(1 for c in conds if c) < 2:
                    analysis.gate_counters['meta_trend'] = analysis.gate_counters.get('meta_trend', 0) + 1
                    return False
                return True
            else:
                return True

    def _dynamic_weights(self, adx: float) -> Dict[str, float]:
        if adx >= 25:
            return {'trend': 0.35, 'momentum': 0.30, 'volume': 0.20, 'sentiment': 0.10, 'macro': 0.05}
        if adx <= 15:
            return {'trend': 0.15, 'momentum': 0.35, 'volume': 0.25, 'sentiment': 0.20, 'macro': 0.05}
        return {'trend': 0.25, 'momentum': 0.30, 'volume': 0.25, 'sentiment': 0.15, 'macro': 0.05}

    def _put_prefilter(self, analysis: 'ProfessionalAnalysis') -> bool:
        """
        Filtros seletivos para PUTs antes da decisão final.
        Retorna True se o PUT deve ser aceito, False se deve ser rejeitado.
        """
        try:
            # 1. Exigir regime macro de baixa
            if analysis.current_price >= analysis.trend.ema_200:
                analysis.rejection_reasons.append("prefilter_reject: price_above_ema200")
                analysis.prefilter_reject = True
                analysis.prefilter_reason = "price_above_ema200"
                return False
            
            # Verificar EMA200 slope (EMA200[t] < EMA200[t-5])
            try:
                recent_data = self._get_recent_candles_for_patterns(analysis.ticker, analysis.analysis_date)
                if recent_data is not None and len(recent_data) >= 10:
                    ema200_current = analysis.trend.ema_200
                    ema200_5_days_ago = recent_data['close'].tail(10).rolling(200).mean().iloc[-6] if len(recent_data) >= 6 else ema200_current
                    
                    if ema200_current >= ema200_5_days_ago:
                        analysis.rejection_reasons.append("prefilter_reject: ema200_not_descending")
                        analysis.prefilter_reject = True
                        analysis.prefilter_reason = "ema200_not_descending"
                        return False
            except:
                pass  # Se houver erro, continua sem este filtro
            
            # 2. Exigir 2 de 4 condições bearish OU death_cross ativo
            bearish_conditions = 0
            death_cross_active = analysis.trend.death_cross
            
            # Condição 1: Candle vermelho e volume > 1.3 * média20
            try:
                if recent_data is not None and len(recent_data) >= 20:
                    current_close = analysis.current_price
                    current_open = recent_data['close'].iloc[-2]  # Aproximação
                    current_volume = recent_data['volume'].iloc[-1]
                    avg_volume_20 = recent_data['volume'].tail(20).mean()
                    
                    if current_close < current_open and current_volume > 1.3 * avg_volume_20:
                        bearish_conditions += 1
            except:
                pass
            
            # Condição 2: MACD hist < 0 e menor que há 3 barras
            try:
                if recent_data is not None and len(recent_data) >= 5:
                    current_macd_hist = analysis.momentum.macd_histogram
                    macd_3_bars_ago = recent_data['close'].tail(5).rolling(12).mean().iloc[-4] - recent_data['close'].tail(5).rolling(26).mean().iloc[-4]
                    
                    if current_macd_hist < 0 and current_macd_hist < macd_3_bars_ago:
                        bearish_conditions += 1
            except:
                pass
            
            # Condição 3: RSI14 < 45 ou cruzando abaixo de 50
            rsi_14 = analysis.momentum.rsi_14
            if rsi_14 < 45:
                bearish_conditions += 1
            elif rsi_14 < 50:
                # Verificar se está cruzando abaixo de 50
                try:
                    if recent_data is not None and len(recent_data) >= 2:
                        rsi_prev = recent_data['close'].tail(2).pct_change().rolling(14).apply(lambda x: 100 - (100 / (1 + x.mean() / (1 - x.mean())))).iloc[-2]
                        if rsi_prev >= 50 and rsi_14 < 50:
                            bearish_conditions += 1
                except:
                    pass
            
            # Condição 4: OBV tendência negativa nos últimos 5-10 candles
            obv_trend = analysis.volume_flow.obv_trend
            if obv_trend < 0:
                bearish_conditions += 1
            
            # Relaxar: permitir PUT se (≥2 triggers bearish) OU death_cross ativo
            if bearish_conditions < 2 and not death_cross_active:
                analysis.rejection_reasons.append(f"prefilter_reject: insufficient_bearish_conditions_{bearish_conditions}/4_and_no_death_cross")
                analysis.prefilter_reject = True
                analysis.prefilter_reason = f"insufficient_bearish_conditions_{bearish_conditions}/4_and_no_death_cross"
                return False
            
            # 3. Evitar chasing: se bb_position < -0.8 (muito colado à banda inferior)
            bb_position = analysis.momentum.bb_position
            if bb_position < -0.8:
                analysis.rejection_reasons.append("prefilter_reject: chasing_lower_band")
                analysis.prefilter_reject = True
                analysis.prefilter_reason = "chasing_lower_band"
                return False
            
            # 4. Em squeeze, só aceite se (close < banda inferior) e volume > 1.5 * média20
            adx_value = float(analysis.momentum.adx)
            if adx_value < 18:  # Regime lateral (squeeze)
                try:
                    if recent_data is not None and len(recent_data) >= 20:
                        current_close = analysis.current_price
                        bb_lower = analysis.momentum.bb_lower
                        current_volume = recent_data['volume'].iloc[-1]
                        avg_volume_20 = recent_data['volume'].tail(20).mean()
                        
                        if not (current_close < bb_lower and current_volume > 1.5 * avg_volume_20):
                            analysis.rejection_reasons.append("prefilter_reject: squeeze_conditions_not_met")
                            analysis.prefilter_reject = True
                            analysis.prefilter_reason = "squeeze_conditions_not_met"
                            return False
                except:
                    pass
            
            # Se passou em todos os filtros
            analysis.prefilter_reject = False
            analysis.prefilter_reason = "passed"
            return True
            
        except Exception as e:
            # Em caso de erro, aceita o sinal (não descarta por erro técnico)
            analysis.rejection_reasons.append(f"prefilter_error: {str(e)}")
            analysis.prefilter_reject = False
            analysis.prefilter_reason = "error"
            return True

    def _meta_label_put_filter(self, analysis: 'ProfessionalAnalysis') -> bool:
        """
        Meta-labeling específico para PUTs: filtro extra no D+1 para descartar sinais frágeis.
        Retorna True se o sinal PUT deve ser mantido, False se deve ser descartado.
        """
        try:
            # Busca dados do próximo dia (D+1) para validação
            next_day_data = self._get_next_day_data(analysis.ticker, analysis.analysis_date)
            if next_day_data is None:
                # Se não há dados do próximo dia, mantém o sinal
                return True
            
            # Análise do comportamento no D+1
            next_open = next_day_data['open']
            next_high = next_day_data['high'] 
            next_low = next_day_data['low']
            next_close = next_day_data['close']
            next_volume = next_day_data['volume']
            
            # Preço de entrada (fechamento do dia do sinal)
            entry_price = analysis.current_price
            
            # 1. D+1 open contra o sinal >= 0.5*ATR -> rejeite (ml_gap)
            try:
                # Calcula ATR(14) para determinar o threshold
                atr_pct = self._calculate_atr_percentage(analysis.ticker, analysis.analysis_date)
                if atr_pct is not None:
                    gap_threshold = 0.5 * atr_pct  # 0.5 * ATR
                    gap_pct = (next_open - entry_price) / entry_price * 100
                    
                    if gap_pct >= gap_threshold:
                        analysis.rejection_reasons.append(f"ml_gap: gap_up_{gap_pct:.2f}%_vs_threshold_{gap_threshold:.2f}%")
                        return False
            except Exception as e:
                # Fallback: se não conseguir calcular ATR, usa threshold fixo de 3%
                gap_pct = (next_open - entry_price) / entry_price * 100
                if gap_pct >= 3.0:
                    analysis.rejection_reasons.append(f"ml_gap: gap_up_{gap_pct:.2f}%_vs_fallback_3.0%")
                    return False
            
            # 2. D+1 close > high do dia do sinal -> rejeite (ml_killbar)
            try:
                # Busca high do dia do sinal
                signal_day_data = self._get_signal_day_data(analysis.ticker, analysis.analysis_date)
                if signal_day_data is not None:
                    signal_day_high = signal_day_data['high']
                    if next_close > signal_day_high:
                        kill_pct = (next_close - signal_day_high) / signal_day_high * 100
                        analysis.rejection_reasons.append(f"ml_killbar: close_{next_close:.2f}_above_signal_high_{signal_day_high:.2f}_gap_{kill_pct:.2f}%")
                        return False
            except Exception as e:
                pass  # Se houver erro, continua sem este filtro
            
            # 2.5. Reversão forte D+1 -> rejeite se >2.5%
            reversal_pct = (next_close - next_open) / next_open * 100 if next_open > 0 else 0
            if reversal_pct > 2.5:  # Reversão > 2.5% descarta PUT
                analysis.rejection_reasons.append(f"ml_reversal: strong_reversal_{reversal_pct:.2f}%")
                return False
            
            # 3. Opcional: se MFI(14) subir 2 dias seguidos após o sinal -> rejeite (ml_flow)
            try:
                # Busca dados de 3 dias a partir do sinal para calcular MFI
                mfi_data = self._get_mfi_data(analysis.ticker, analysis.analysis_date, days=3)
                if mfi_data is not None and len(mfi_data) >= 2:
                    # Verifica se MFI subiu por 2 dias seguidos
                    mfi_d1 = mfi_data.iloc[0]['mfi']  # D+1
                    mfi_d2 = mfi_data.iloc[1]['mfi']  # D+2
                    
                    if mfi_d1 > mfi_d2:  # MFI subindo por 2 dias seguidos
                        mfi_improvement = mfi_d1 - mfi_d2
                        analysis.rejection_reasons.append(f"ml_flow: mfi_rising_2days_mfi_d1_{mfi_d1:.1f}_d2_{mfi_d2:.1f}_diff_{mfi_improvement:.1f}")
                        return False
            except Exception as e:
                pass  # Se houver erro, continua sem este filtro
            
            # Se passou em todos os filtros, mantém o sinal PUT
            analysis.rejection_reasons.append("meta_label_passed: all_filters_ok")
            return True
            
        except Exception as e:
            # Em caso de erro, mantém o sinal (não descarta por erro técnico)
            analysis.rejection_reasons.append(f"meta_label_error: {str(e)}")
            return True

    def _get_next_day_data(self, ticker: str, analysis_date: datetime) -> Optional[pd.Series]:
        """
        Busca dados do próximo dia útil (D+1) para meta-labeling.
        """
        try:
            from src.core.data.data import get_historical_data
            
            # Busca dados de 3 dias a partir da data de análise
            start_date = analysis_date.strftime('%Y-%m-%d')
            end_date = (analysis_date + timedelta(days=3)).strftime('%Y-%m-%d')
            
            df = get_historical_data(ticker, start_date, end_date)
            if df.empty:
                return None
            
            # Filtra apenas dias após a data de análise
            df = df[df['date'] > analysis_date]
            
            if df.empty:
                return None
            
            # Retorna o primeiro dia útil após a análise
            return df.iloc[0]
            
        except Exception as e:
            return None

    def _calculate_atr_percentage(self, ticker: str, analysis_date: datetime) -> Optional[float]:
        """
        Calcula ATR(14) em percentual para usar como threshold.
        """
        try:
            from src.core.data.data import get_historical_data
            import numpy as np
            
            # Busca dados históricos para calcular ATR
            start_date = (analysis_date - timedelta(days=120)).strftime('%Y-%m-%d')
            end_date = analysis_date.strftime('%Y-%m-%d')
            
            df = get_historical_data(ticker, start_date, end_date)
            if df.empty or len(df) < 20:
                return None
            
            # Filtra até a data de análise
            df = df[df['date'] <= analysis_date]
            
            if len(df) < 20:
                return None
            
            # Calcula True Range
            h = df['high'].astype(float).values
            l = df['low'].astype(float).values
            c = df['close'].astype(float).values
            
            prev_c = np.roll(c, 1)
            prev_c[0] = c[0]
            
            tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
            
            # Converte TR para % sobre o fechamento anterior
            with np.errstate(divide='ignore', invalid='ignore'):
                tr_pct = np.where(prev_c != 0, (tr / prev_c) * 100.0, 0.0)
            
            # ATR(14) em %
            if len(tr_pct) >= 14:
                atr_pct = float(pd.Series(tr_pct).rolling(14).mean().iloc[-1])
            else:
                atr_pct = float(np.nanmean(tr_pct[-14:])) if len(tr_pct) > 0 else None
            
            if not np.isfinite(atr_pct) or atr_pct <= 0:
                return None
                
            return atr_pct
            
        except Exception as e:
            return None

    def _get_signal_day_data(self, ticker: str, analysis_date: datetime) -> Optional[pd.Series]:
        """
        Busca dados do dia do sinal para obter o high do dia.
        """
        try:
            from src.core.data.data import get_historical_data
            
            # Busca dados do dia do sinal
            start_date = analysis_date.strftime('%Y-%m-%d')
            end_date = (analysis_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            df = get_historical_data(ticker, start_date, end_date)
            if df.empty:
                return None
            
            # Filtra apenas o dia do sinal
            df = df[df['date'] == analysis_date]
            
            if df.empty:
                return None
            
            return df.iloc[0]
            
        except Exception as e:
            return None

    def _get_mfi_data(self, ticker: str, analysis_date: datetime, days: int = 3) -> Optional[pd.DataFrame]:
        """
        Busca dados para calcular MFI(14) dos próximos dias após o sinal.
        """
        try:
            from src.core.data.data import get_historical_data
            import numpy as np
            
            # Busca dados históricos incluindo os dias após o sinal
            start_date = (analysis_date - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = (analysis_date + timedelta(days=days+1)).strftime('%Y-%m-%d')
            
            df = get_historical_data(ticker, start_date, end_date)
            if df.empty or len(df) < 20:
                return None
            
            # Filtra até alguns dias após a análise
            df = df[df['date'] <= analysis_date + timedelta(days=days)]
            
            if len(df) < 20:
                return None
            
            # Calcula MFI(14) para cada dia
            high = df['high'].astype(float).values
            low = df['low'].astype(float).values
            close = df['close'].astype(float).values
            volume = df['volume'].astype(float).values
            
            mfi_values = []
            dates = []
            
            for i in range(14, len(df)):
                # Calcula Typical Price
                typical_price = (high[i-14:i+1] + low[i-14:i+1] + close[i-14:i+1]) / 3
                
                # Calcula Raw Money Flow
                raw_money_flow = typical_price * volume[i-14:i+1]
                
                # Determina Positive/Negative Money Flow
                positive_flow = 0
                negative_flow = 0
                
                for j in range(1, len(typical_price)):
                    if typical_price[j] > typical_price[j-1]:
                        positive_flow += raw_money_flow[j]
                    elif typical_price[j] < typical_price[j-1]:
                        negative_flow += raw_money_flow[j]
                
                # Calcula MFI
                if negative_flow == 0:
                    mfi = 100
                else:
                    money_flow_ratio = positive_flow / negative_flow
                    mfi = 100 - (100 / (1 + money_flow_ratio))
                
                mfi_values.append(mfi)
                dates.append(df.iloc[i]['date'])
            
            # Retorna apenas os dias após o sinal
            signal_date = analysis_date
            result_data = []
            
            for i, date in enumerate(dates):
                if date > signal_date:
                    result_data.append({
                        'date': date,
                        'mfi': mfi_values[i]
                    })
            
            if not result_data:
                return None
                
            return pd.DataFrame(result_data)
            
        except Exception as e:
            return None

    def _passes_quality_gates(self, analysis: 'ProfessionalAnalysis', final_score: float, thr_put: float) -> bool:
        adx = float(analysis.momentum.adx)
        confidence = self._calculate_calibrated_confidence(final_score, Direction.PUT if final_score <= -thr_put else Direction.CALL if final_score >= thr_put else Direction.NEUTRAL)
        
        # Regras relaxadas para sinais PUT
        if final_score <= -thr_put and confidence >= 55:
            # Para sinais PUT: relaxe regras
            # Ignore gate de volume fraco
            # Ignore gate de squeeze
            analysis.gates_relaxed = True
            # Apenas verifica ADX mínimo
            if (self.horizon == 'curto' and adx < 8) or adx < 6:
                analysis.gate_counters['gate_adx'] = analysis.gate_counters.get('gate_adx', 0) + 1
                return False
            return True
        
        # Regras normais para outros sinais
        analysis.gates_relaxed = False
        
        # Piso de ADX mais baixo para permitir mais sinais
        if (self.horizon == 'curto' and adx < 8) or adx < 6:  # Mais permissivo
            analysis.gate_counters['gate_adx'] = analysis.gate_counters.get('gate_adx', 0) + 1
            return False
        # Volume: bloqueia apenas se muito contrário
        if analysis.volume_flow.volume_score < -0.4:  # Mais permissivo
            analysis.gate_counters['gate_volume'] = analysis.gate_counters.get('gate_volume', 0) + 1
            return False
        # Sentimento: bloqueia apenas se muito negativo
        if analysis.options_sentiment.sentiment_score < -0.5:  # Mais permissivo
            analysis.gate_counters['gate_sentiment'] = analysis.gate_counters.get('gate_sentiment', 0) + 1
            return False
        # Alinhamento tendência x momentum menos restritivo para permitir scores negativos
        trend = analysis.trend.trend_score
        mom = analysis.momentum.momentum_score
        if self.horizon == 'curto':
            # Permite mais divergências para gerar PUTs
            if (trend < -0.30 and mom > 0.40) or (trend > 0.30 and mom < -0.40):
                analysis.gate_counters['gate_align'] = analysis.gate_counters.get('gate_align', 0) + 1
                return False
            if abs(mom) < 0.01:  # Mais permissivo
                analysis.gate_counters['gate_momentum_weak'] = analysis.gate_counters.get('gate_momentum_weak', 0) + 1
                return False
        else:
            if trend < -0.35 and mom > 0.35:
                analysis.gate_counters['gate_align'] = analysis.gate_counters.get('gate_align', 0) + 1
                return False
            if trend > 0.35 and mom < -0.35:
                analysis.gate_counters['gate_align'] = analysis.gate_counters.get('gate_align', 0) + 1
                return False
        return True

    def _adaptive_threshold(self, adx: float, base: float = 0.15) -> float:
        # Em tendência forte, permita corte menor para capturar movimentos
        if adx >= 25:
            return max(0.10, base - 0.05)
        # Em mercado lateral, eleve o corte moderadamente
        if adx <= 15:
            return base + 0.05
        # Neutro mantém base
        return base
    
    def generate_strategy_recommendation(self, analysis: 'ProfessionalAnalysis') -> str:
        """Gera recomendação de estratégia profissional."""
        direction = analysis.direction
        confidence = analysis.confidence
        
        if direction == Direction.CALL:
            if confidence > 70:
                return "Compra de CALL OTM com delta 0.30-0.40 para alavancagem"
            elif confidence > 50:
                return "Venda de PUT OTM com delta 0.15-0.25 para renda"
            else:
                return "Venda de CALL coberta para renda conservadora"
        
        elif direction == Direction.PUT:
            if confidence > 70:
                return "Compra de PUT OTM com delta 0.30-0.40 para proteção"
            elif confidence > 50:
                return "Venda de CALL OTM com delta 0.15-0.25 para renda"
            else:
                return "Venda de PUT coberta para renda conservadora"
        
        else:  # NEUTRAL
            return "Straddle ou Strangle para aproveitar volatilidade, ou não operar"
    
    def analyze(self, ticker: str, price_data: pd.DataFrame) -> ProfessionalAnalysis:
        """Executa análise profissional completa."""
        # Validação: verifica se temos dados suficientes
        if price_data.empty or len(price_data) < 50:
            raise ValueError(f"Dados insuficientes para análise profissional. "
                           f"Recebido: {len(price_data)} registros, mínimo: 50")
        
        if 'close' not in price_data.columns:
            raise ValueError("Dados devem conter coluna 'close'")
        
        current_price = price_data['close'].iloc[-1]
        
        # Executa todas as camadas de análise
        trend = self.analyze_trend(price_data)
        momentum = self.analyze_momentum(price_data)
        volume_flow = self.analyze_volume_flow(price_data)
        options_sentiment = self.analyze_options_sentiment(ticker, price_data)
        macro_context = self.analyze_macro_context(ticker, price_data)
        
        # Cria análise completa
        analysis = ProfessionalAnalysis(
            ticker=ticker,
            current_price=float(current_price),
            analysis_date=price_data.index[-1].to_pydatetime() if hasattr(price_data.index, 'to_pydatetime') else datetime.now(),
            trend=trend,
            momentum=momentum,
            volume_flow=volume_flow,
            options_sentiment=options_sentiment,
            macro_context=macro_context,
            final_score=0.0,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            key_drivers=[],
            strategy_recommendation="",
            gate_counters={},
            rejection_reasons=[]
        )
        
        # Calcula decisão final
        direction, confidence, drivers = self.calculate_final_decision(analysis)
        
        # Calcula horizonte dinâmico baseado no regime de mercado
        dynamic_horizon = self._calculate_dynamic_horizon(trend, momentum, volume_flow)
        
        # Usa exatamente os mesmos pesos efetivos definidos em calculate_final_decision
        eff = getattr(analysis, 'effective_weights', None) or self._dynamic_weights(float(momentum.adx))
        analysis.final_score = (
            trend.trend_score * eff['trend'] +
            momentum.momentum_score * eff['momentum'] +
            volume_flow.volume_score * eff['volume'] +
            options_sentiment.sentiment_score * eff['sentiment'] +
            macro_context.overall_context_score * eff['macro']
        )
        analysis.effective_weights = eff
        analysis.direction = direction
        analysis.confidence = confidence
        analysis.key_drivers = drivers
        analysis.strategy_recommendation = self.generate_strategy_recommendation(analysis)
        analysis.dynamic_horizon = dynamic_horizon
        
        return analysis
    
    # Métodos auxiliares para cálculos técnicos
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calcula Average True Range."""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcula RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        # Evita divisão por zero
        rs = gain / loss.where(loss != 0, 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula MACD."""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal
        return macd_line, signal, histogram
    
    def _calculate_adx(self, price_data: pd.DataFrame) -> float:
        """Calcula ADX simplificado."""
        high = price_data.get('high', price_data['close'])
        low = price_data.get('low', price_data['close'] * 0.98)
        close = price_data['close']
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        # Usa numpy para evitar ambiguidade de comparações booleanas
        dm_plus_condition = (dm_plus > dm_minus) & (dm_plus > 0)
        dm_minus_condition = (dm_minus > dm_plus) & (dm_minus > 0)
        
        dm_plus = dm_plus.where(dm_plus_condition, 0)
        dm_minus = dm_minus.where(dm_minus_condition, 0)
        
        # Smoothed values
        atr = tr.rolling(14).mean()
        di_plus = 100 * (dm_plus.rolling(14).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(14).mean() / atr)
        
        # ADX - evita divisão por zero
        di_sum = di_plus + di_minus
        dx = 100 * ((di_plus - di_minus).abs() / di_sum.where(di_sum != 0, 1))
        adx = dx.rolling(14).mean()
        
        return float(adx.iloc[-1]) if not adx.empty and not pd.isna(adx.iloc[-1]) else 20.0
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calcula On-Balance Volume."""
        price_change = close.diff()
        obv = np.where(price_change > 0, volume, 
                      np.where(price_change < 0, -volume, 0))
        return pd.Series(obv, index=close.index).cumsum()
    
    def _calculate_accumulation_distribution(self, price_data: pd.DataFrame) -> pd.Series:
        """Calcula Accumulation/Distribution Line."""
        high = price_data.get('high', price_data['close'])
        low = price_data.get('low', price_data['close'] * 0.98)
        close = price_data['close']
        volume = price_data.get('volume', pd.Series([1.0] * len(close), index=close.index))
        
        # Evita divisão por zero no CLV
        hl_diff = high - low
        clv = ((close - low) - (high - close)) / hl_diff.where(hl_diff != 0, 1e-10)
        clv = clv.fillna(0)
        return (clv * volume).cumsum()
    
    def _calculate_macd_adaptive(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula MACD com parâmetros adaptativos."""
        ema_fast = prices.ewm(span=self.params["macd_fast"]).mean()
        ema_slow = prices.ewm(span=self.params["macd_slow"]).mean()
        macd_line = ema_fast - ema_slow
        signal = macd_line.ewm(span=self.params["macd_signal"]).mean()
        histogram = macd_line - signal
        return macd_line, signal, histogram
    
    def _calculate_adx_adaptive(self, price_data: pd.DataFrame) -> float:
        """Calcula ADX com período adaptativo."""
        high = price_data.get('high', price_data['close'])
        low = price_data.get('low', price_data['close'] * 0.98)
        close = price_data['close']
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        # Usa numpy para evitar ambiguidade de comparações booleanas
        dm_plus_condition = (dm_plus > dm_minus) & (dm_plus > 0)
        dm_minus_condition = (dm_minus > dm_plus) & (dm_minus > 0)
        
        dm_plus = dm_plus.where(dm_plus_condition, 0)
        dm_minus = dm_minus.where(dm_minus_condition, 0)
        
        # Smoothed values com período adaptativo
        period = self.params["adx_period"]
        atr = tr.rolling(period).mean()
        di_plus = 100 * (dm_plus.rolling(period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(period).mean() / atr)
        
        # ADX - evita divisão por zero
        di_sum = di_plus + di_minus
        dx = 100 * ((di_plus - di_minus).abs() / di_sum.where(di_sum != 0, 1))
        adx = dx.rolling(period).mean()
        
        return float(adx.iloc[-1]) if not adx.empty and not pd.isna(adx.iloc[-1]) else 20.0
    
    def _calculate_calibrated_confidence(self, final_score: float, direction: Direction) -> float:
        """Calcula confiança calibrada baseada no score e direção."""
        abs_score = abs(final_score)
        
        if direction == Direction.NEUTRAL:
            # Para NEUTRAL: confiança baixa (20-40%) proporcional ao score
            # Quanto menor o score absoluto, maior a confiança no NEUTRAL
            neutral_confidence = 40 - (abs_score * 20)  # Score 0 = 40%, Score 1 = 20%
            return max(20, min(40, neutral_confidence))
        else:
            # Para CALL/PUT: confiança proporcional ao score absoluto
            # Score 0.3 → ~55%, Score 0.6 → ~70%, Score 0.9 → ~85%
            # Fórmula: 50 + (abs_score * 40)
            directional_confidence = 50 + (abs_score * 40)
            return max(50, min(90, directional_confidence))
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[float, bool]:
        """
        Calcula posição nas Bollinger Bands e detecta squeeze.
        
        Returns:
            bb_position: Posição normalizada (-1 a +1)
            bb_squeeze: Se está em squeeze (baixa volatilidade)
        """
        if len(prices) < period:
            return 0.0, False
        
        # Média móvel e desvio padrão
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        # Bandas
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Valores atuais
        current_price = float(prices.iloc[-1])
        current_sma = float(sma.iloc[-1])
        current_upper = float(upper_band.iloc[-1])
        current_lower = float(lower_band.iloc[-1])
        
        # Posição normalizada (-1 = banda inferior, 0 = SMA, +1 = banda superior)
        band_width = current_upper - current_lower
        if band_width > 0:
            bb_position = (current_price - current_sma) / (band_width / 2)
            bb_position = max(-1.0, min(1.0, bb_position))
        else:
            bb_position = 0.0
        
        # Squeeze: quando as bandas estão muito próximas (baixa volatilidade)
        # Compara largura atual com média histórica
        if len(std) >= period * 2:
            avg_std = std.rolling(period * 2).mean().iloc[-1]
            current_std = std.iloc[-1]
            bb_squeeze = current_std < (avg_std * 0.7)  # 30% abaixo da média
        else:
            bb_squeeze = False
        
        return float(bb_position), bb_squeeze
    
    def _calculate_stochastic(self, price_data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[float, float, str]:
        """
        Calcula Stochastic Oscillator.
        
        Returns:
            stoch_k: %K do Stochastic
            stoch_d: %D do Stochastic (média móvel do %K)
            stoch_signal: "OVERSOLD", "OVERBOUGHT", "NEUTRAL"
        """
        close = price_data['close']
        high = price_data.get('high', close)
        low = price_data.get('low', close * 0.98)  # Aproximação se não tiver high/low
        
        if len(close) < k_period:
            return 50.0, 50.0, "NEUTRAL"
        
        # Highest High e Lowest Low no período
        highest_high = high.rolling(k_period).max()
        lowest_low = low.rolling(k_period).min()
        
        # %K
        k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        k_percent = k_percent.fillna(50)  # Valores neutros para NaN
        
        # %D (média móvel do %K)
        d_percent = k_percent.rolling(d_period).mean()
        
        # Valores atuais
        stoch_k = float(k_percent.iloc[-1])
        stoch_d = float(d_percent.iloc[-1])
        
        # Sinal baseado nos níveis
        if stoch_k < 20 and stoch_d < 20:
            stoch_signal = "OVERSOLD"
        elif stoch_k > 80 and stoch_d > 80:
            stoch_signal = "OVERBOUGHT"
        else:
            stoch_signal = "NEUTRAL"
        
        return stoch_k, stoch_d, stoch_signal
    
    def _calculate_dynamic_horizon(self, trend: TrendAnalysis, momentum: MomentumAnalysis, volume_flow: VolumeFlowAnalysis) -> DynamicHorizonConfig:
        """
        Calcula horizonte dinâmico baseado no regime de mercado.
        
        Lógica:
        - ADX < 15: Mercado lateral → janelas curtas (5-7 dias)
        - ADX > 25: Tendência clara → janelas médias (10-15 dias)  
        - BB Squeeze + Volume alto: Possível breakout → janelas longas (15-20 dias)
        """
        adx = momentum.adx
        bb_squeeze = momentum.bb_squeeze
        volume_strength = volume_flow.volume_score
        trend_strength = trend.trend_score
        
        # Determina regime de mercado
        if adx < 15:
            # Mercado lateral - operações mais rápidas
            regime = "LATERAL"
            evaluation_days = 5
            position_size = 0.3  # Posição menor em mercado lateral
            stop_loss_pct = 2.0  # Stop mais apertado
            take_profit_pct = 3.0
            
        elif adx > 25:
            # Tendência clara - pode segurar mais tempo
            regime = "TRENDING"
            evaluation_days = 12
            position_size = 0.5  # Posição média em tendência
            stop_loss_pct = 3.0
            take_profit_pct = 6.0
            
        elif bb_squeeze and abs(volume_strength) > 0.3:
            # Possível breakout - aguardar movimento
            regime = "BREAKOUT"
            evaluation_days = 20
            position_size = 0.7  # Posição maior em breakout
            stop_loss_pct = 4.0  # Stop mais largo para breakout
            take_profit_pct = 8.0
            
        else:
            # Regime neutro - configuração padrão
            regime = "NEUTRAL"
            evaluation_days = 10
            position_size = 0.4
            stop_loss_pct = 2.5
            take_profit_pct = 5.0
        
        # Ajustes baseados na força da tendência
        if abs(trend_strength) > 0.6:  # Tendência muito forte
            evaluation_days = int(evaluation_days * 1.3)  # Alonga horizonte
            take_profit_pct *= 1.2  # Target maior
        elif abs(trend_strength) < 0.2:  # Tendência fraca
            evaluation_days = max(5, int(evaluation_days * 0.8))  # Encurta horizonte
            stop_loss_pct *= 0.9  # Stop mais apertado
        
        return DynamicHorizonConfig(
            evaluation_days=min(25, max(5, evaluation_days)),  # Limita entre 5-25 dias
            position_size=max(0.1, min(0.8, position_size)),   # Limita entre 10-80%
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            regime=regime
        )
    
    def _calculate_sma_distance_score(self, sma_medium: float, sma_long: float, sma_trend: float) -> float:
        """
        Calcula score baseado na distância entre SMAs para médio prazo.
        
        Lógica:
        - Maior distância entre SMA50/100 vs SMA200 = tendência mais forte
        - SMAs alinhadas e divergindo = tendência se fortalecendo  
        - SMAs convergindo = possível reversão
        """
        # Distância percentual entre SMA médio/longo e SMA tendência
        medium_distance = (sma_medium - sma_trend) / sma_trend if sma_trend > 0 else 0
        long_distance = (sma_long - sma_trend) / sma_trend if sma_trend > 0 else 0
        
        # Score baseado na distância (normalizado)
        distance_score = (medium_distance + long_distance) / 2
        
        # Bonifica alinhamento das SMAs (todas subindo ou descendo)
        if (sma_medium > sma_long > sma_trend) or (sma_medium < sma_long < sma_trend):
            alignment_bonus = 0.1  # SMAs alinhadas
        else:
            alignment_bonus = -0.05  # SMAs desalinhadas (indecisão)
        
        # Limita o score para não dominar outros fatores
        total_score = distance_score + alignment_bonus
        return max(-0.2, min(0.2, total_score))


__all__ = ["ProfessionalAnalyzer", "ProfessionalAnalysis", "Direction", "DynamicHorizonConfig"]
