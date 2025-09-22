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
from datetime import datetime, timedelta

from oplab_client import OpLabClient


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
    macd_signal: float
    adx: float
    roc_10: float
    roc_20: float
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


class ProfessionalAnalyzer:
    """Analisador profissional para Radar de Direção."""
    
    def __init__(self, client: Optional[OpLabClient] = None, horizon: str = "médio"):
        self.client = client or OpLabClient()
        self.horizon = horizon
        self.params = self._get_horizon_parameters(horizon)
    
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
        sma_short = close.rolling(self.params["sma_short"]).mean().iloc[-1]
        sma_medium = close.rolling(self.params["sma_medium"]).mean().iloc[-1]
        sma_long = close.rolling(self.params["sma_long"]).mean().iloc[-1]
        sma_trend = close.rolling(self.params["sma_trend"]).mean().iloc[-1]
        
        # Golden/Death Cross (usando médias adaptativas)
        golden_cross = bool(sma_medium > sma_trend)
        death_cross = bool(sma_medium < sma_trend)
        
        # Regime de volatilidade (ATR)
        high = price_data.get('high', close)
        low = price_data.get('low', close)
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
        
        # Score de tendência normalizado
        trend_score = 0.0
        
        # Peso das médias móveis adaptativas
        if sma_short > sma_medium:
            trend_score += 0.2
        if sma_medium > sma_long:
            trend_score += 0.3
        if sma_long > sma_trend:
            trend_score += 0.3
        if golden_cross:
            trend_score += 0.2
        elif death_cross:
            trend_score -= 0.2
        
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
        
        # MACD
        macd_hist_val = float(macd_hist.iloc[-1])
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
            macd_signal=float(macd_signal.iloc[-1]),
            adx=adx,
            roc_10=roc_10_val,
            roc_20=float(roc_20.iloc[-1]),
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
        
        volume = price_data.get('volume', pd.Series([1] * len(close)))
        
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
        
        # Accumulation/Distribution
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
            
            # Volume e Open Interest - filtra apenas opções com dados válidos
            calls_with_volume = calls[calls['volume'].notna() & (calls['volume'] > 0)]
            puts_with_volume = puts[puts['volume'].notna() & (puts['volume'] > 0)]
            
            call_volume = calls_with_volume['volume'].sum()
            put_volume = puts_with_volume['volume'].sum()
            total_volume = call_volume + put_volume
            
            # Open Interest (se disponível)
            calls_with_oi = calls[calls.get('open_interest', pd.Series()).notna() & (calls.get('open_interest', pd.Series()) > 0)]
            puts_with_oi = puts[puts.get('open_interest', pd.Series()).notna() & (puts.get('open_interest', pd.Series()) > 0)]
            
            call_oi = calls_with_oi.get('open_interest', pd.Series()).sum() if not calls_with_oi.empty else 0
            put_oi = puts_with_oi.get('open_interest', pd.Series()).sum() if not puts_with_oi.empty else 0
            total_oi = call_oi + put_oi
            
            # Calcula ratios apenas se temos dados válidos
            if total_volume > 0:
                call_volume_ratio = call_volume / total_volume
                put_volume_ratio = put_volume / total_volume
            else:
                # Se não há volume, tenta usar quantidade de contratos
                call_count = len(calls_with_volume)
                put_count = len(puts_with_volume)
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
            
            # Volatility Skew (simplificado)
            current_price = float(price_data['close'].iloc[-1])
            lower_bound = current_price * 0.95
            upper_bound = current_price * 1.05
            
            # Evita erro de ambiguidade com comparações de Series
            atm_calls = calls[(calls['strike'] >= lower_bound) & 
                            (calls['strike'] <= upper_bound)]
            atm_puts = puts[(puts['strike'] >= lower_bound) & 
                           (puts['strike'] <= upper_bound)]
            
            # Calcula volatility skew se temos dados de IV
            if not atm_calls.empty and not atm_puts.empty and 'implied_volatility' in calls.columns:
                call_iv = atm_calls['implied_volatility'].fillna(0).mean()
                put_iv = atm_puts['implied_volatility'].fillna(0).mean()
                volatility_skew = put_iv - call_iv if call_iv > 0 and put_iv > 0 else 0.0
            else:
                # Aproximação usando preços de opções (premium/strike ratio)
                if not atm_calls.empty and not atm_puts.empty:
                    call_premium_ratio = (atm_calls['last'].fillna(atm_calls['bid'].fillna(0)).mean() / current_price)
                    put_premium_ratio = (atm_puts['last'].fillna(atm_puts['bid'].fillna(0)).mean() / current_price)
                    volatility_skew = put_premium_ratio - call_premium_ratio
                else:
                    volatility_skew = 0.0
            
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
            context_bias = "NEUTRO"
        
        return MacroContextAnalysis(
            sector_score=sector_score,
            macro_score=macro_score,
            commodity_score=commodity_score,
            overall_context_score=overall_context_score,
            context_bias=context_bias
        )
    
    def calculate_final_decision(self, analysis: 'ProfessionalAnalysis') -> Tuple[Direction, float, List[str]]:
        """Camada 6: Modelo de decisão com score normalizado."""
        # Pesos das camadas
        weights = {
            'trend': 0.45,      # 45% - tendência é fundamental
            'momentum': 0.25,   # 25% - momentum confirma tendência
            'volume': 0.15,     # 15% - volume confirma movimento
            'sentiment': 0.10,  # 10% - sentimento do mercado
            'macro': 0.05       # 5% - contexto macro
        }
        
        # Score final ponderado
        final_score = (
            analysis.trend.trend_score * weights['trend'] +
            analysis.momentum.momentum_score * weights['momentum'] +
            analysis.volume_flow.volume_score * weights['volume'] +
            analysis.options_sentiment.sentiment_score * weights['sentiment'] +
            analysis.macro_context.overall_context_score * weights['macro']
        )
        
        # Normaliza para -1 a +1
        final_score = max(-1.0, min(1.0, final_score))
        
        # Determina direção com calibração logística (ajustado para scores normalizados)
        if final_score >= 0.15:
            direction = Direction.CALL
            confidence = self._calculate_logistic_confidence(final_score)
        elif final_score <= -0.15:
            direction = Direction.PUT
            confidence = self._calculate_logistic_confidence(final_score)
        else:
            direction = Direction.NEUTRAL
            confidence = max(20, min(40, 30 + abs(final_score) * 20))
        
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
            current_price=current_price,
            analysis_date=datetime.now(),
            trend=trend,
            momentum=momentum,
            volume_flow=volume_flow,
            options_sentiment=options_sentiment,
            macro_context=macro_context,
            final_score=0.0,  # Será calculado
            direction=Direction.NEUTRAL,  # Será calculado
            confidence=0.0,  # Será calculado
            key_drivers=[],  # Será calculado
            strategy_recommendation=""  # Será calculado
        )
        
        # Calcula decisão final
        direction, confidence, drivers = self.calculate_final_decision(analysis)
        
        # Atualiza análise
        analysis.final_score = (
            trend.trend_score * 0.45 +
            momentum.momentum_score * 0.25 +
            volume_flow.volume_score * 0.15 +
            options_sentiment.sentiment_score * 0.10 +
            macro_context.overall_context_score * 0.05
        )
        analysis.direction = direction
        analysis.confidence = confidence
        analysis.key_drivers = drivers
        analysis.strategy_recommendation = self.generate_strategy_recommendation(analysis)
        
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
        volume = price_data.get('volume', pd.Series([1] * len(close)))
        
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
    
    def _calculate_logistic_confidence(self, final_score: float) -> float:
        """Calcula confiança usando curva logística calibrada."""
        # Curva logística: confidence = 100 / (1 + exp(-5 * final_score))
        # Ajustada para dar valores mais realistas entre 50-90%
        raw_confidence = 100 / (1 + np.exp(-5 * abs(final_score)))
        # Mapeia para range 50-90% para scores significativos
        calibrated_confidence = 50 + (raw_confidence - 50) * 0.8
        return min(90, max(50, calibrated_confidence))


__all__ = ["ProfessionalAnalyzer", "ProfessionalAnalysis", "Direction"]
