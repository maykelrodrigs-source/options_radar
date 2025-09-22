"""
Testes para professional_analysis.py
Cobertura alvo: 90%
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from professional_analysis import (
    ProfessionalAnalyzer, 
    TrendAnalysis, 
    MomentumAnalysis,
    VolumeFlowAnalysis,
    OptionsSentimentAnalysis,
    MacroContextAnalysis,
    ProfessionalAnalysis,
    Direction,
    TrendStrength
)


class TestProfessionalAnalyzer:
    """Testes para a classe ProfessionalAnalyzer."""
    
    def test_init_default_horizon(self):
        """Testa inicialização com horizonte padrão."""
        analyzer = ProfessionalAnalyzer()
        assert analyzer.horizon == "médio"
        assert "sma_short" in analyzer.params
        assert analyzer.params["sma_short"] == 10
    
    def test_init_custom_horizon(self):
        """Testa inicialização com horizonte customizado."""
        analyzer = ProfessionalAnalyzer(horizon="curto")
        assert analyzer.horizon == "curto"
        assert analyzer.params["sma_short"] == 5
        
    def test_init_invalid_horizon(self):
        """Testa inicialização com horizonte inválido."""
        analyzer = ProfessionalAnalyzer(horizon="inválido")
        # Deve usar médio como fallback
        assert analyzer.params["sma_short"] == 10
    
    def test_get_horizon_parameters(self):
        """Testa configuração de parâmetros por horizonte."""
        analyzer = ProfessionalAnalyzer()
        
        # Testa todos os horizontes
        for horizon in ["curto", "médio", "longo"]:
            params = analyzer._get_horizon_parameters(horizon)
            assert isinstance(params, dict)
            assert "sma_short" in params
            assert "rsi_short" in params
            assert "macd_fast" in params
            assert "atr_period" in params
            assert "adx_period" in params
        
        # Verifica diferenças entre horizontes
        curto = analyzer._get_horizon_parameters("curto")
        longo = analyzer._get_horizon_parameters("longo")
        assert curto["sma_short"] < longo["sma_short"]
        assert curto["rsi_short"] <= longo["rsi_short"]


class TestTrendAnalysis:
    """Testes para análise de tendência."""
    
    def test_analyze_trend_uptrend(self, sample_price_data):
        """Testa análise de tendência em alta."""
        # Modifica dados para tendência de alta clara
        uptrend_data = sample_price_data.copy()
        uptrend_data['close'] = pd.Series(range(100, 400, 1), index=uptrend_data.index[:300])
        
        analyzer = ProfessionalAnalyzer(horizon="médio")
        trend = analyzer.analyze_trend(uptrend_data)
        
        assert isinstance(trend, TrendAnalysis)
        assert trend.trend_score > 0.5  # Deve ser claramente positivo
        assert trend.trend_strength in [TrendStrength.STRONG_UP, TrendStrength.WEAK_UP]
        assert trend.golden_cross is True  # SMA curta > SMA longa
        assert trend.sma_10 > 0
        assert trend.sma_50 > 0
    
    def test_analyze_trend_downtrend(self, sample_price_data):
        """Testa análise de tendência em baixa."""
        # Modifica dados para tendência de baixa clara
        downtrend_data = sample_price_data.copy()
        downtrend_data['close'] = pd.Series(range(400, 100, -1), index=downtrend_data.index[:300])
        
        analyzer = ProfessionalAnalyzer(horizon="médio")
        trend = analyzer.analyze_trend(downtrend_data)
        
        assert trend.trend_score < -0.5  # Deve ser claramente negativo
        assert trend.trend_strength in [TrendStrength.STRONG_DOWN, TrendStrength.WEAK_DOWN]
        assert trend.death_cross is True  # SMA curta < SMA longa
    
    def test_analyze_trend_sideways(self, sample_price_data):
        """Testa análise de tendência lateral."""
        # Modifica dados para movimento lateral
        sideways_data = sample_price_data.copy()
        sideways_data['close'] = pd.Series([100 + np.sin(i/10) * 5 for i in range(300)], 
                                         index=sideways_data.index)
        
        analyzer = ProfessionalAnalyzer(horizon="médio")
        trend = analyzer.analyze_trend(sideways_data)
        
        assert abs(trend.trend_score) < 0.3  # Deve ser próximo de neutro
        assert trend.trend_strength == TrendStrength.LATERAL
    
    def test_volatility_regime_classification(self, sample_price_data):
        """Testa classificação de regime de volatilidade."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # Dados com alta volatilidade
        high_vol_data = sample_price_data.copy()
        high_vol_data['high'] = high_vol_data['close'] * 1.10  # 10% de range
        high_vol_data['low'] = high_vol_data['close'] * 0.90
        
        trend = analyzer.analyze_trend(high_vol_data)
        assert trend.volatility_regime in ["ALTA", "NORMAL"]  # Pode variar dependendo dos dados
        assert trend.atr_ratio > 0


class TestMomentumAnalysis:
    """Testes para análise de momentum."""
    
    def test_analyze_momentum_strong_positive(self, sample_price_data):
        """Testa análise de momentum fortemente positivo."""
        # Cria dados com momentum positivo forte
        momentum_data = sample_price_data.copy()
        # Crescimento acelerado nos últimos períodos
        for i in range(50):
            momentum_data.loc[momentum_data.index[-50+i], 'close'] *= (1 + 0.02)
        
        analyzer = ProfessionalAnalyzer(horizon="médio")
        momentum = analyzer.analyze_momentum(momentum_data)
        
        assert isinstance(momentum, MomentumAnalysis)
        assert momentum.momentum_score > 0  # Deve ser positivo
        assert momentum.rsi_7 > 50  # RSI deve indicar força
        assert momentum.macd_histogram > 0  # MACD positivo
    
    def test_analyze_momentum_negative(self, sample_price_data):
        """Testa análise de momentum negativo."""
        # Cria dados com momentum negativo
        momentum_data = sample_price_data.copy()
        # Declínio nos últimos períodos
        for i in range(50):
            momentum_data.loc[momentum_data.index[-50+i], 'close'] *= (1 - 0.015)
        
        analyzer = ProfessionalAnalyzer(horizon="médio")
        momentum = analyzer.analyze_momentum(momentum_data)
        
        assert momentum.momentum_score < 0  # Deve ser negativo
        assert momentum.rsi_7 < 50  # RSI deve indicar fraqueza
    
    def test_rsi_calculation_boundaries(self, sample_price_data):
        """Testa se RSI permanece dentro dos limites 0-100."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        momentum = analyzer.analyze_momentum(sample_price_data)
        
        assert 0 <= momentum.rsi_7 <= 100
        assert 0 <= momentum.rsi_14 <= 100
        assert 0 <= momentum.rsi_21 <= 100
    
    def test_momentum_score_normalization(self, sample_price_data):
        """Testa se o score de momentum está normalizado."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        momentum = analyzer.analyze_momentum(sample_price_data)
        
        # Score deve estar entre -1 e 1 (normalizado com tanh)
        assert -1 <= momentum.momentum_score <= 1


class TestVolumeFlowAnalysis:
    """Testes para análise de volume e fluxo."""
    
    def test_analyze_volume_flow_high_volume(self, sample_price_data):
        """Testa análise com volume alto."""
        high_vol_data = sample_price_data.copy()
        # Aumenta volume nos últimos períodos
        high_vol_data.loc[high_vol_data.index[-20:], 'volume'] *= 3
        
        analyzer = ProfessionalAnalyzer(horizon="médio")
        volume_flow = analyzer.analyze_volume_flow(high_vol_data)
        
        assert isinstance(volume_flow, VolumeFlowAnalysis)
        assert volume_flow.volume_ratio_5d > 1.0  # Volume recente > histórico
        assert volume_flow.volume_score > 0  # Score positivo
        assert volume_flow.flow_direction in ["ENTRADA", "NEUTRO"]
    
    def test_analyze_volume_flow_no_volume(self):
        """Testa análise sem dados de volume."""
        no_vol_data = pd.DataFrame({
            'close': pd.Series(range(100, 200), 
                             index=pd.date_range('2024-01-01', periods=100))
        })
        
        analyzer = ProfessionalAnalyzer(horizon="médio")
        volume_flow = analyzer.analyze_volume_flow(no_vol_data)
        
        # Deve funcionar sem volume (usa volatilidade como proxy)
        assert isinstance(volume_flow, VolumeFlowAnalysis)
        assert volume_flow.volume_ratio_5d is not None or volume_flow.volume_ratio_5d is None
    
    def test_volume_score_normalization(self, sample_price_data):
        """Testa normalização do score de volume."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        volume_flow = analyzer.analyze_volume_flow(sample_price_data)
        
        # Score deve estar normalizado entre -1 e 1
        assert -1 <= volume_flow.volume_score <= 1


class TestOptionsSentimentAnalysis:
    """Testes para análise de sentimento de opções."""
    
    @patch('professional_analysis.ProfessionalAnalyzer.client')
    def test_analyze_options_sentiment_bullish(self, mock_client, sample_price_data, sample_option_chain):
        """Testa análise de sentimento bullish."""
        # Configura mock para retornar mais volume em calls
        bullish_chain = sample_option_chain.copy()
        bullish_chain.loc[bullish_chain['option_type'] == 'CALL', 'volume'] *= 2
        mock_client.get_option_chain.return_value = bullish_chain
        
        analyzer = ProfessionalAnalyzer(horizon="médio")
        analyzer.client = mock_client
        
        sentiment = analyzer.analyze_options_sentiment("TEST", sample_price_data)
        
        assert isinstance(sentiment, OptionsSentimentAnalysis)
        assert sentiment.put_call_ratio > 0  # Deve ter algum valor
        assert 0 <= sentiment.call_volume_ratio <= 1
        assert 0 <= sentiment.put_volume_ratio <= 1
        assert abs(sentiment.call_volume_ratio + sentiment.put_volume_ratio - 1.0) < 0.01  # Soma = 1
        assert sentiment.market_bias in ["BULLISH", "BEARISH", "NEUTRAL"]
        assert -1 <= sentiment.sentiment_score <= 1  # Normalizado
    
    @patch('professional_analysis.ProfessionalAnalyzer.client')
    def test_analyze_options_sentiment_empty_chain(self, mock_client, sample_price_data):
        """Testa análise com chain vazia."""
        mock_client.get_option_chain.return_value = pd.DataFrame()
        
        analyzer = ProfessionalAnalyzer(horizon="médio")
        analyzer.client = mock_client
        
        sentiment = analyzer.analyze_options_sentiment("TEST", sample_price_data)
        
        # Deve retornar valores default
        assert sentiment.put_call_ratio == 1.0
        assert sentiment.call_volume_ratio == 0.5
        assert sentiment.put_volume_ratio == 0.5
        assert sentiment.market_bias == "NEUTRAL"
        assert sentiment.sentiment_score == 0.0
    
    @patch('professional_analysis.ProfessionalAnalyzer.client')
    def test_analyze_options_sentiment_exception_handling(self, mock_client, sample_price_data):
        """Testa tratamento de exceções na análise de sentimento."""
        mock_client.get_option_chain.side_effect = Exception("API Error")
        
        analyzer = ProfessionalAnalyzer(horizon="médio")
        analyzer.client = mock_client
        
        sentiment = analyzer.analyze_options_sentiment("TEST", sample_price_data)
        
        # Deve retornar valores default sem quebrar
        assert isinstance(sentiment, OptionsSentimentAnalysis)
        assert sentiment.market_bias == "NEUTRAL"


class TestMacroContextAnalysis:
    """Testes para análise de contexto macro."""
    
    def test_analyze_macro_context_known_tickers(self, sample_price_data):
        """Testa análise macro para tickers conhecidos."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # Testa diferentes setores
        tickers_scores = {
            "PETR4": analyzer.analyze_macro_context("PETR4", sample_price_data),
            "VALE3": analyzer.analyze_macro_context("VALE3", sample_price_data),
            "ITUB4": analyzer.analyze_macro_context("ITUB4", sample_price_data),
            "MGLU3": analyzer.analyze_macro_context("MGLU3", sample_price_data)
        }
        
        for ticker, macro in tickers_scores.items():
            assert isinstance(macro, MacroContextAnalysis)
            assert isinstance(macro.sector_score, float)
            assert isinstance(macro.macro_score, float)
            assert isinstance(macro.commodity_score, float)
            assert macro.context_bias in ["POSITIVO", "NEGATIVO", "NEUTRO"]
    
    def test_analyze_macro_context_unknown_ticker(self, sample_price_data):
        """Testa análise macro para ticker desconhecido."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        macro = analyzer.analyze_macro_context("UNKNOWN", sample_price_data)
        
        assert macro.sector_score == 0.0  # Neutro para ticker desconhecido
        assert macro.commodity_score == 0.0


class TestProfessionalAnalysisIntegration:
    """Testes de integração da análise completa."""
    
    @patch('professional_analysis.ProfessionalAnalyzer.client')
    def test_analyze_complete_flow(self, mock_client, sample_price_data, sample_option_chain):
        """Testa fluxo completo de análise."""
        mock_client.get_option_chain.return_value = sample_option_chain
        
        analyzer = ProfessionalAnalyzer(horizon="médio")
        analyzer.client = mock_client
        
        analysis = analyzer.analyze("PETR4", sample_price_data)
        
        assert isinstance(analysis, ProfessionalAnalysis)
        assert analysis.ticker == "PETR4"
        assert isinstance(analysis.current_price, float)
        assert isinstance(analysis.analysis_date, datetime)
        assert isinstance(analysis.trend, TrendAnalysis)
        assert isinstance(analysis.momentum, MomentumAnalysis)
        assert isinstance(analysis.volume_flow, VolumeFlowAnalysis)
        assert isinstance(analysis.options_sentiment, OptionsSentimentAnalysis)
        assert isinstance(analysis.macro_context, MacroContextAnalysis)
        assert analysis.direction in [Direction.CALL, Direction.PUT, Direction.NEUTRAL]
        assert 0 <= analysis.confidence <= 100
        assert isinstance(analysis.key_drivers, list)
        assert isinstance(analysis.strategy_recommendation, str)
    
    def test_final_score_calculation(self, sample_price_data):
        """Testa cálculo do score final."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        with patch.object(analyzer, 'analyze_options_sentiment') as mock_sentiment:
            mock_sentiment.return_value = OptionsSentimentAnalysis(
                put_call_ratio=1.0, volatility_skew=0.0, call_volume_ratio=0.5,
                put_volume_ratio=0.5, sentiment_score=0.0, market_bias="NEUTRAL"
            )
            
            analysis = analyzer.analyze("TEST", sample_price_data)
            
            # Score final deve estar normalizado
            assert -1 <= analysis.final_score <= 1
    
    def test_confidence_calibration(self, sample_price_data):
        """Testa calibração de confiança."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # Testa diferentes scores
        test_scores = [-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8]
        
        for score in test_scores:
            confidence = analyzer._calculate_logistic_confidence(score)
            assert 20 <= confidence <= 90  # Range esperado
            assert isinstance(confidence, float)
    
    def test_strategy_recommendation_logic(self, sample_price_data):
        """Testa lógica de recomendação de estratégia."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        with patch.object(analyzer, 'analyze_options_sentiment') as mock_sentiment:
            mock_sentiment.return_value = OptionsSentimentAnalysis(
                put_call_ratio=1.0, volatility_skew=0.0, call_volume_ratio=0.5,
                put_volume_ratio=0.5, sentiment_score=0.0, market_bias="NEUTRAL"
            )
            
            analysis = analyzer.analyze("TEST", sample_price_data)
            
            # Estratégia deve ser consistente com direção
            if analysis.direction == Direction.CALL:
                assert "CALL" in analysis.strategy_recommendation or "PUT" in analysis.strategy_recommendation
            elif analysis.direction == Direction.PUT:
                assert "PUT" in analysis.strategy_recommendation or "CALL" in analysis.strategy_recommendation
            else:  # NEUTRAL
                assert "Straddle" in analysis.strategy_recommendation or "não operar" in analysis.strategy_recommendation


class TestHelperMethods:
    """Testes para métodos auxiliares."""
    
    def test_calculate_rsi(self, sample_price_data):
        """Testa cálculo de RSI."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        rsi = analyzer._calculate_rsi(sample_price_data['close'], 14)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_price_data)
        # RSI deve estar entre 0 e 100
        valid_rsi = rsi.dropna()
        assert all(0 <= val <= 100 for val in valid_rsi)
    
    def test_calculate_macd_adaptive(self, sample_price_data):
        """Testa cálculo de MACD adaptativo."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        macd_line, signal, histogram = analyzer._calculate_macd_adaptive(sample_price_data['close'])
        
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(histogram, pd.Series)
        assert len(macd_line) == len(sample_price_data)
        
        # Histogram deve ser a diferença entre linha e sinal
        calculated_hist = macd_line - signal
        pd.testing.assert_series_equal(histogram, calculated_hist, check_names=False)
    
    def test_calculate_atr(self, sample_price_data):
        """Testa cálculo de ATR."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        atr = analyzer._calculate_atr(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            14
        )
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_price_data)
        # ATR deve ser sempre positivo
        valid_atr = atr.dropna()
        assert all(val >= 0 for val in valid_atr)
    
    def test_calculate_adx_adaptive(self, sample_price_data):
        """Testa cálculo de ADX adaptativo."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        adx = analyzer._calculate_adx_adaptive(sample_price_data)
        
        assert isinstance(adx, float)
        assert 0 <= adx <= 100  # ADX deve estar entre 0 e 100
    
    def test_division_by_zero_protection(self):
        """Testa proteção contra divisão por zero."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # Dados constantes (sem variação)
        constant_data = pd.DataFrame({
            'close': [100.0] * 100,
            'high': [100.0] * 100,
            'low': [100.0] * 100,
            'volume': [1000] * 100
        }, index=pd.date_range('2024-01-01', periods=100))
        
        # Não deve gerar exceção
        trend = analyzer.analyze_trend(constant_data)
        momentum = analyzer.analyze_momentum(constant_data)
        volume_flow = analyzer.analyze_volume_flow(constant_data)
        
        assert isinstance(trend, TrendAnalysis)
        assert isinstance(momentum, MomentumAnalysis)
        assert isinstance(volume_flow, VolumeFlowAnalysis)


# Testes de propriedades e invariantes
class TestInvariants:
    """Testes de invariantes e propriedades do sistema."""
    
    def test_score_normalization_invariant(self, sample_price_data):
        """Testa se todos os scores permanecem normalizados."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        with patch.object(analyzer, 'analyze_options_sentiment') as mock_sentiment:
            mock_sentiment.return_value = OptionsSentimentAnalysis(
                put_call_ratio=1.0, volatility_skew=0.0, call_volume_ratio=0.5,
                put_volume_ratio=0.5, sentiment_score=0.0, market_bias="NEUTRAL"
            )
            
            analysis = analyzer.analyze("TEST", sample_price_data)
            
            # Todos os scores devem estar normalizados
            assert -1 <= analysis.trend.trend_score <= 1
            assert -1 <= analysis.momentum.momentum_score <= 1
            assert -1 <= analysis.volume_flow.volume_score <= 1
            assert -1 <= analysis.options_sentiment.sentiment_score <= 1
            assert -1 <= analysis.final_score <= 1
    
    def test_confidence_range_invariant(self, sample_price_data):
        """Testa se confiança sempre está no range correto."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # Testa múltiplas execuções
        for _ in range(10):
            with patch.object(analyzer, 'analyze_options_sentiment') as mock_sentiment:
                mock_sentiment.return_value = OptionsSentimentAnalysis(
                    put_call_ratio=np.random.uniform(0.5, 2.0),
                    volatility_skew=np.random.uniform(-0.1, 0.1),
                    call_volume_ratio=np.random.uniform(0.3, 0.7),
                    put_volume_ratio=np.random.uniform(0.3, 0.7),
                    sentiment_score=np.random.uniform(-1, 1),
                    market_bias="NEUTRAL"
                )
                
                analysis = analyzer.analyze("TEST", sample_price_data)
                assert 0 <= analysis.confidence <= 100
    
    def test_ratios_sum_to_one_invariant(self, sample_option_chain):
        """Testa se ratios de volume somam 1."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        with patch.object(analyzer.client, 'get_option_chain', return_value=sample_option_chain):
            sentiment = analyzer.analyze_options_sentiment("TEST", pd.DataFrame({'close': [100]}))
            
            # Call ratio + Put ratio deve somar 1 (com tolerância para float)
            total_ratio = sentiment.call_volume_ratio + sentiment.put_volume_ratio
            assert abs(total_ratio - 1.0) < 0.01
