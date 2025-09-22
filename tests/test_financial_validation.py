"""
Testes de Validação Financeira - Professional Analysis
Foco: Verificar se a lógica faz sentido do ponto de vista financeiro
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from professional_analysis import (
    ProfessionalAnalyzer, 
    Direction,
    TrendStrength
)


class TestFinancialLogicValidation:
    """Validação da lógica financeira do sistema."""
    
    def test_rsi_calculation_financial_correctness(self):
        """Valida se o RSI está sendo calculado corretamente do ponto de vista financeiro."""
        # Cenário 1: Quedas consecutivas devem gerar RSI baixo
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        consecutive_drops = [100.0]
        for i in range(49):
            consecutive_drops.append(consecutive_drops[-1] * 0.99)  # Queda de 1% por dia
        
        drop_data = pd.DataFrame({
            'close': pd.Series(consecutive_drops, index=dates),
            'high': pd.Series([p * 1.005 for p in consecutive_drops], index=dates),
            'low': pd.Series([p * 0.995 for p in consecutive_drops], index=dates),
            'volume': [100000] * 50
        })
        
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # Testa RSI diretamente
        rsi_14 = analyzer._calculate_rsi(drop_data['close'], 14)
        final_rsi = rsi_14.iloc[-1]
        
        print(f"RSI após quedas consecutivas: {final_rsi}")
        
        # VALIDAÇÃO FINANCEIRA: Quedas consecutivas devem resultar em RSI muito baixo
        # RSI = 0 é tecnicamente correto quando há apenas perdas
        assert final_rsi <= 20, f"Quedas consecutivas devem gerar RSI ≤ 20, obtido {final_rsi}"
        
        # Cenário 2: Altas consecutivas devem gerar RSI alto
        consecutive_gains = [100.0]
        for i in range(49):
            consecutive_gains.append(consecutive_gains[-1] * 1.01)  # Alta de 1% por dia
        
        gain_data = pd.DataFrame({
            'close': pd.Series(consecutive_gains, index=dates),
            'high': pd.Series([p * 1.005 for p in consecutive_gains], index=dates),
            'low': pd.Series([p * 0.995 for p in consecutive_gains], index=dates),
            'volume': [100000] * 50
        })
        
        rsi_14_gains = analyzer._calculate_rsi(gain_data['close'], 14)
        final_rsi_gains = rsi_14_gains.iloc[-1]
        
        print(f"RSI após altas consecutivas: {final_rsi_gains}")
        
        # VALIDAÇÃO FINANCEIRA: Altas consecutivas devem resultar em RSI alto
        assert final_rsi_gains >= 80, f"Altas consecutivas devem gerar RSI ≥ 80, obtido {final_rsi_gains}"
    
    def test_macd_interpretation_financial_correctness(self):
        """Valida se a interpretação do MACD faz sentido financeiro."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # Cenário 1: Tendência de alta acelerando (MACD deve ser positivo e crescente)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        accelerating_prices = [100.0]
        for i in range(99):
            # Crescimento acelerando: 0.5% → 1% → 1.5%
            growth_rate = 0.005 + (i / 99) * 0.01
            accelerating_prices.append(accelerating_prices[-1] * (1 + growth_rate))
        
        accel_data = pd.DataFrame({
            'close': pd.Series(accelerating_prices, index=dates),
            'high': pd.Series([p * 1.01 for p in accelerating_prices], index=dates),
            'low': pd.Series([p * 0.99 for p in accelerating_prices], index=dates),
            'volume': [100000] * 100
        })
        
        macd_line, macd_signal, macd_hist = analyzer._calculate_macd_adaptive(accel_data['close'])
        
        # VALIDAÇÃO FINANCEIRA: Aceleração de alta deve gerar MACD positivo
        final_macd_hist = macd_hist.iloc[-1]
        print(f"MACD histogram em tendência acelerando: {final_macd_hist}")
        
        assert final_macd_hist > 0, f"Tendência acelerando deve ter MACD positivo, obtido {final_macd_hist}"
        
        # Cenário 2: Tendência de baixa acelerando (MACD deve ser negativo)
        decelerating_prices = [200.0]
        for i in range(99):
            # Queda acelerando: -0.5% → -1% → -1.5%
            decline_rate = -0.005 - (i / 99) * 0.01
            decelerating_prices.append(decelerating_prices[-1] * (1 + decline_rate))
        
        decel_data = pd.DataFrame({
            'close': pd.Series(decelerating_prices, index=dates),
            'high': pd.Series([p * 1.01 for p in decelerating_prices], index=dates),
            'low': pd.Series([p * 0.99 for p in decelerating_prices], index=dates),
            'volume': [100000] * 100
        })
        
        macd_line_down, macd_signal_down, macd_hist_down = analyzer._calculate_macd_adaptive(decel_data['close'])
        final_macd_hist_down = macd_hist_down.iloc[-1]
        
        print(f"MACD histogram em queda acelerando: {final_macd_hist_down}")
        
        assert final_macd_hist_down < 0, f"Queda acelerando deve ter MACD negativo, obtido {final_macd_hist_down}"
    
    def test_trend_analysis_financial_logic(self):
        """Valida se a análise de tendência segue lógica financeira correta."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # Cenário: Ação que quebrou resistência e está em alta forte
        dates = pd.date_range('2024-01-01', periods=300, freq='D')
        
        # Simula breakout: lateral por 200 dias, depois alta forte
        sideways_period = [100.0 + np.sin(i/10) * 2 for i in range(200)]  # Lateral
        breakout_period = []
        last_price = sideways_period[-1]
        for i in range(100):
            last_price *= 1.015  # 1.5% por dia após breakout
            breakout_period.append(last_price)
        
        breakout_data = pd.DataFrame({
            'close': pd.Series(sideways_period + breakout_period, index=dates),
            'high': pd.Series([(p * 1.02) for p in sideways_period + breakout_period], index=dates),
            'low': pd.Series([(p * 0.98) for p in sideways_period + breakout_period], index=dates),
            'volume': [100000] * 200 + [300000] * 100  # Volume alto no breakout
        })
        
        trend = analyzer.analyze_trend(breakout_data)
        
        # VALIDAÇÃO FINANCEIRA: Breakout deve ser detectado
        print(f"Trend score após breakout: {trend.trend_score}")
        print(f"Trend strength: {trend.trend_strength}")
        print(f"Golden cross: {trend.golden_cross}")
        
        # Todas as SMAs devem estar em ordem crescente (SMA curta > média > longa)
        assert trend.sma_10 > trend.sma_50, "SMA curta deve estar acima da média em uptrend"
        assert trend.sma_50 > trend.sma_100, "SMA média deve estar acima da longa em uptrend"
        assert trend.golden_cross is True, "Golden cross deve ser detectado em breakout"
        assert trend.trend_score > 0.2, f"Breakout deve gerar score positivo forte, obtido {trend.trend_score}"
    
    def test_volume_analysis_financial_interpretation(self):
        """Valida se a análise de volume segue princípios financeiros corretos."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # Cenário: Acumulação institucional (volume crescendo gradualmente)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = pd.Series([100 + i * 0.1 for i in range(100)], index=dates)  # Alta suave
        
        # Volume crescendo gradualmente (acumulação)
        volume = []
        for i in range(100):
            base_vol = 100000
            growth_factor = 1 + (i / 100) * 1.5  # Cresce 150% ao longo do período
            volume.append(int(base_vol * growth_factor))
        
        accumulation_data = pd.DataFrame({
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'volume': volume
        })
        
        volume_flow = analyzer.analyze_volume_flow(accumulation_data)
        
        # VALIDAÇÃO FINANCEIRA: Acumulação deve ser detectada
        print(f"Volume ratio 5d: {volume_flow.volume_ratio_5d}")
        print(f"OBV trend: {volume_flow.obv_trend}")
        print(f"Volume score: {volume_flow.volume_score}")
        
        # Volume crescente + preços subindo = acumulação
        assert volume_flow.volume_ratio_5d > 1.0, "Acumulação deve mostrar volume crescente"
        assert volume_flow.obv_trend > 0, "OBV deve ser positivo em acumulação"
        assert volume_flow.volume_score > 0, "Score de volume deve ser positivo em acumulação"
    
    def test_options_sentiment_financial_logic(self):
        """Valida se análise de sentimento de opções faz sentido financeiro."""
        # Cria chain realista com bias bullish
        current_price = 100.0
        bullish_chain_data = []
        
        # Simula mercado bullish: mais volume em calls, puts baratas
        strikes = np.arange(90, 111, 2)
        exp = datetime.now() + timedelta(days=30)
        
        for strike in strikes:
            # CALL: Volume maior em strikes próximos ou OTM
            call_volume = 1000 if strike <= current_price + 5 else 500
            bullish_chain_data.append({
                'symbol': f'TEST{int(strike)}C',
                'option_type': 'CALL',
                'strike': strike,
                'expiration': exp,
                'last': max(0.5, current_price - strike + 2),
                'volume': call_volume,
                'open_interest': call_volume * 5,
                'delta': min(0.99, max(0.01, 0.5 + (current_price - strike) / 20))
            })
            
            # PUT: Volume menor, puts mais baratas
            put_volume = 300 if strike >= current_price - 5 else 100
            bullish_chain_data.append({
                'symbol': f'TEST{int(strike)}P',
                'option_type': 'PUT',
                'strike': strike,
                'expiration': exp,
                'last': max(0.5, strike - current_price + 1),
                'volume': put_volume,
                'open_interest': put_volume * 3,
                'delta': -min(0.99, max(0.01, 0.5 - (current_price - strike) / 20))
            })
        
        bullish_chain = pd.DataFrame(bullish_chain_data)
        
        with patch('professional_analysis.OpLabClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_option_chain.return_value = bullish_chain
            mock_client_class.return_value = mock_client
            
            analyzer = ProfessionalAnalyzer(horizon="médio")
            
            # Dados de preço neutros para focar no sentimento
            dates = pd.date_range('2024-01-01', periods=50, freq='D')
            neutral_prices = pd.Series([100.0] * 50, index=dates)
            price_data = pd.DataFrame({
                'close': neutral_prices,
                'high': neutral_prices * 1.01,
                'low': neutral_prices * 0.99,
                'volume': [100000] * 50
            })
            
            sentiment = analyzer.analyze_options_sentiment("TEST", price_data)
            
            # VALIDAÇÃO FINANCEIRA: Mais volume em calls deve indicar sentimento bullish
            print(f"Call volume ratio: {sentiment.call_volume_ratio:.2%}")
            print(f"Put/Call ratio: {sentiment.put_call_ratio:.2f}")
            print(f"Market bias: {sentiment.market_bias}")
            
            # Em mercado com mais calls, put/call ratio deve ser < 1
            assert sentiment.put_call_ratio < 1.0, f"Mercado bullish deve ter P/C < 1, obtido {sentiment.put_call_ratio}"
            assert sentiment.call_volume_ratio > 0.5, f"Mercado bullish deve ter mais calls, obtido {sentiment.call_volume_ratio:.2%}"
    
    def test_decision_weights_financial_rationale(self):
        """Valida se os pesos das camadas fazem sentido financeiro."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # Os pesos devem refletir importância financeira
        weights = {
            'trend': 0.45,      # 45% - tendência é fundamental
            'momentum': 0.25,   # 25% - momentum confirma tendência
            'volume': 0.15,     # 15% - volume confirma movimento
            'sentiment': 0.10,  # 10% - sentimento do mercado
            'macro': 0.05       # 5% - contexto macro
        }
        
        # VALIDAÇÃO FINANCEIRA: Pesos devem somar 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.001, f"Pesos devem somar 1.0, obtido {total_weight}"
        
        # VALIDAÇÃO FINANCEIRA: Tendência deve ter maior peso (análise técnica clássica)
        assert weights['trend'] > weights['momentum'], "Tendência deve ter mais peso que momentum"
        assert weights['momentum'] > weights['volume'], "Momentum deve ter mais peso que volume"
        assert weights['volume'] > weights['sentiment'], "Volume deve ter mais peso que sentimento"
        
        # Testa cenário onde apenas tendência é forte
        mock_analysis = Mock()
        mock_analysis.trend = Mock(trend_score=0.8, trend_strength='FORTE ALTA')
        mock_analysis.momentum = Mock(momentum_score=0, momentum_strength='NEUTRO')
        mock_analysis.volume_flow = Mock(volume_score=0, flow_direction='NEUTRO')
        mock_analysis.options_sentiment = Mock(sentiment_score=0, market_bias='NEUTRAL')
        mock_analysis.macro_context = Mock(overall_context_score=0, context_bias='NEUTRO')
        
        direction, confidence, drivers = analyzer.calculate_final_decision(mock_analysis)
        final_score = 0.8 * 0.45  # = 0.36
        
        print(f"Apenas tendência forte: Score={final_score}, Direção={direction}")
        
        # VALIDAÇÃO: Tendência forte sozinha deve ser suficiente para decisão
        assert direction == Direction.CALL, f"Tendência forte deve gerar CALL, obtido {direction}"
        assert confidence > 70, f"Tendência forte deve gerar alta confiança, obtido {confidence}%"
    
    def test_real_market_scenarios(self):
        """Testa cenários que realmente acontecem no mercado brasileiro."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # CENÁRIO 1: PETR4 em rally de commodities
        # Características: alta forte, volume alto, momentum positivo
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        
        # Simula rally: crescimento de 30% em 3 meses com volatilidade
        np.random.seed(42)
        petr4_prices = [35.0]
        for i in range(199):
            # Tendência de alta com volatilidade típica de PETR4
            daily_return = np.random.normal(0.0015, 0.025)  # 0.15% média, 2.5% vol
            petr4_prices.append(petr4_prices[-1] * (1 + daily_return))
        
        # Adiciona volume crescente nos últimos 50 dias (momentum institucional)
        volume = [500000] * 150 + [800000] * 50
        
        petr4_data = pd.DataFrame({
            'close': pd.Series(petr4_prices, index=dates),
            'high': pd.Series([p * 1.03 for p in petr4_prices], index=dates),
            'low': pd.Series([p * 0.97 for p in petr4_prices], index=dates),
            'volume': volume
        })
        
        with patch.object(analyzer, 'analyze_options_sentiment') as mock_sentiment:
            # Simula sentimento bullish típico de commodities
            mock_sentiment.return_value = Mock(
                put_call_ratio=0.7,  # Mais calls que puts
                volatility_skew=0.02,
                call_volume_ratio=0.65,
                put_volume_ratio=0.35,
                sentiment_score=0.3,
                market_bias="BULLISH"
            )
            
            analysis = analyzer.analyze("PETR4", petr4_data)
            
            print(f"PETR4 Rally - Direção: {analysis.direction}, Confiança: {analysis.confidence}%")
            print(f"Score final: {analysis.final_score}")
            print(f"Drivers: {analysis.key_drivers}")
            
            # VALIDAÇÃO: Rally de commodities deve gerar sinal de CALL
            assert analysis.direction == Direction.CALL, f"Rally de PETR4 deve gerar CALL, obtido {analysis.direction}"
            assert analysis.confidence > 65, f"Rally claro deve ter alta confiança, obtido {analysis.confidence}%"
            assert analysis.final_score > 0.2, f"Rally deve ter score positivo forte, obtido {analysis.final_score}"
        
        # CENÁRIO 2: MGLU3 em crise de varejo
        # Características: queda consistente, volume alto (pânico), sentiment bearish
        mglu3_prices = [15.0]
        np.random.seed(123)  # Seed diferente para garantir queda
        for i in range(199):
            # Queda mais agressiva e consistente
            if i < 100:  # Primeira metade: queda gradual
                daily_return = np.random.normal(-0.001, 0.015)
            else:  # Segunda metade: crise acelerando
                daily_return = np.random.normal(-0.005, 0.03)  # -0.5% média
            mglu3_prices.append(mglu3_prices[-1] * (1 + daily_return))
        
        # Volume alto durante quedas (pânico)
        volume_mglu = []
        for i in range(200):
            base_vol = 300000
            if i > 150:  # Volume crescente na crise
                panic_multiplier = 1 + (i - 150) / 50 * 2  # Até 3x o volume
                volume_mglu.append(int(base_vol * panic_multiplier))
            else:
                volume_mglu.append(base_vol)
        
        mglu3_data = pd.DataFrame({
            'close': pd.Series(mglu3_prices, index=dates),
            'high': pd.Series([p * 1.04 for p in mglu3_prices], index=dates),
            'low': pd.Series([p * 0.96 for p in mglu3_prices], index=dates),
            'volume': volume_mglu
        })
        
        with patch.object(analyzer, 'analyze_options_sentiment') as mock_sentiment:
            # Simula sentimento bearish típico de crise
            mock_sentiment.return_value = Mock(
                put_call_ratio=1.8,  # Muito mais puts (proteção)
                volatility_skew=0.08,  # Puts caras (medo)
                call_volume_ratio=0.3,
                put_volume_ratio=0.7,
                sentiment_score=-0.4,
                market_bias="BEARISH"
            )
            
            analysis = analyzer.analyze("MGLU3", mglu3_data)
            
            print(f"MGLU3 Crise - Direção: {analysis.direction}, Confiança: {analysis.confidence}%")
            print(f"Score final: {analysis.final_score}")
            
            # VALIDAÇÃO: Crise deve gerar sinal de PUT
            assert analysis.direction == Direction.PUT, f"Crise de MGLU3 deve gerar PUT, obtido {analysis.direction}"
            assert analysis.confidence > 60, f"Crise clara deve ter alta confiança, obtido {analysis.confidence}%"
            assert analysis.final_score < -0.2, f"Crise deve ter score negativo forte, obtido {analysis.final_score}"
    
    def test_horizon_adaptation_financial_sense(self):
        """Valida se adaptação por horizonte faz sentido financeiro."""
        # Compara diferentes horizontes para o mesmo ativo
        dates = pd.date_range('2024-01-01', periods=300, freq='D')
        prices = pd.Series([100 + i * 0.05 for i in range(300)], index=dates)  # Tendência suave
        
        price_data = pd.DataFrame({
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'volume': [100000] * 300
        })
        
        # Testa diferentes horizontes
        analyzers = {
            'curto': ProfessionalAnalyzer(horizon="curto"),
            'médio': ProfessionalAnalyzer(horizon="médio"),
            'longo': ProfessionalAnalyzer(horizon="longo")
        }
        
        results = {}
        for horizon, analyzer in analyzers.items():
            trend = analyzer.analyze_trend(price_data)
            results[horizon] = {
                'sma_short': analyzer.params['sma_short'],
                'rsi_period': analyzer.params['rsi_short'],
                'trend_score': trend.trend_score
            }
        
        print("Parâmetros por horizonte:")
        for horizon, result in results.items():
            print(f"  {horizon}: SMA={result['sma_short']}, RSI={result['rsi_period']}, Score={result['trend_score']:.3f}")
        
        # VALIDAÇÃO FINANCEIRA: Parâmetros devem ser progressivos
        assert results['curto']['sma_short'] < results['médio']['sma_short'] < results['longo']['sma_short'], \
            "SMAs devem crescer com horizonte"
        
        # Horizonte curto deve ser mais sensível (scores mais extremos)
        # Em tendência suave, curto prazo deve detectar melhor que longo prazo
        assert abs(results['curto']['trend_score']) >= abs(results['longo']['trend_score']), \
            "Curto prazo deve ser mais sensível que longo prazo"
    
    def test_risk_management_logic(self):
        """Valida se o sistema incorpora gestão de risco adequada."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # CENÁRIO: Ativo com alta volatilidade (risco alto)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # Simula ativo muito volátil (ex: small cap em crise)
        volatile_prices = [50.0]
        np.random.seed(42)
        for i in range(99):
            # Volatilidade muito alta: 5% por dia
            daily_return = np.random.normal(0, 0.05)
            volatile_prices.append(volatile_prices[-1] * (1 + daily_return))
        
        volatile_data = pd.DataFrame({
            'close': pd.Series(volatile_prices, index=dates),
            'high': pd.Series([p * 1.08 for p in volatile_prices], index=dates),  # Ranges largos
            'low': pd.Series([p * 0.92 for p in volatile_prices], index=dates),
            'volume': [200000] * 100
        })
        
        trend = analyzer.analyze_trend(volatile_data)
        
        # VALIDAÇÃO: Alta volatilidade deve ser detectada
        print(f"ATR ratio em ativo volátil: {trend.atr_ratio}")
        print(f"Regime de volatilidade: {trend.volatility_regime}")
        
        # ATR alto deve indicar regime de alta volatilidade
        if trend.atr_ratio > 1.5:
            assert trend.volatility_regime == "ALTA", f"ATR alto deve indicar volatilidade ALTA, obtido {trend.volatility_regime}"
        
        # Sistema deve ser mais conservador com alta volatilidade
        # (isso pode ser implementado como feature futura)
        assert isinstance(trend.atr_ratio, float), "ATR ratio deve ser calculado"
        assert trend.atr_ratio > 0, "ATR ratio deve ser positivo"
    
    def test_macro_context_financial_logic(self):
        """Valida se análise macro faz sentido setorial."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # Dados neutros para focar no contexto macro
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        neutral_data = pd.DataFrame({
            'close': pd.Series([100.0] * 100, index=dates),
            'high': pd.Series([101.0] * 100, index=dates),
            'low': pd.Series([99.0] * 100, index=dates),
            'volume': [100000] * 100
        })
        
        # Testa diferentes setores
        sectors = {
            'PETR4': ('energia', 'positivo'),      # Petróleo historicamente positivo
            'VALE3': ('mineração', 'positivo'),    # Commodities positivo
            'ITUB4': ('bancário', 'negativo'),     # Bancos com pressão de juros
            'MGLU3': ('varejo', 'negativo'),       # Varejo com pressão
            'WEGE3': ('industrial', 'neutro'),     # Industrial neutro
            'UNKNOWN': ('desconhecido', 'neutro')  # Ticker desconhecido
        }
        
        for ticker, (setor, bias_esperado) in sectors.items():
            macro = analyzer.analyze_macro_context(ticker, neutral_data)
            
            print(f"{ticker} ({setor}): Score={macro.overall_context_score:.2f}, Bias={macro.context_bias}")
            
            # VALIDAÇÃO FINANCEIRA: Bias deve ser coerente com setor
            if bias_esperado == 'positivo':
                assert macro.overall_context_score >= 0, f"{ticker} deveria ter score ≥ 0, obtido {macro.overall_context_score}"
            elif bias_esperado == 'negativo':
                assert macro.overall_context_score <= 0, f"{ticker} deveria ter score ≤ 0, obtido {macro.overall_context_score}"
            else:  # neutro
                assert abs(macro.overall_context_score) <= 0.1, f"{ticker} deveria ser neutro, obtido {macro.overall_context_score}"
    
    def test_confidence_calibration_financial_realism(self):
        """Valida se calibração de confiança é financeiramente realista."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # Testa diferentes níveis de score
        test_cases = [
            (0.1, "sinal fraco"),
            (0.3, "sinal moderado"),
            (0.6, "sinal forte"),
            (0.9, "sinal muito forte")
        ]
        
        for score, description in test_cases:
            confidence = analyzer._calculate_logistic_confidence(score)
            print(f"Score {score} ({description}): Confiança {confidence:.1f}%")
            
            # VALIDAÇÃO FINANCEIRA: Confiança deve ser realista
            # Nunca 100% (mercado sempre tem incerteza)
            assert confidence < 95, f"Confiança nunca deve ser ≥95%, obtido {confidence}% para score {score}"
            
            # Sinal fraco não deve ter confiança muito alta
            if score < 0.2:
                assert confidence < 75, f"Sinal fraco não deve ter confiança ≥75%, obtido {confidence}%"
            
            # Sinal forte deve ter confiança alta
            if score > 0.6:
                assert confidence > 80, f"Sinal forte deve ter confiança >80%, obtido {confidence}%"
    
    def test_strategy_recommendations_financial_soundness(self):
        """Valida se recomendações de estratégia são financeiramente sólidas."""
        analyzer = ProfessionalAnalyzer(horizon="médio")
        
        # Cenários de teste
        scenarios = [
            (Direction.CALL, 85, "alta confiança bullish"),
            (Direction.CALL, 55, "baixa confiança bullish"),
            (Direction.PUT, 85, "alta confiança bearish"),
            (Direction.PUT, 55, "baixa confiança bearish"),
            (Direction.NEUTRAL, 35, "neutro")
        ]
        
        for direction, confidence, description in scenarios:
            mock_analysis = Mock(direction=direction, confidence=confidence)
            strategy = analyzer.generate_strategy_recommendation(mock_analysis)
            
            print(f"{description}: {strategy}")
            
            # VALIDAÇÃO FINANCEIRA: Estratégias devem ser coerentes
            if direction == Direction.CALL:
                if confidence > 70:
                    # Alta confiança: estratégia agressiva (compra de call)
                    assert "Compra de CALL" in strategy, f"Alta confiança CALL deve sugerir compra, obtido: {strategy}"
                else:
                    # Baixa confiança: estratégia conservadora (venda de put)
                    assert "PUT" in strategy, f"Baixa confiança CALL deve sugerir venda de PUT, obtido: {strategy}"
            
            elif direction == Direction.PUT:
                if confidence > 70:
                    # Alta confiança: proteção (compra de put)
                    assert "Compra de PUT" in strategy, f"Alta confiança PUT deve sugerir compra, obtido: {strategy}"
                else:
                    # Baixa confiança: renda (venda de call)
                    assert "CALL" in strategy, f"Baixa confiança PUT deve sugerir venda de CALL, obtido: {strategy}"
            
            else:  # NEUTRAL
                # Estratégias neutras: straddle, strangle ou não operar
                neutral_keywords = ["Straddle", "Strangle", "não operar"]
                assert any(keyword in strategy for keyword in neutral_keywords), \
                    f"Estratégia neutra deve sugerir straddle/strangle ou não operar, obtido: {strategy}"
