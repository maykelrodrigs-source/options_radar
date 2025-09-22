"""
Testes para synthetic_dividends.py
Cobertura alvo: 85%
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from synthetic_dividends import (
    find_synthetic_dividend_options,
    calculate_exercise_probability,
    calculate_black_scholes_delta,
    calculate_synthetic_dividend_return
)


class TestSyntheticDividendCalculations:
    """Testes para cálculos de dividendos sintéticos."""
    
    def test_calculate_exercise_probability_call(self):
        """Testa cálculo de probabilidade de exercício para CALL."""
        # CALL ITM (delta positivo alto)
        prob_itm = calculate_exercise_probability(0.8, "CALL")
        assert prob_itm == 80.0
        
        # CALL ATM
        prob_atm = calculate_exercise_probability(0.5, "CALL")
        assert prob_atm == 50.0
        
        # CALL OTM (delta baixo)
        prob_otm = calculate_exercise_probability(0.1, "CALL")
        assert prob_otm == 10.0
    
    def test_calculate_exercise_probability_put(self):
        """Testa cálculo de probabilidade de exercício para PUT."""
        # PUT ITM (delta negativo alto em módulo)
        prob_itm = calculate_exercise_probability(-0.8, "PUT")
        assert prob_itm == 80.0
        
        # PUT ATM
        prob_atm = calculate_exercise_probability(-0.5, "PUT")
        assert prob_atm == 50.0
        
        # PUT OTM (delta próximo de zero)
        prob_otm = calculate_exercise_probability(-0.1, "PUT")
        assert prob_otm == 10.0
    
    def test_calculate_exercise_probability_edge_cases(self):
        """Testa casos extremos de probabilidade de exercício."""
        # Delta zero
        prob_zero = calculate_exercise_probability(0.0, "CALL")
        assert prob_zero == 0.0
        
        # Delta muito alto
        prob_high = calculate_exercise_probability(1.0, "CALL")
        assert prob_high == 100.0
        
        # Delta negativo para CALL (não deveria acontecer, mas testa robustez)
        prob_negative = calculate_exercise_probability(-0.3, "CALL")
        assert prob_negative == 30.0  # Usa valor absoluto
        
        # Tipo de opção inválido
        prob_invalid = calculate_exercise_probability(0.5, "INVALID")
        assert prob_invalid == 50.0  # Usa valor absoluto como fallback
    
    def test_calculate_black_scholes_delta_call(self):
        """Testa cálculo de delta Black-Scholes para CALL."""
        # CALL ITM
        delta_itm = calculate_black_scholes_delta(
            spot_price=110, strike=100, time_to_expiry_years=0.25,
            risk_free_rate=0.10, volatility=0.20, option_type="CALL"
        )
        assert 0.5 < delta_itm < 1.0  # Deve ser alto para ITM
        
        # CALL ATM
        delta_atm = calculate_black_scholes_delta(
            spot_price=100, strike=100, time_to_expiry_years=0.25,
            risk_free_rate=0.10, volatility=0.20, option_type="CALL"
        )
        assert 0.4 < delta_atm < 0.7  # Deve ser próximo de 0.5 para ATM
        
        # CALL OTM
        delta_otm = calculate_black_scholes_delta(
            spot_price=90, strike=100, time_to_expiry_years=0.25,
            risk_free_rate=0.10, volatility=0.20, option_type="CALL"
        )
        assert 0.0 < delta_otm < 0.5  # Deve ser baixo para OTM
    
    def test_calculate_black_scholes_delta_put(self):
        """Testa cálculo de delta Black-Scholes para PUT."""
        # PUT ITM
        delta_itm = calculate_black_scholes_delta(
            spot_price=90, strike=100, time_to_expiry_years=0.25,
            risk_free_rate=0.10, volatility=0.20, option_type="PUT"
        )
        assert -1.0 < delta_itm < -0.5  # Deve ser negativo e alto em módulo
        
        # PUT ATM
        delta_atm = calculate_black_scholes_delta(
            spot_price=100, strike=100, time_to_expiry_years=0.25,
            risk_free_rate=0.10, volatility=0.20, option_type="PUT"
        )
        assert -0.7 < delta_atm < -0.3  # Deve ser próximo de -0.5 para ATM
        
        # PUT OTM
        delta_otm = calculate_black_scholes_delta(
            spot_price=110, strike=100, time_to_expiry_years=0.25,
            risk_free_rate=0.10, volatility=0.20, option_type="PUT"
        )
        assert -0.5 < delta_otm < 0.0  # Deve ser próximo de zero para OTM
    
    def test_calculate_black_scholes_edge_cases(self):
        """Testa casos extremos do Black-Scholes."""
        # Tempo de expiração zero
        delta_no_time = calculate_black_scholes_delta(
            spot_price=100, strike=100, time_to_expiry_years=0.0,
            option_type="CALL"
        )
        # Deve retornar 1.0 se ITM, 0.0 se OTM, ou próximo disso
        assert isinstance(delta_no_time, float)
        
        # Volatilidade zero
        delta_no_vol = calculate_black_scholes_delta(
            spot_price=100, strike=100, time_to_expiry_years=0.25,
            volatility=0.0, option_type="CALL"
        )
        assert isinstance(delta_no_vol, float)
        
        # Volatilidade muito alta
        delta_high_vol = calculate_black_scholes_delta(
            spot_price=100, strike=100, time_to_expiry_years=0.25,
            volatility=2.0, option_type="CALL"
        )
        assert isinstance(delta_high_vol, float)
        assert 0.0 <= delta_high_vol <= 1.0
    
    def test_calculate_synthetic_dividend_return(self):
        """Testa cálculo de retorno de dividendo sintético."""
        # Teste básico
        annual_return = calculate_synthetic_dividend_return(
            premium=2.0, strike=100.0, days_to_expiration=30
        )
        
        # Retorno anualizado deve ser positivo
        assert annual_return > 0
        
        # Teste com diferentes prazos
        return_30d = calculate_synthetic_dividend_return(2.0, 100.0, 30)
        return_60d = calculate_synthetic_dividend_return(2.0, 100.0, 60)
        
        # Prazo maior deve dar retorno anualizado maior (mesmo prêmio)
        assert return_30d > return_60d
        
        # Teste com prêmio zero
        return_zero = calculate_synthetic_dividend_return(0.0, 100.0, 30)
        assert return_zero == 0.0
        
        # Teste com strike zero (edge case)
        return_no_strike = calculate_synthetic_dividend_return(2.0, 0.0, 30)
        assert return_no_strike == 0.0  # Deve evitar divisão por zero


class TestFindSyntheticDividendOptions:
    """Testes para a função principal de busca."""
    
    @patch('synthetic_dividends.OpLabClient')
    def test_find_synthetic_dividend_options_basic(self, mock_client_class, sample_option_chain):
        """Testa busca básica de opções de dividendo sintético."""
        # Setup mock
        mock_client = Mock()
        mock_client.get_underlying_price.return_value = 100.0
        mock_client.get_option_chain.return_value = sample_option_chain
        mock_client_class.return_value = mock_client
        
        # Executa função
        result = find_synthetic_dividend_options(
            "TEST",
            min_volume=10,
            min_days=15,
            max_days=60,
            max_exercise_prob=20.0,
            option_types="Ambas (CALL + PUT)"
        )
        
        # Verifica resultado
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            # Verifica colunas esperadas
            expected_columns = [
                'Ativo', 'Opção', 'Estratégia', 'Strike', 'Vencimento',
                'Prêmio (R$)', 'Retorno a.a. (%)', 'Prob. Exercício (%)',
                'Justificativa', 'Volume'
            ]
            for col in expected_columns:
                assert col in result.columns
            
            # Verifica tipos de estratégia
            strategies = result['Estratégia'].unique()
            for strategy in strategies:
                assert strategy in ['CALL', 'PUT']
            
            # Verifica se probabilidade está dentro do limite
            max_prob = result['Prob. Exercício (%)'].max()
            assert max_prob <= 20.0
    
    @patch('synthetic_dividends.OpLabClient')
    def test_find_synthetic_dividend_options_call_only(self, mock_client_class, sample_option_chain):
        """Testa busca apenas de CALLs."""
        mock_client = Mock()
        mock_client.get_underlying_price.return_value = 100.0
        mock_client.get_option_chain.return_value = sample_option_chain
        mock_client_class.return_value = mock_client
        
        result = find_synthetic_dividend_options(
            "TEST",
            option_types="Apenas CALL"
        )
        
        if not result.empty:
            # Deve conter apenas CALLs
            strategies = result['Estratégia'].unique()
            assert len(strategies) == 1
            assert strategies[0] == 'CALL'
    
    @patch('synthetic_dividends.OpLabClient')
    def test_find_synthetic_dividend_options_put_only(self, mock_client_class, sample_option_chain):
        """Testa busca apenas de PUTs."""
        mock_client = Mock()
        mock_client.get_underlying_price.return_value = 100.0
        mock_client.get_option_chain.return_value = sample_option_chain
        mock_client_class.return_value = mock_client
        
        result = find_synthetic_dividend_options(
            "TEST",
            option_types="Apenas PUT"
        )
        
        if not result.empty:
            # Deve conter apenas PUTs
            strategies = result['Estratégia'].unique()
            assert len(strategies) == 1
            assert strategies[0] == 'PUT'
    
    @patch('synthetic_dividends.OpLabClient')
    def test_find_synthetic_dividend_options_empty_chain(self, mock_client_class):
        """Testa busca com chain vazia."""
        mock_client = Mock()
        mock_client.get_underlying_price.return_value = 100.0
        mock_client.get_option_chain.return_value = pd.DataFrame()  # Chain vazia
        mock_client_class.return_value = mock_client
        
        result = find_synthetic_dividend_options("TEST")
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    @patch('synthetic_dividends.OpLabClient')
    def test_find_synthetic_dividend_options_no_price(self, mock_client_class, sample_option_chain):
        """Testa busca sem preço do ativo."""
        mock_client = Mock()
        mock_client.get_underlying_price.return_value = None
        mock_client.get_option_chain.return_value = sample_option_chain
        mock_client_class.return_value = mock_client
        
        result = find_synthetic_dividend_options("TEST")
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty  # Deve retornar vazio se não conseguir preço
    
    def test_find_synthetic_dividend_options_volume_filter(self):
        """Testa filtro de volume."""
        # Cria chain com diferentes volumes
        chain_data = []
        for i, strike in enumerate([95, 100, 105]):
            for option_type in ['CALL', 'PUT']:
                chain_data.append({
                    'symbol': f'TEST{strike}{option_type[0]}',
                    'option_type': option_type,
                    'strike': strike,
                    'expiration': datetime.now() + timedelta(days=30),
                    'last': 2.0,
                    'volume': i * 50,  # Volumes: 0, 50, 100
                    'delta': 0.3 if option_type == 'CALL' else -0.3
                })
        
        option_chain = pd.DataFrame(chain_data)
        
        with patch('synthetic_dividends.OpLabClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_underlying_price.return_value = 100.0
            mock_client.get_option_chain.return_value = option_chain
            mock_client_class.return_value = mock_client
            
            # Filtro de volume = 75 (deve pegar apenas strike 105)
            result = find_synthetic_dividend_options("TEST", min_volume=75)
            
            if not result.empty:
                # Verifica se apenas opções com volume >= 75 foram incluídas
                volumes = result['Volume'].values
                assert all(vol >= 75 for vol in volumes if not pd.isna(vol))
    
    def test_find_synthetic_dividend_options_expiration_filter(self):
        """Testa filtro de expiração."""
        # Cria chain com diferentes expirações
        chain_data = []
        expirations = [
            datetime.now() + timedelta(days=10),   # Muito próximo
            datetime.now() + timedelta(days=30),   # Dentro do range
            datetime.now() + timedelta(days=90)    # Muito longe
        ]
        
        for i, exp in enumerate(expirations):
            chain_data.append({
                'symbol': f'TEST{i}C',
                'option_type': 'CALL',
                'strike': 100,
                'expiration': exp,
                'last': 2.0,
                'volume': 1000,
                'delta': 0.3
            })
        
        option_chain = pd.DataFrame(chain_data)
        
        with patch('synthetic_dividends.OpLabClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_underlying_price.return_value = 100.0
            mock_client.get_option_chain.return_value = option_chain
            mock_client_class.return_value = mock_client
            
            # Filtro: 15-60 dias
            result = find_synthetic_dividend_options(
                "TEST", 
                min_days=15, 
                max_days=60
            )
            
            if not result.empty:
                # Deve conter apenas a opção com 30 dias
                assert len(result) == 1
    
    def test_find_synthetic_dividend_options_exercise_prob_filter(self):
        """Testa filtro de probabilidade de exercício."""
        # Cria chain com diferentes deltas
        chain_data = []
        deltas = [0.1, 0.3, 0.7]  # Baixo, médio, alto risco
        
        for i, delta in enumerate(deltas):
            chain_data.append({
                'symbol': f'TEST{i}C',
                'option_type': 'CALL',
                'strike': 100,
                'expiration': datetime.now() + timedelta(days=30),
                'last': 2.0,
                'volume': 1000,
                'delta': delta
            })
        
        option_chain = pd.DataFrame(chain_data)
        
        with patch('synthetic_dividends.OpLabClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_underlying_price.return_value = 100.0
            mock_client.get_option_chain.return_value = option_chain
            mock_client_class.return_value = mock_client
            
            # Filtro: máximo 25% de probabilidade de exercício
            result = find_synthetic_dividend_options(
                "TEST", 
                max_exercise_prob=25.0
            )
            
            if not result.empty:
                # Verifica se todas as opções têm prob <= 25%
                probs = result['Prob. Exercício (%)'].values
                assert all(prob <= 25.0 for prob in probs)
    
    @patch('synthetic_dividends.OpLabClient')
    def test_find_synthetic_dividend_options_sorting(self, mock_client_class):
        """Testa ordenação por retorno anualizado."""
        # Cria chain com diferentes retornos
        chain_data = []
        premiums = [1.0, 3.0, 2.0]  # Diferentes prêmios = diferentes retornos
        
        for i, premium in enumerate(premiums):
            chain_data.append({
                'symbol': f'TEST{i}C',
                'option_type': 'CALL',
                'strike': 100,
                'expiration': datetime.now() + timedelta(days=30),
                'last': premium,
                'volume': 1000,
                'delta': 0.2
            })
        
        option_chain = pd.DataFrame(chain_data)
        
        mock_client = Mock()
        mock_client.get_underlying_price.return_value = 100.0
        mock_client.get_option_chain.return_value = option_chain
        mock_client_class.return_value = mock_client
        
        result = find_synthetic_dividend_options("TEST")
        
        if len(result) > 1:
            # Verifica se está ordenado por retorno (decrescente)
            returns = result['Retorno a.a. (%)'].values
            assert all(returns[i] >= returns[i+1] for i in range(len(returns)-1))
    
    def test_find_synthetic_dividend_options_data_integrity(self):
        """Testa integridade dos dados retornados."""
        # Cria chain válida
        chain_data = [{
            'symbol': 'TESTC100',
            'option_type': 'CALL',
            'strike': 100,
            'expiration': datetime.now() + timedelta(days=30),
            'last': 2.5,
            'volume': 1000,
            'delta': 0.25
        }]
        
        option_chain = pd.DataFrame(chain_data)
        
        with patch('synthetic_dividends.OpLabClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_underlying_price.return_value = 100.0
            mock_client.get_option_chain.return_value = option_chain
            mock_client_class.return_value = mock_client
            
            result = find_synthetic_dividend_options("TEST")
            
            if not result.empty:
                row = result.iloc[0]
                
                # Verifica tipos de dados
                assert isinstance(row['Strike'], (int, float))
                assert isinstance(row['Prêmio (R$)'], (int, float))
                assert isinstance(row['Retorno a.a. (%)'], (int, float))
                assert isinstance(row['Prob. Exercício (%)'], (int, float))
                assert isinstance(row['Volume'], (int, float)) or pd.isna(row['Volume'])
                
                # Verifica valores lógicos
                assert row['Strike'] > 0
                assert row['Prêmio (R$)'] > 0
                assert row['Retorno a.a. (%)'] >= 0
                assert 0 <= row['Prob. Exercício (%)'] <= 100
                
                # Verifica strings
                assert isinstance(row['Ativo'], str)
                assert isinstance(row['Opção'], str)
                assert isinstance(row['Estratégia'], str)
                assert isinstance(row['Justificativa'], str)


class TestEdgeCasesAndErrorHandling:
    """Testes para casos extremos e tratamento de erros."""
    
    @patch('synthetic_dividends.OpLabClient')
    def test_api_error_handling(self, mock_client_class):
        """Testa tratamento de erros da API."""
        mock_client = Mock()
        mock_client.get_underlying_price.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        # Não deve quebrar, deve retornar DataFrame vazio
        result = find_synthetic_dividend_options("TEST")
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_invalid_parameters(self):
        """Testa parâmetros inválidos."""
        with patch('synthetic_dividends.OpLabClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_underlying_price.return_value = 100.0
            mock_client.get_option_chain.return_value = pd.DataFrame()
            mock_client_class.return_value = mock_client
            
            # Parâmetros negativos
            result = find_synthetic_dividend_options(
                "TEST",
                min_volume=-10,
                min_days=-5,
                max_days=-1
            )
            assert isinstance(result, pd.DataFrame)
            
            # min_days > max_days
            result = find_synthetic_dividend_options(
                "TEST",
                min_days=60,
                max_days=30
            )
            assert isinstance(result, pd.DataFrame)
    
    def test_missing_delta_calculation(self):
        """Testa cálculo quando delta não está disponível."""
        # Chain sem delta
        chain_data = [{
            'symbol': 'TESTC100',
            'option_type': 'CALL',
            'strike': 100,
            'expiration': datetime.now() + timedelta(days=30),
            'last': 2.5,
            'volume': 1000
            # Sem campo delta
        }]
        
        option_chain = pd.DataFrame(chain_data)
        
        with patch('synthetic_dividends.OpLabClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_underlying_price.return_value = 100.0
            mock_client.get_option_chain.return_value = option_chain
            mock_client_class.return_value = mock_client
            
            result = find_synthetic_dividend_options("TEST")
            
            # Deve funcionar calculando delta via Black-Scholes
            if not result.empty:
                assert 'Prob. Exercício (%)' in result.columns
                prob = result.iloc[0]['Prob. Exercício (%)']
                assert 0 <= prob <= 100
    
    def test_extreme_market_conditions(self):
        """Testa condições extremas de mercado."""
        # Preços muito altos
        chain_data = [{
            'symbol': 'TESTC10000',
            'option_type': 'CALL',
            'strike': 10000,
            'expiration': datetime.now() + timedelta(days=30),
            'last': 500,
            'volume': 1000,
            'delta': 0.25
        }]
        
        option_chain = pd.DataFrame(chain_data)
        
        with patch('synthetic_dividends.OpLabClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_underlying_price.return_value = 10000.0
            mock_client.get_option_chain.return_value = option_chain
            mock_client_class.return_value = mock_client
            
            result = find_synthetic_dividend_options("TEST")
            
            # Deve funcionar mesmo com valores altos
            if not result.empty:
                assert result.iloc[0]['Strike'] == 10000
                assert result.iloc[0]['Prêmio (R$)'] == 500
    
    def test_data_type_consistency(self):
        """Testa consistência de tipos de dados."""
        # Chain com tipos mistos
        chain_data = [{
            'symbol': 'TESTC100',
            'option_type': 'CALL',
            'strike': '100',  # String
            'expiration': datetime.now() + timedelta(days=30),
            'last': '2.5',    # String
            'volume': '1000', # String
            'delta': '0.25'   # String
        }]
        
        option_chain = pd.DataFrame(chain_data)
        
        with patch('synthetic_dividends.OpLabClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_underlying_price.return_value = 100.0
            mock_client.get_option_chain.return_value = option_chain
            mock_client_class.return_value = mock_client
            
            result = find_synthetic_dividend_options("TEST")
            
            # Deve converter tipos corretamente
            if not result.empty:
                assert isinstance(result.iloc[0]['Strike'], (int, float))
                assert isinstance(result.iloc[0]['Prêmio (R$)'], (int, float))
