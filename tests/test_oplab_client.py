"""
Testes para oplab_client.py
Cobertura alvo: 85%
"""
import pytest
import pandas as pd
import requests
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from oplab_client import OpLabClient


class TestOpLabClient:
    """Testes para a classe OpLabClient."""
    
    def test_init_with_env_vars(self):
        """Testa inicialização com variáveis de ambiente."""
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key',
            'OPLAB_API_AUTH_HEADER': 'Authorization',
            'OPLAB_API_AUTH_SCHEME': 'Bearer',
            'OPLAB_OPTION_CHAIN_ENDPOINT': '/test/options/{ticker}',
            'OPLAB_QUOTE_ENDPOINT': '/test/quote/{ticker}',
            'OPLAB_MOST_ACTIVES_ENDPOINT': '/test/actives'
        }):
            client = OpLabClient()
            assert client.base_url == 'https://test.api.com'
            assert client.api_key == 'test_key'
            assert client.auth_header == 'Authorization'
            assert client.auth_scheme == 'Bearer'
    
    def test_init_missing_env_vars(self):
        """Testa inicialização com variáveis de ambiente faltando."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OPLAB_API_BASE_URL não configurado"):
                OpLabClient()
    
    def test_init_missing_api_key(self):
        """Testa inicialização sem API key."""
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com'
        }, clear=True):
            with pytest.raises(ValueError, match="OPLAB_API_KEY não configurado"):
                OpLabClient()
    
    @patch('oplab_client.requests.get')
    def test_get_successful_request(self, mock_get):
        """Testa requisição GET bem-sucedida."""
        # Setup
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test'}
        mock_get.return_value = mock_response
        
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key',
            'OPLAB_API_AUTH_HEADER': 'Authorization',
            'OPLAB_API_AUTH_SCHEME': 'Bearer',
            'OPLAB_OPTION_CHAIN_ENDPOINT': '/options/{ticker}',
            'OPLAB_QUOTE_ENDPOINT': '/quote/{ticker}'
        }):
            client = OpLabClient()
            result = client._get('/test/endpoint')
            
            assert result == {'data': 'test'}
            mock_get.assert_called_once()
            
            # Verifica headers
            call_args = mock_get.call_args
            headers = call_args[1]['headers']
            assert 'Authorization' in headers
            assert headers['Authorization'] == 'Bearer test_key'
    
    @patch('oplab_client.requests.get')
    def test_get_http_error(self, mock_get):
        """Testa tratamento de erro HTTP."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("Not Found")
        mock_get.return_value = mock_response
        
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key'
        }):
            client = OpLabClient()
            
            with pytest.raises(requests.HTTPError):
                client._get('/test/endpoint')
    
    @patch('oplab_client.requests.get')
    def test_get_connection_error(self, mock_get):
        """Testa tratamento de erro de conexão."""
        mock_get.side_effect = requests.ConnectionError("Connection failed")
        
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key'
        }):
            client = OpLabClient()
            
            with pytest.raises(requests.ConnectionError):
                client._get('/test/endpoint')
    
    @patch('oplab_client.requests.get')
    def test_get_with_ticker_parameter(self, mock_get):
        """Testa requisição com parâmetro ticker."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'ticker': 'PETR4'}
        mock_get.return_value = mock_response
        
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key',
            'OPLAB_OPTION_CHAIN_ENDPOINT': '/options/{ticker}'
        }):
            client = OpLabClient()
            result = client._get('/options/{ticker}', ticker='PETR4')
            
            # Verifica se URL foi formatada corretamente
            expected_url = 'https://test.api.com/options/PETR4'
            mock_get.assert_called_once()
            actual_url = mock_get.call_args[0][0]
            assert actual_url == expected_url
    
    @patch.object(OpLabClient, '_get')
    def test_get_underlying_price_success(self, mock_get):
        """Testa busca de preço do ativo com sucesso."""
        mock_get.return_value = {'price': 35.50}
        
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key',
            'OPLAB_QUOTE_ENDPOINT': '/quote/{ticker}'
        }):
            client = OpLabClient()
            price = client.get_underlying_price('PETR4')
            
            assert price == 35.50
            mock_get.assert_called_once_with('/quote/{ticker}', ticker='PETR4')
    
    @patch.object(OpLabClient, '_get')
    def test_get_underlying_price_multiple_formats(self, mock_get):
        """Testa diferentes formatos de resposta de preço."""
        test_cases = [
            ({'price': 35.50}, 35.50),
            ({'last': 40.25}, 40.25),
            ({'close': 30.75}, 30.75),
            ({'value': 45.00}, 45.00),
            ({'current_price': 50.25}, 50.25),
            ({}, None),  # Sem campos de preço
            ({'other_field': 'value'}, None)  # Campos irrelevantes
        ]
        
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key',
            'OPLAB_QUOTE_ENDPOINT': '/quote/{ticker}'
        }):
            client = OpLabClient()
            
            for response, expected in test_cases:
                mock_get.return_value = response
                result = client.get_underlying_price('TEST')
                assert result == expected
    
    @patch.object(OpLabClient, '_get')
    def test_get_option_chain_success(self, mock_get):
        """Testa busca de chain de opções com sucesso."""
        mock_response = {
            'options': [
                {
                    'symbol': 'PETR4C35',
                    'type': 'CALL',
                    'strike': 35.0,
                    'expiration': '2024-12-20',
                    'bid': 1.50,
                    'ask': 1.60,
                    'last': 1.55,
                    'volume': 1000,
                    'open_interest': 5000,
                    'delta': 0.45
                }
            ]
        }
        mock_get.return_value = mock_response
        
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key',
            'OPLAB_OPTION_CHAIN_ENDPOINT': '/options/{ticker}'
        }):
            client = OpLabClient()
            chain = client.get_option_chain('PETR4')
            
            assert isinstance(chain, pd.DataFrame)
            assert len(chain) == 1
            assert 'symbol' in chain.columns
            assert 'option_type' in chain.columns  # Normalizado de 'type'
            assert 'strike' in chain.columns
            assert chain.iloc[0]['option_type'] == 'CALL'
            assert chain.iloc[0]['strike'] == 35.0
    
    @patch.object(OpLabClient, '_get')
    def test_get_option_chain_empty_response(self, mock_get):
        """Testa chain de opções com resposta vazia."""
        test_cases = [
            {},  # Objeto vazio
            {'options': []},  # Lista vazia
            {'data': []},  # Campo data vazio
            None  # Resposta nula
        ]
        
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key',
            'OPLAB_OPTION_CHAIN_ENDPOINT': '/options/{ticker}'
        }):
            client = OpLabClient()
            
            for response in test_cases:
                mock_get.return_value = response
                chain = client.get_option_chain('TEST')
                assert isinstance(chain, pd.DataFrame)
                assert chain.empty
    
    def test_normalize_option_record_complete(self):
        """Testa normalização de registro completo de opção."""
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key'
        }):
            client = OpLabClient()
            
            raw_record = {
                'symbol': 'PETR4C35',
                'type': 'CALL',
                'strike': 35.0,
                'expiration': '2024-12-20',
                'bid': 1.50,
                'ask': 1.60,
                'last': 1.55,
                'volume': 1000,
                'open_interest': 5000,
                'delta': 0.45
            }
            
            normalized = client._normalize_option_record(raw_record)
            
            assert normalized['symbol'] == 'PETR4C35'
            assert normalized['option_type'] == 'CALL'  # Normalizado
            assert normalized['strike'] == 35.0
            assert normalized['bid'] == 1.50
            assert normalized['volume'] == 1000
            assert normalized['open_interest'] == 5000
            assert normalized['delta'] == 0.45
    
    def test_normalize_option_record_missing_fields(self):
        """Testa normalização com campos faltando."""
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key'
        }):
            client = OpLabClient()
            
            raw_record = {
                'symbol': 'PETR4P30'
                # Campos faltando
            }
            
            normalized = client._normalize_option_record(raw_record)
            
            assert normalized['symbol'] == 'PETR4P30'
            assert normalized['option_type'] is None
            assert normalized['strike'] is None
            assert normalized['volume'] is None
            assert normalized['open_interest'] is None
    
    def test_normalize_option_record_alternative_keys(self):
        """Testa normalização com chaves alternativas."""
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key'
        }):
            client = OpLabClient()
            
            # Testa diferentes chaves para volume
            test_cases = [
                ({'vol': 500}, 500),
                ({'totalVolume': 750}, 750),
                ({'daily_volume': 1000}, None),  # Chave não reconhecida
                ({}, None)  # Sem volume
            ]
            
            for raw_data, expected_volume in test_cases:
                raw_record = {'symbol': 'TEST', **raw_data}
                normalized = client._normalize_option_record(raw_record)
                assert normalized['volume'] == expected_volume
            
            # Testa diferentes chaves para open interest
            oi_test_cases = [
                ({'openInterest': 1000}, 1000),
                ({'oi': 2000}, 2000),
                ({'open_int': 3000}, 3000),
                ({'contracts_open': 4000}, 4000),
                ({'unknown_oi': 5000}, None),  # Chave não reconhecida
                ({}, None)  # Sem OI
            ]
            
            for raw_data, expected_oi in oi_test_cases:
                raw_record = {'symbol': 'TEST', **raw_data}
                normalized = client._normalize_option_record(raw_record)
                assert normalized['open_interest'] == expected_oi
    
    def test_convert_to_numeric_valid_data(self):
        """Testa conversão para numérico com dados válidos."""
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key'
        }):
            client = OpLabClient()
            
            df = pd.DataFrame({
                'strike': ['35.0', '40.0', '45.0'],
                'bid': ['1.5', '2.0', '2.5'],
                'volume': ['1000', '2000', '3000'],
                'text_field': ['A', 'B', 'C']  # Não deve ser convertido
            })
            
            numeric_columns = ['strike', 'bid', 'volume']
            result = client._convert_to_numeric(df, numeric_columns)
            
            assert result['strike'].dtype == 'float64'
            assert result['bid'].dtype == 'float64'
            assert result['volume'].dtype == 'float64'
            assert result['text_field'].dtype == 'object'  # Permanece texto
            
            assert result['strike'].iloc[0] == 35.0
            assert result['volume'].iloc[2] == 3000.0
    
    def test_convert_to_numeric_invalid_data(self):
        """Testa conversão com dados inválidos."""
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key'
        }):
            client = OpLabClient()
            
            df = pd.DataFrame({
                'strike': ['35.0', 'invalid', '45.0'],
                'volume': ['1000', '', 'abc'],
                'bid': [None, '2.0', '2.5']
            })
            
            numeric_columns = ['strike', 'volume', 'bid']
            result = client._convert_to_numeric(df, numeric_columns)
            
            # Valores inválidos devem ser NaN
            assert pd.isna(result['strike'].iloc[1])  # 'invalid'
            assert pd.isna(result['volume'].iloc[1])  # ''
            assert pd.isna(result['volume'].iloc[2])  # 'abc'
            assert pd.isna(result['bid'].iloc[0])     # None
            
            # Valores válidos devem ser convertidos
            assert result['strike'].iloc[0] == 35.0
            assert result['bid'].iloc[1] == 2.0
    
    @patch.object(OpLabClient, '_get')
    def test_get_most_active_stocks_success(self, mock_get):
        """Testa busca de ações mais ativas."""
        mock_response = {
            'stocks': [
                {'symbol': 'PETR4', 'volume': 1000000},
                {'symbol': 'VALE3', 'volume': 800000}
            ]
        }
        mock_get.return_value = mock_response
        
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key',
            'OPLAB_MOST_ACTIVES_ENDPOINT': '/actives'
        }):
            client = OpLabClient()
            result = client.get_most_active_stocks(limit=10)
            
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0] == 'PETR4'
            assert result[1] == 'VALE3'
    
    @patch.object(OpLabClient, '_get')
    def test_get_most_active_stocks_empty(self, mock_get):
        """Testa busca de ações ativas com resposta vazia."""
        mock_get.return_value = {}
        
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key',
            'OPLAB_MOST_ACTIVES_ENDPOINT': '/actives'
        }):
            client = OpLabClient()
            result = client.get_most_active_stocks()
            
            assert isinstance(result, list)
            assert len(result) == 0
    
    def test_expiration_parsing(self):
        """Testa parsing de diferentes formatos de data de expiração."""
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key'
        }):
            client = OpLabClient()
            
            test_cases = [
                ('2024-12-20', '2024-12-20'),
                ('20/12/2024', '20/12/2024'),
                ('2024-12-20T00:00:00', '2024-12-20T00:00:00'),
                (None, None),
                ('', None),
                ('invalid_date', 'invalid_date')  # Mantém original se inválido
            ]
            
            for input_date, expected in test_cases:
                raw_record = {
                    'symbol': 'TEST',
                    'expiration': input_date
                }
                
                normalized = client._normalize_option_record(raw_record)
                assert normalized['expiration'] == expected


class TestOpLabClientIntegration:
    """Testes de integração do cliente."""
    
    @patch.object(OpLabClient, '_get')
    def test_real_workflow_simulation(self, mock_get):
        """Simula um fluxo real de uso do cliente."""
        # Setup mocks para simular API real
        def mock_get_side_effect(endpoint, **kwargs):
            if '/quote/' in endpoint:
                return {'price': 35.50}
            elif '/options/' in endpoint:
                return {
                    'options': [
                        {
                            'symbol': 'PETR4C35',
                            'type': 'CALL',
                            'strike': 35.0,
                            'expiration': '2024-12-20',
                            'last': 1.55,
                            'volume': 1000,
                            'delta': 0.45
                        }
                    ]
                }
            elif '/actives' in endpoint:
                return {
                    'stocks': [
                        {'symbol': 'PETR4'},
                        {'symbol': 'VALE3'}
                    ]
                }
            return {}
        
        mock_get.side_effect = mock_get_side_effect
        
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key',
            'OPLAB_OPTION_CHAIN_ENDPOINT': '/options/{ticker}',
            'OPLAB_QUOTE_ENDPOINT': '/quote/{ticker}',
            'OPLAB_MOST_ACTIVES_ENDPOINT': '/actives'
        }):
            client = OpLabClient()
            
            # Fluxo completo
            actives = client.get_most_active_stocks()
            assert len(actives) == 2
            
            ticker = actives[0]
            price = client.get_underlying_price(ticker)
            assert price == 35.50
            
            chain = client.get_option_chain(ticker)
            assert not chain.empty
            assert len(chain) == 1
            assert chain.iloc[0]['option_type'] == 'CALL'
    
    def test_error_handling_resilience(self):
        """Testa resiliência do cliente a diferentes tipos de erro."""
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key',
            'OPLAB_OPTION_CHAIN_ENDPOINT': '/options/{ticker}'
        }):
            client = OpLabClient()
            
            # Testa diferentes cenários de erro
            with patch.object(client, '_get') as mock_get:
                # Timeout
                mock_get.side_effect = requests.Timeout("Request timeout")
                with pytest.raises(requests.Timeout):
                    client.get_underlying_price('PETR4')
                
                # Rate limit
                mock_response = Mock()
                mock_response.status_code = 429
                mock_response.raise_for_status.side_effect = requests.HTTPError("Rate limited")
                mock_get.side_effect = None
                mock_get.return_value = mock_response
                
                with pytest.raises(requests.HTTPError):
                    client.get_option_chain('PETR4')
    
    def test_data_consistency(self):
        """Testa consistência dos dados retornados."""
        with patch.dict('os.environ', {
            'OPLAB_API_BASE_URL': 'https://test.api.com',
            'OPLAB_API_KEY': 'test_key'
        }):
            client = OpLabClient()
            
            # Testa se normalização mantém consistência
            raw_records = [
                {'symbol': 'TEST1', 'type': 'CALL', 'strike': '35.0'},
                {'symbol': 'TEST2', 'option_type': 'PUT', 'strike': 40.0},
                {'symbol': 'TEST3', 'strike': None}
            ]
            
            normalized_records = [client._normalize_option_record(r) for r in raw_records]
            
            # Todos devem ter as mesmas chaves
            keys = set(normalized_records[0].keys())
            for record in normalized_records[1:]:
                assert set(record.keys()) == keys
            
            # Tipos devem ser consistentes
            for record in normalized_records:
                if record['strike'] is not None:
                    assert isinstance(record['strike'], (int, float))
                if record['option_type'] is not None:
                    assert record['option_type'] in ['CALL', 'PUT']
