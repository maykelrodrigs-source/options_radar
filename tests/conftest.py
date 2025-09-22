"""
Configurações globais para testes pytest.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import sys
import os

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_price_data():
    """Fixture com dados de preço para testes."""
    dates = pd.date_range(start='2024-01-01', periods=300, freq='D')
    base_price = 100.0
    
    # Cria série de preços com tendência e volatilidade
    np.random.seed(42)  # Para reprodutibilidade
    returns = np.random.normal(0.001, 0.02, 300)  # 0.1% média, 2% vol
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    price_series = pd.Series(prices, index=dates)
    
    return pd.DataFrame({
        'close': price_series,
        'high': price_series * 1.02,
        'low': price_series * 0.98,
        'volume': np.random.randint(100000, 1000000, 300)
    })

@pytest.fixture
def sample_option_chain():
    """Fixture com chain de opções para testes."""
    current_price = 100.0
    strikes = np.arange(80, 121, 2.5)  # Strikes de 80 a 120
    expirations = [
        datetime.now() + timedelta(days=30),
        datetime.now() + timedelta(days=60),
        datetime.now() + timedelta(days=90)
    ]
    
    options = []
    for exp in expirations:
        for strike in strikes:
            # CALL
            call_premium = max(0.1, current_price - strike + np.random.normal(0, 2))
            options.append({
                'symbol': f'TEST{int(strike)}C{exp.strftime("%m%d")}',
                'option_type': 'CALL',
                'strike': strike,
                'expiration': exp,
                'bid': max(0.05, call_premium - 0.1),
                'ask': call_premium + 0.1,
                'last': call_premium,
                'volume': np.random.randint(0, 1000) if np.random.random() > 0.3 else np.nan,
                'open_interest': np.random.randint(0, 5000) if np.random.random() > 0.4 else np.nan,
                'delta': max(0.01, min(0.99, 0.5 + (current_price - strike) / 50))
            })
            
            # PUT
            put_premium = max(0.1, strike - current_price + np.random.normal(0, 2))
            options.append({
                'symbol': f'TEST{int(strike)}P{exp.strftime("%m%d")}',
                'option_type': 'PUT',
                'strike': strike,
                'expiration': exp,
                'bid': max(0.05, put_premium - 0.1),
                'ask': put_premium + 0.1,
                'last': put_premium,
                'volume': np.random.randint(0, 1000) if np.random.random() > 0.3 else np.nan,
                'open_interest': np.random.randint(0, 5000) if np.random.random() > 0.4 else np.nan,
                'delta': -max(0.01, min(0.99, 0.5 - (current_price - strike) / 50))
            })
    
    return pd.DataFrame(options)

@pytest.fixture
def mock_oplab_client():
    """Mock do cliente OpLab para testes."""
    client = Mock()
    client.get_underlying_price.return_value = 100.0
    client.get_option_chain.return_value = sample_option_chain()
    return client

@pytest.fixture
def horizon_params():
    """Fixture com parâmetros de horizonte para testes."""
    return {
        "curto": {
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
        "médio": {
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
        }
    }

# Configuração para ignorar warnings específicos
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup do ambiente de teste."""
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
