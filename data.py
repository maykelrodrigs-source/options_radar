"""
Módulo de dados históricos - DADOS REAIS via Yahoo Finance
Integração com yfinance para dados históricos da B3
"""

import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _convert_ticker_to_yfinance(ticker: str) -> str:
    """
    Converte ticker brasileiro para formato Yahoo Finance.
    
    Args:
        ticker: Ticker brasileiro (ex: PETR4)
        
    Returns:
        Ticker no formato Yahoo Finance (ex: PETR4.SA)
    """
    ticker = ticker.upper().strip()
    if not ticker.endswith('.SA'):
        ticker += '.SA'
    return ticker


def get_price_history(ticker: str, history_days: int) -> pd.DataFrame:
    """
    Busca dados históricos REAIS via Yahoo Finance.
    
    Args:
        ticker: Código da ação (ex: PETR4)
        history_days: Número de dias de histórico
        
    Returns:
        DataFrame com dados históricos REAIS
    """
    try:
        logger.info(f"📊 Buscando dados históricos REAIS para {ticker} via Yahoo Finance")
        
        # Converte ticker para formato Yahoo Finance
        yf_ticker = _convert_ticker_to_yfinance(ticker)
        
        # Calcula período baseado em dias úteis (aproximadamente)
        # Adiciona 40% para compensar fins de semana e feriados
        period_days = int(history_days * 1.4)
        
        # Busca dados do Yahoo Finance
        stock = yf.Ticker(yf_ticker)
        hist = stock.history(period=f"{period_days}d")
        
        if hist.empty:
            logger.error(f"❌ Yahoo Finance não retornou dados para {yf_ticker}")
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Converte para formato esperado pelo sistema
        df = pd.DataFrame({
            'date': hist.index,
            'open': hist['Open'],
            'high': hist['High'], 
            'low': hist['Low'],
            'close': hist['Close'],
            'volume': hist['Volume']
        })
        
        # Garante que temos pelo menos os dias solicitados
        df = df.tail(history_days).reset_index(drop=True)
        
        logger.info(f"✅ Dados históricos REAIS obtidos para {ticker}: {len(df)} registros")
        logger.info(f"📅 Período: {df['date'].iloc[0].date()} a {df['date'].iloc[-1].date()}")
        logger.info(f"💰 Preço: R$ {df['close'].iloc[0]:.2f} → R$ {df['close'].iloc[-1]:.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Erro ao buscar dados históricos para {ticker}: {e}")
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])


def get_historical_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Busca dados históricos REAIS por período específico.
    
    Args:
        ticker: Código da ação
        start_date: Data inicial (YYYY-MM-DD)
        end_date: Data final (YYYY-MM-DD)
        
    Returns:
        DataFrame com dados históricos REAIS
    """
    try:
        logger.info(f"📊 Buscando dados históricos REAIS para {ticker}: {start_date} a {end_date}")
        
        # Converte ticker para formato Yahoo Finance
        yf_ticker = _convert_ticker_to_yfinance(ticker)
        
        # Busca dados do Yahoo Finance
        stock = yf.Ticker(yf_ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            logger.error(f"❌ Yahoo Finance não retornou dados para {yf_ticker}")
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Converte para formato esperado pelo sistema
        df = pd.DataFrame({
            'date': hist.index,
            'open': hist['Open'],
            'high': hist['High'],
            'low': hist['Low'], 
            'close': hist['Close'],
            'volume': hist['Volume']
        })
        
        df = df.reset_index(drop=True)
        
        logger.info(f"✅ Dados históricos REAIS obtidos para {ticker}: {len(df)} registros")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Erro ao buscar dados históricos para {ticker}: {e}")
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])


# Instância global para compatibilidade
oplab_data_provider = None

# Exports
__all__ = [
    'get_price_history',
    'get_historical_data', 
    'oplab_data_provider'
]