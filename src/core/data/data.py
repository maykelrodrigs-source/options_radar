"""
Módulo de dados históricos - DADOS REAIS via Yahoo Finance
Integração com yfinance para dados históricos da B3
"""

import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Optional
import os
from functools import lru_cache

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.getcwd(), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


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


def _cache_path(ticker: str, start_date: str = None, end_date: str = None, days: int = None) -> str:
    ticker = ticker.upper().strip()
    if days is not None:
        return os.path.join(CACHE_DIR, f"{ticker}_last_{days}d.parquet")
    return os.path.join(CACHE_DIR, f"{ticker}_{start_date}_{end_date}.parquet")


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    try:
        if not df.empty:
            df.to_parquet(path, index=False)
    except Exception as e:
        logger.warning(f"⚠️ Falha ao salvar cache em {path}: {e}")


def _read_parquet(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            return df
    except Exception as e:
        logger.warning(f"⚠️ Falha ao ler cache em {path}: {e}")
    return None


# Wrappers com cache persistente + LRU em memória
@lru_cache(maxsize=256)
def get_price_history_cached(ticker: str, history_days: int) -> pd.DataFrame:
    path = _cache_path(ticker, days=history_days)
    cached = _read_parquet(path)
    if cached is not None and not cached.empty:
        logger.info(f"🗂️ Cache hit (price_history): {ticker} last {history_days}d")
        return cached
    df = get_price_history(ticker, history_days)
    _write_parquet(df, path)
    return df


@lru_cache(maxsize=512)
def get_historical_data_cached(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    path = _cache_path(ticker, start_date, end_date)
    cached = _read_parquet(path)
    if cached is not None and not cached.empty:
        logger.info(f"🗂️ Cache hit (period): {ticker} {start_date}..{end_date}")
        return cached
    df = get_historical_data(ticker, start_date, end_date)
    _write_parquet(df, path)
    return df


# Instância global para compatibilidade
oplab_data_provider = None

# Exports
__all__ = [
    'get_price_history',
    'get_historical_data',
    'get_price_history_cached',
    'get_historical_data_cached',
    'oplab_data_provider'
]