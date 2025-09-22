"""
Módulo para busca de dados históricos com cache inteligente.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from typing import Optional


# @st.cache_data(ttl=900)  # 15 min - CACHE DESABILITADO PARA DEBUG
def get_price_history(ticker: str, history_days: int) -> pd.DataFrame:
    """
    Busca dados históricos simulados com cache baseado no horizonte.
    
    Args:
        ticker: Código do ativo
        history_days: Número de dias de histórico solicitado
        
    Returns:
        DataFrame com colunas ['date', 'close', 'volume']
    """
    # Simula dados históricos baseados no ticker e horizonte
    # O history_days é incluído na chave do cache para invalidar quando mudar
    
    # Seed baseado no ticker para reprodutibilidade
    seed_value = hash(f"{ticker}_{history_days}") % (2**32)
    np.random.seed(seed_value)
    
    # Gera datas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(history_days * 1.3))  # +30% para garantir dias úteis
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Remove fins de semana
    dates = dates[dates.weekday < 5]
    
    # Garante pelo menos history_days dias úteis
    if len(dates) < history_days:
        extra_days = history_days - len(dates)
        start_date = end_date - timedelta(days=int(history_days * 1.5))
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]
    
    n_days = len(dates)
    
    # Simula preço atual baseado no ticker
    base_prices = {
        "PETR4": 35.50,
        "VALE3": 62.30,
        "ITUB4": 25.80,
        "BBDC4": 18.90,
        "BBAS3": 42.10,
    }
    
    current_price = base_prices.get(ticker.upper(), 30.0)
    
    # Gera retornos mais variados para backtest
    # Adiciona ciclos e tendências baseados no ticker e período
    ticker_factor = hash(ticker) % 100 / 100.0  # 0 a 1
    
    # Base trends por horizonte
    if history_days <= 90:  # Curto prazo
        base_trend = 0.0005
        base_volatility = 0.030
    elif history_days <= 260:  # Médio prazo
        base_trend = 0.0003
        base_volatility = 0.025
    else:  # Longo prazo
        base_trend = 0.0002
        base_volatility = 0.022
    
    # Ajusta tendência baseado no ticker (some sobem, outros caem)
    if ticker_factor < 0.3:
        trend_multiplier = 1.5  # Tendência de alta
    elif ticker_factor > 0.7:
        trend_multiplier = -1.2  # Tendência de baixa
    else:
        trend_multiplier = 0.5   # Lateral
    
    trend = base_trend * trend_multiplier
    volatility = base_volatility * (0.8 + ticker_factor * 0.4)  # Varia volatilidade
    
    # Gera retornos com alguns períodos de maior volatilidade
    returns = np.random.normal(trend, volatility, n_days)
    
    # Adiciona alguns "eventos" de alta volatilidade
    n_events = max(1, n_days // 50)  # 1 evento a cada ~50 dias
    for _ in range(n_events):
        event_day = np.random.randint(0, n_days)
        event_magnitude = np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.15)  # ±5% a ±15%
        returns[event_day] = event_magnitude
    
    # Gera preços do passado para o presente
    prices = [current_price]
    for i in range(n_days - 1, 0, -1):
        prices.insert(0, prices[0] / (1 + returns[i]))
    
    # Simula volume
    base_volume = 1000000
    volume_noise = np.random.lognormal(0, 0.4, n_days)
    volumes = (base_volume * volume_noise).astype(int)
    
    # Cria DataFrame
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': volumes
    })
    
    # Retorna apenas os últimos history_days dias úteis
    result_df = df.tail(history_days).reset_index(drop=True)
    return result_df
