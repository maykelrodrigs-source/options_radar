"""
Módulo para obter dados históricos para análise técnica.
Como a API do OpLab não fornece dados históricos, vamos simular dados baseados no preço atual.
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta
import random


class HistoricalDataProvider:
    """Provedor de dados históricos (simulado)."""
    
    def __init__(self):
        # Seed para reprodutibilidade
        np.random.seed(42)
        random.seed(42)
    
    def get_historical_data(self, ticker: str, current_price: float, 
                          days: int = 300) -> pd.DataFrame:
        """
        Simula dados históricos baseados no preço atual.
        
        Args:
            ticker: Código do ativo
            current_price: Preço atual
            days: Número de dias de histórico
            
        Returns:
            DataFrame com colunas ['date', 'close', 'volume']
        """
        # Define seed baseado no ticker para reprodutibilidade
        # Mas com variação baseada no preço para gerar cenários diferentes
        seed_value = hash(ticker) % (2**32)
        np.random.seed(seed_value)
        random.seed(seed_value)
        
        # Gera datas com mais dias para garantir 200 períodos úteis
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Remove fins de semana (mantém apenas dias úteis)
        dates = dates[dates.weekday < 5]
        
        # Garante pelo menos 250 dias úteis (mais que os 200 necessários)
        if len(dates) < 250:
            # Se não tem dias suficientes, gera mais dias no passado
            extra_days = 250 - len(dates)
            start_date = end_date - timedelta(days=days + extra_days + 50)  # +50 para fins de semana
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            dates = dates[dates.weekday < 5]
        
        # Simula preços com tendência e volatilidade realistas
        n_days = len(dates)
        
        # Gera retornos com tendência mais variada e realista
        # Usa o preço como base para determinar cenários diferentes
        price_factor = current_price / 25.0  # Normaliza em torno de 25
        
        # Cria cenários mais variados baseados no preço
        if price_factor > 1.5:  # Preços altos
            trend = 0.0008  # Ligeira tendência de alta
            volatility = 0.018
        elif price_factor < 0.8:  # Preços baixos  
            trend = -0.0008  # Ligeira tendência de baixa
            volatility = 0.022
        else:  # Preços médios
            trend = np.random.choice([-0.0005, 0.0005])  # Tendência aleatória
            volatility = 0.020
        
        returns = np.random.normal(trend, volatility, n_days)
        
        # Gera preços do passado para o presente
        prices = [current_price]
        for i in range(n_days - 1, 0, -1):
            prices.insert(0, prices[0] / (1 + returns[i]))
        
        # Simula volume (mais realista)
        base_volume = 1000000  # Volume base
        volume_noise = np.random.lognormal(0, 0.5, n_days)
        volumes = (base_volume * volume_noise).astype(int)
        
        # Cria DataFrame
        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': volumes
        })
        
        return df
    
    def get_mock_macro_data(self) -> dict:
        """
        Retorna dados macro simulados.
        
        Returns:
            Dicionário com indicadores macro
        """
        return {
            'selic': 10.75,  # Taxa SELIC atual
            'usd_brl': 5.20,  # USD/BRL
            'ipca': 4.62,    # IPCA acumulado 12 meses
            'ibov': 127500   # IBOVESPA
        }


# Instância global
historical_provider = HistoricalDataProvider()
