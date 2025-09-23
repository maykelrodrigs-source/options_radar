"""
Módulo de análise de fundamentos e valuation de ações brasileiras.
Implementa múltiplos métodos de cálculo de preço justo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.core.data.oplab_client import OpLabClient


@dataclass
class FundamentalData:
    """Dados fundamentais de uma ação."""
    ticker: str
    preco_atual: float
    lpa: float  # Lucro por ação
    vpa: float  # Valor patrimonial por ação
    dps: float  # Dividendos por ação
    dividend_yield: float  # Dividend yield atual
    payout: float  # Payout ratio
    crescimento_esperado: float  # Crescimento esperado (%)
    roe: float  # Return on Equity
    pl: float  # P/L atual
    pvp: float  # P/VPA atual
    peg_ratio: float  # PEG ratio


@dataclass
class ValuationResult:
    """Resultado de valuation com múltiplos métodos."""
    ticker: str
    preco_atual: float
    preco_graham: float
    preco_dividendos: float
    preco_pl10: float
    preco_pl12: float
    preco_pl15: float
    preco_pvp1: float
    preco_pvp1_5: float
    preco_bazin: float
    peg_ratio: float
    media_precos_justos: float
    desconto: bool
    caro: bool
    margem_seguranca: Dict[str, float]


def calculate_graham_price(lpa: float, vpa: float) -> float:
    """
    Calcula preço justo usando fórmula de Graham.
    Fórmula: sqrt(22.5 × LPA × VPA)
    """
    if lpa <= 0 or vpa <= 0:
        return 0.0
    
    return math.sqrt(22.5 * lpa * vpa)


def calculate_dividend_price(dps: float, yield_min: float = 6.0) -> float:
    """
    Calcula preço teto baseado no dividend yield desejado.
    Fórmula: DPS / yield_min
    """
    if dps <= 0 or yield_min <= 0:
        return 0.0
    
    return dps / (yield_min / 100)


def calculate_pl_price(lpa: float, pl_target: float) -> float:
    """
    Calcula preço baseado em P/L alvo.
    Fórmula: LPA × P/L_alvo
    """
    if lpa <= 0 or pl_target <= 0:
        return 0.0
    
    return lpa * pl_target


def calculate_pvp_price(vpa: float, pvp_target: float) -> float:
    """
    Calcula preço baseado em P/VPA alvo.
    Fórmula: VPA × P/VPA_alvo
    """
    if vpa <= 0 or pvp_target <= 0:
        return 0.0
    
    return vpa * pvp_target


def calculate_bazin_price(dps: float, crescimento: float, taxa_desconto: float = 6.0) -> float:
    """
    Calcula preço usando fórmula de Bazin.
    Fórmula: DPS / taxa_desconto
    """
    if dps <= 0 or taxa_desconto <= 0:
        return 0.0
    
    taxa_decimal = taxa_desconto / 100
    
    return dps / taxa_decimal


def calculate_peg_ratio(pl: float, crescimento: float) -> float:
    """
    Calcula PEG ratio.
    Fórmula: P/L / crescimento
    """
    if crescimento <= 0:
        return float('inf')
    
    return pl / crescimento


def calculate_margin_of_safety(preco_atual: float, preco_justo: float) -> float:
    """
    Calcula margem de segurança.
    Fórmula: (preco_justo - preco_atual) / preco_justo × 100
    """
    if preco_justo <= 0:
        return 0.0
    
    return ((preco_justo - preco_atual) / preco_justo) * 100


def analyze_fundamentals(
    ticker: str,
    fundamental_data: FundamentalData,
    yield_min: float = 6.0,
    pl_targets: List[float] = None,
    pvp_targets: List[float] = None,
    taxa_desconto: float = 6.0
) -> ValuationResult:
    """
    Analisa fundamentos de uma ação usando múltiplos métodos de valuation.
    
    Args:
        ticker: Código da ação
        fundamental_data: Dados fundamentais
        yield_min: Yield mínimo desejado (%)
        pl_targets: Lista de P/Ls alvo
        pvp_targets: Lista de P/VPA alvos
        taxa_desconto: Taxa de desconto para Bazin (%)
    
    Returns:
        ValuationResult com todos os cálculos
    """
    if pl_targets is None:
        pl_targets = [10, 12, 15]
    if pvp_targets is None:
        pvp_targets = [1.0, 1.5]
    
    # Cálculos de preço justo
    preco_graham = calculate_graham_price(fundamental_data.lpa, fundamental_data.vpa)
    preco_dividendos = calculate_dividend_price(fundamental_data.dps, yield_min)
    
    # Preços por P/L alvo
    preco_pl10 = calculate_pl_price(fundamental_data.lpa, pl_targets[0])
    preco_pl12 = calculate_pl_price(fundamental_data.lpa, pl_targets[1])
    preco_pl15 = calculate_pl_price(fundamental_data.lpa, pl_targets[2])
    
    # Preços por P/VPA alvo
    preco_pvp1 = calculate_pvp_price(fundamental_data.vpa, pvp_targets[0])
    preco_pvp1_5 = calculate_pvp_price(fundamental_data.vpa, pvp_targets[1])
    
    # Preço Bazin
    preco_bazin = calculate_bazin_price(
        fundamental_data.dps, 
        fundamental_data.crescimento_esperado, 
        taxa_desconto
    )
    
    # PEG ratio
    peg_ratio = calculate_peg_ratio(fundamental_data.pl, fundamental_data.crescimento_esperado)
    
    # Lista de preços justos válidos (excluindo zeros)
    precos_justos = [
        preco_graham, preco_dividendos, preco_pl10, preco_pl12, preco_pl15,
        preco_pvp1, preco_pvp1_5, preco_bazin
    ]
    precos_validos = [p for p in precos_justos if p > 0]
    
    # Média dos preços justos
    media_precos_justos = sum(precos_validos) / len(precos_validos) if precos_validos else 0.0
    
    # Flags de avaliação
    desconto = fundamental_data.preco_atual < media_precos_justos
    caro = all(fundamental_data.preco_atual > p for p in precos_validos) if precos_validos else False
    
    # Margens de segurança
    margem_seguranca = {
        "graham": calculate_margin_of_safety(fundamental_data.preco_atual, preco_graham),
        "dividendos": calculate_margin_of_safety(fundamental_data.preco_atual, preco_dividendos),
        "pl10": calculate_margin_of_safety(fundamental_data.preco_atual, preco_pl10),
        "pl12": calculate_margin_of_safety(fundamental_data.preco_atual, preco_pl12),
        "pl15": calculate_margin_of_safety(fundamental_data.preco_atual, preco_pl15),
        "pvp1": calculate_margin_of_safety(fundamental_data.preco_atual, preco_pvp1),
        "pvp1_5": calculate_margin_of_safety(fundamental_data.preco_atual, preco_pvp1_5),
        "bazin": calculate_margin_of_safety(fundamental_data.preco_atual, preco_bazin),
        "media": calculate_margin_of_safety(fundamental_data.preco_atual, media_precos_justos)
    }
    
    return ValuationResult(
        ticker=ticker,
        preco_atual=fundamental_data.preco_atual,
        preco_graham=preco_graham,
        preco_dividendos=preco_dividendos,
        preco_pl10=preco_pl10,
        preco_pl12=preco_pl12,
        preco_pl15=preco_pl15,
        preco_pvp1=preco_pvp1,
        preco_pvp1_5=preco_pvp1_5,
        preco_bazin=preco_bazin,
        peg_ratio=peg_ratio,
        media_precos_justos=media_precos_justos,
        desconto=desconto,
        caro=caro,
        margem_seguranca=margem_seguranca
    )


def get_real_fundamental_data(ticker: str, client: Optional[OpLabClient] = None) -> FundamentalData:
    """
    Busca dados fundamentais reais via APIs.
    Fallback para dados simulados se APIs não disponíveis.
    """
    client = client or OpLabClient()
    
    try:
        # Buscar preço atual real
        preco_atual = client.get_underlying_price(ticker)
        
        if preco_atual <= 0:
            raise ValueError(f"Preço não encontrado para {ticker}")
        
        # TODO: Integrar com APIs de fundamentos reais
        # Por enquanto, usar dados simulados com preço real
        sample_data = get_sample_fundamental_data(ticker)
        sample_data.preco_atual = preco_atual
        
        # Recalcular métricas baseadas no preço real
        if sample_data.lpa > 0:
            sample_data.pl = preco_atual / sample_data.lpa
        if sample_data.vpa > 0:
            sample_data.pvp = preco_atual / sample_data.vpa
        if sample_data.dps > 0:
            sample_data.dividend_yield = (sample_data.dps / preco_atual) * 100
        if sample_data.lpa > 0 and sample_data.dps > 0:
            sample_data.payout = (sample_data.dps / sample_data.lpa) * 100
        if sample_data.vpa > 0 and sample_data.lpa > 0:
            sample_data.roe = (sample_data.lpa / sample_data.vpa) * 100
        if sample_data.crescimento_esperado > 0:
            sample_data.peg_ratio = sample_data.pl / sample_data.crescimento_esperado
        
        return sample_data
        
    except Exception as e:
        print(f"Erro ao buscar dados reais para {ticker}: {e}")
        # Fallback para dados simulados
        return get_sample_fundamental_data(ticker)


def get_sample_fundamental_data(ticker: str) -> FundamentalData:
    """
    Retorna dados fundamentais de exemplo para teste.
    Em produção, estes dados viriam de uma API de fundamentos.
    """
    # Dados de exemplo para BBAS3
    sample_data = {
        "BBAS3": FundamentalData(
            ticker="BBAS3",
            preco_atual=22.16,
            lpa=2.85,
            vpa=18.50,
            dps=1.20,
            dividend_yield=5.4,
            payout=42.1,
            crescimento_esperado=8.0,
            roe=15.4,
            pl=7.8,
            pvp=1.2,
            peg_ratio=0.98
        ),
        "PETR4": FundamentalData(
            ticker="PETR4",
            preco_atual=28.45,
            lpa=4.20,
            vpa=15.80,
            dps=2.10,
            dividend_yield=7.4,
            payout=50.0,
            crescimento_esperado=5.0,
            roe=26.6,
            pl=6.8,
            pvp=1.8,
            peg_ratio=1.36
        ),
        "VALE3": FundamentalData(
            ticker="VALE3",
            preco_atual=58.20,
            lpa=8.90,
            vpa=25.40,
            dps=4.50,
            dividend_yield=7.7,
            payout=50.6,
            crescimento_esperado=3.0,
            roe=35.0,
            pl=6.5,
            pvp=2.3,
            peg_ratio=2.17
        )
    }
    
    return sample_data.get(ticker.upper(), FundamentalData(
        ticker=ticker.upper(),
        preco_atual=0.0,
        lpa=0.0,
        vpa=0.0,
        dps=0.0,
        dividend_yield=0.0,
        payout=0.0,
        crescimento_esperado=0.0,
        roe=0.0,
        pl=0.0,
        pvp=0.0,
        peg_ratio=0.0
    ))


def analyze_multiple_tickers(
    tickers: List[str],
    yield_min: float = 6.0,
    pl_targets: List[float] = None,
    pvp_targets: List[float] = None,
    taxa_desconto: float = 6.0,
    client: Optional[OpLabClient] = None
) -> List[ValuationResult]:
    """
    Analisa múltiplos tickers e retorna ordenado por maior desconto.
    Usa dados reais quando disponíveis.
    """
    results = []
    client = client or OpLabClient()
    
    for ticker in tickers:
        try:
            # Tentar buscar dados reais primeiro
            fundamental_data = get_real_fundamental_data(ticker, client)
            
            if fundamental_data.preco_atual > 0:  # Só inclui se tem dados
                result = analyze_fundamentals(
                    ticker, fundamental_data, yield_min, pl_targets, pvp_targets, taxa_desconto
                )
                results.append(result)
        except Exception as e:
            print(f"Erro ao analisar {ticker}: {e}")
            continue
    
    # Ordena por maior margem de segurança (desconto)
    results.sort(key=lambda x: x.margem_seguranca["media"], reverse=True)
    
    return results


__all__ = [
    "FundamentalData",
    "ValuationResult", 
    "analyze_fundamentals",
    "analyze_multiple_tickers",
    "get_sample_fundamental_data"
]
