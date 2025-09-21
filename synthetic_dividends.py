from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Optional
import math

import pandas as pd
from scipy.stats import norm

from oplab_client import OpLabClient


Strategy = Literal["CALL coberta", "PUT coberta"]


@dataclass
class Suggestion:
    ticker: str
    strategy: Strategy
    strike: float
    distance_pct: float
    expiration: datetime
    premium: float
    delta: float
    exercise_prob: Optional[float]
    expected_return_pct: Optional[float]
    rationale: str


def _get_option_price(row: pd.Series, option_type: str) -> float:
    """Retorna o preço relevante para a estratégia de dividendos sintéticos.
    
    Para CALLs (vender): precisa de compradores (ask > 0)
    Para PUTs (vender): precisa de compradores (bid > 0)
    
    Retorna NaN se não há liquidez para a estratégia.
    """
    bid = row.get("bid")
    ask = row.get("ask")
    
    if option_type == "CALL":
        # Para vender CALL, precisamos de compradores (ask > 0)
        if pd.notna(ask) and ask > 0:
            return float(ask)
        # Sem compradores = sem liquidez
        return float("nan")
    else:  # PUT
        # Para vender PUT, precisamos de compradores (bid > 0)
        if pd.notna(bid) and bid > 0:
            return float(bid)
        # Sem compradores = sem liquidez
        return float("nan")


def _calculate_delta(spot: float, strike: float, time_to_expiry: float, 
                    risk_free_rate: float = 0.12, volatility: float = 0.25, option_type: str = "CALL") -> float:
    """Calcula delta usando Black-Scholes simplificado."""
    if time_to_expiry <= 0:
        return 0.5 if option_type == "CALL" else -0.5
    
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
    
    if option_type == "CALL":
        return norm.cdf(d1)
    else:  # PUT
        return norm.cdf(d1) - 1

def _prob_exercise_from_delta(delta: float, option_type: str) -> float:
    """Converte delta em probabilidade de exercício aproximada."""
    abs_delta = abs(delta)
    prob = max(0.0, min(100.0, abs_delta * 100))
    # Arredondar para 1% para melhor legibilidade
    return round(prob)

def _get_risk_level(prob_exercise: float) -> str:
    """Classifica o nível de risco baseado na probabilidade de exercício."""
    if prob_exercise < 5:
        return "baixo"
    elif prob_exercise < 15:
        return "moderado"
    else:
        return "alto"

def _annualize_return(return_pct: float, days: int) -> float:
    """Anualiza um retorno baseado no número de dias."""
    if days <= 0:
        return 0.0
    # Fórmula: (1 + retorno)^(365/dias) - 1
    annual_return = ((1 + return_pct/100) ** (365/days) - 1) * 100
    # Debug para valores suspeitos
    if annual_return > 50:  # Se retorno anual > 50%, algo está errado
        print(f"DEBUG: return_pct={return_pct}, days={days}, annual={annual_return}")
    return annual_return


def _build_rationale(option_type: str, distance_pct: float, premium: float, spot: float, 
                     days_to_expiry: int, prob_exercise: float) -> str:
    """Constrói justificativa concisa em 1 linha."""
    
    if option_type == "CALL":
        yield_pct = (premium / spot) * 100
        risk_level = _get_risk_level(prob_exercise)
        return f"CALL gera {yield_pct:.2f}% em {days_to_expiry}d, risco {risk_level}"
    else:
        # Para PUT, yield sobre capital comprometido (strike)
        strike = spot * (1 + distance_pct/100)
        yield_pct = (premium / strike) * 100
        return f"PUT gera {yield_pct:.2f}% em {days_to_expiry}d, você só compra {abs(distance_pct):.0f}% mais barato"


def find_synthetic_dividend_options(
    ticker: str,
    *,
    client: Optional[OpLabClient] = None,
    min_volume: int = 100,
    min_days: int = 30,
    max_days: int = 45,
    call_min_distance_pct: float = 15.0,
    call_max_delta: float = 0.20,
    put_max_distance_pct: float = -10.0,
    put_min_delta: float = -0.20,
) -> pd.DataFrame:
    client = client or OpLabClient()

    spot = client.get_underlying_price(ticker)
    chain = client.get_option_chain(ticker)

    if chain.empty:
        return pd.DataFrame(columns=[
            "Opção", "Estratégia", "Strike", "Validade", "Prêmio (R$)", 
            "Retorno (%)", "Retorno a.a. (%)", "Prob. Exercício (%)", "Contratos ativos", "Justificativa"
        ])

    # Normalizações necessárias
    chain = chain.copy()
    chain = chain[chain["expiration"].notna()]
    chain = chain[chain["strike"].notna()]
    chain = chain[chain["option_type"].isin(["CALL", "PUT"])]

    # Janela de vencimento parametrizada
    today = datetime.utcnow().date()
    min_dt = today + timedelta(days=min_days)
    max_dt = today + timedelta(days=max_days)
    chain["expiration_date"] = pd.to_datetime(chain["expiration"]).dt.date
    chain = chain[(chain["expiration_date"] >= min_dt) & (chain["expiration_date"] <= max_dt)]

    if chain.empty:
        return pd.DataFrame(columns=[
            "Opção", "Estratégia", "Strike", "Validade", "Prêmio (R$)", 
            "Retorno (%)", "Retorno a.a. (%)", "Prob. Exercício (%)", "Contratos ativos", "Justificativa"
        ])

    # Cálculos auxiliares - preço baseado na estratégia
    chain["premium"] = chain.apply(lambda row: _get_option_price(row, row["option_type"]), axis=1)
    
    # Filtrar opções sem liquidez (premium = NaN)
    chain = chain[chain["premium"].notna()].copy()
    
    if chain.empty:
        return pd.DataFrame(columns=[
            "Opção", "Estratégia", "Strike", "Validade", "Prêmio (R$)", 
            "Retorno (%)", "Retorno a.a. (%)", "Prob. Exercício (%)", "Contratos ativos", "Justificativa"
        ])
    
    chain["distance_pct"] = (chain["strike"] - spot) / spot * 100.0
    
    # Dias até vencimento
    chain["expiration_dt"] = pd.to_datetime(chain["expiration"], errors="coerce")
    chain["days_to_expiry"] = (chain["expiration_dt"] - pd.Timestamp(today)).dt.days
    
    # Calcular delta usando Black-Scholes (assumindo volatilidade de 25% e taxa de 12%)
    chain["time_to_expiry_years"] = chain["days_to_expiry"] / 365.0
    chain["delta"] = chain.apply(
        lambda row: _calculate_delta(spot, row["strike"], row["time_to_expiry_years"], 
                                   option_type=row["option_type"]), 
        axis=1
    )
    
    # Probabilidade de exercício a partir do delta calculado
    chain["exercise_prob"] = chain.apply(
        lambda row: _prob_exercise_from_delta(row["delta"], row["option_type"]), 
        axis=1
    )

    # Filtro de liquidez (volume em contratos)
    chain["volume"] = pd.to_numeric(chain.get("volume", pd.Series([float("nan")] * len(chain))), errors="coerce")
    
    # Aplicar filtro de volume (só aplica se o valor mínimo for > 0)
    liquid = chain.copy()
    if min_volume > 0:
        liquid = liquid[liquid["volume"] >= min_volume]

    # CALL coberta
    calls = liquid[liquid["option_type"] == "CALL"].copy()
    calls = calls[(calls["distance_pct"] >= float(call_min_distance_pct))]
    calls = calls[(calls["delta"].astype(float) <= float(call_max_delta)) | (calls["delta"].isna())]

    # PUT coberta
    puts = liquid[liquid["option_type"] == "PUT"].copy()
    puts = puts[(puts["distance_pct"] <= float(put_max_distance_pct))]
    puts = puts[(puts["delta"].astype(float) >= float(put_min_delta)) | (puts["delta"].isna())]

    # Monta sugestões
    def to_rows(df: pd.DataFrame, strategy: Strategy) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[
                "Ativo", "Ticker (opção)", "Estratégia", "Strike", "Distância (%)", "Vencimento",
                "Prêmio (R$)", "Delta", "Prob. Exercício", "Retorno (%)", "Justificativa"
            ])
        strike_num = pd.to_numeric(df["strike"], errors="coerce")
        distance_num = pd.to_numeric(df["distance_pct"], errors="coerce")
        premium_num = pd.to_numeric(df["premium"], errors="coerce")
        delta_num = pd.to_numeric(df["delta"], errors="coerce")
        prob_num = pd.to_numeric(df["exercise_prob"], errors="coerce")

        volume_num = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        days_to_expiry_num = pd.to_numeric(df.get("days_to_expiry", 0), errors="coerce").fillna(0)
        
        # Break-even point e retornos
        if strategy == "CALL":
            breakeven = strike_num + premium_num
            retorno_pct = (premium_num / spot * 100.0).round(2)
        else:  # PUT
            breakeven = strike_num - premium_num
            # Para PUT, retorno sobre capital comprometido (strike)
            retorno_pct = (premium_num / strike_num * 100.0).round(2)
        
        # Retorno anualizado
        retorno_annual = []
        for i, (_, row) in enumerate(df.iterrows()):
            annual_return = _annualize_return(retorno_pct.iloc[i], row.days_to_expiry)
            retorno_annual.append(annual_return)
        retorno_annual = pd.Series(retorno_annual, index=df.index)
        
        # Valor financeiro do volume (volume × prêmio × 100 contratos)
        volume_financeiro = (volume_num * premium_num * 100).round(0)
        
        # Theta aproximado (decay temporal)
        theta_aproximado = (premium_num / days_to_expiry_num).round(3)
        
        # Preparar justificativas
        justificativas = []
        for i, (_, row) in enumerate(df.iterrows()):
            justificativa = _build_rationale(
                row.option_type, row.distance_pct, row.premium, spot, 
                row.days_to_expiry, row.exercise_prob
            )
            justificativas.append(justificativa)

        out = pd.DataFrame({
            "Opção": df["symbol"].astype(str),
            "Estratégia": strategy,
            "Strike": strike_num.round(2),
            "Validade": pd.to_datetime(df["expiration"]).dt.date,
            "Prêmio (R$)": premium_num.round(2),
            "Retorno (%)": retorno_pct,
            "Retorno a.a. (%)": retorno_annual,
            "Prob. Exercício (%)": prob_num.astype(int),
            "Contratos ativos": volume_num.astype(int),
            "Justificativa": justificativas,
        })
        # Ordena por maior retorno anualizado primeiro
        out = out.sort_values(by=["Retorno a.a. (%)"], ascending=[False])
        return out

    call_rows = to_rows(calls, "CALL")
    put_rows = to_rows(puts, "PUT")

    # Concatenar com PUTs primeiro (geralmente melhores retornos sintéticos)
    # Filtrar apenas DataFrames não vazios para evitar problemas de concatenação
    non_empty_rows = [df for df in [put_rows, call_rows] if not df.empty]
    if non_empty_rows:
        result = pd.concat(non_empty_rows, ignore_index=True)
    else:
        result = pd.DataFrame(columns=[
            "Opção", "Estratégia", "Strike", "Validade", "Prêmio (R$)", 
            "Retorno (%)", "Retorno a.a. (%)", "Prob. Exercício (%)", "Contratos ativos", "Justificativa"
        ])
    
    # Ordenação final global: por maior retorno anualizado
    if not result.empty:
        result = result.sort_values(by=["Retorno a.a. (%)"], ascending=[False])
    return result


__all__ = ["find_synthetic_dividend_options"]


