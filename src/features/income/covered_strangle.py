import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import math

from src.core.data.oplab_client import OpLabClient
from src.features.income.income_opportunities import (
    calculate_black_scholes_delta,
    calculate_exercise_probability,
)


def _within_price_band(strike: float, spot_price: float, band: float = 0.30) -> bool:
    return (spot_price * (1 - band)) <= float(strike) <= (spot_price * (1 + band))


def _get_effective_volume(row: pd.Series) -> float:
    # Prioriza média diária se existir, senão usa 'volume'
    for col in ["avg_daily_volume", "average_daily_volume", "avg_volume", "volume"]:
        if col in row and pd.notna(row[col]):
            try:
                return float(row[col])
            except Exception:
                continue
    return 0.0


def _calc_spread_pct(row: pd.Series) -> Optional[float]:
    bid = row.get("bid", None)
    ask = row.get("ask", None)
    try:
        bid = float(bid) if bid is not None else None
        ask = float(ask) if ask is not None else None
    except Exception:
        return None
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid


def _compute_probabilities_and_filters(
    df: pd.DataFrame,
    spot_price: float,
    dte_min: int,
    dte_max: int,
    min_prob_success: float,
    min_premium: float,
    option_type: str,
    debug: bool = False,
) -> pd.DataFrame:
    if df.empty:
        if debug:
            print(f"DEBUG: {option_type} - DataFrame vazio inicial")
        return df

    data = df.copy()
    initial_count = len(data)
    if debug:
        print(f"DEBUG: {option_type} - Inicial: {initial_count} opções")

    # Normaliza DTE
    if 'expiration' in data.columns:
        data['days_to_exp'] = (data['expiration'] - datetime.now()).dt.days
    else:
        data['days_to_exp'] = (pd.to_datetime(data['expiration']) - datetime.now()).dt.days

    # Filtra janela de DTE
    data = data[(data['days_to_exp'] >= dte_min) & (data['days_to_exp'] <= dte_max)]
    if debug:
        print(f"DEBUG: {option_type} - Após DTE ({dte_min}-{dte_max}): {len(data)} opções")
    if data.empty:
        return data

    # Calcula delta efetivo
    if 'delta' not in data.columns:
        data['delta'] = float('nan')
    data['time_to_expiry_years'] = data['days_to_exp'] / 365.0

    def get_delta(row):
        if pd.notna(row['delta']) and isinstance(row['delta'], (float, int)) and row['delta'] != 0:
            return float(row['delta'])
        return calculate_black_scholes_delta(
            spot_price=spot_price,
            strike=row['strike'],
            time_to_expiry_years=row['time_to_expiry_years'],
            option_type=option_type
        )

    data['delta_eff'] = data.apply(get_delta, axis=1)

    # Faixa de preço (±30%) e delta por tipo - mais flexível
    if option_type == 'CALL':
        data = data[(data['strike'] > spot_price) & (data['delta_eff'].between(0.05, 0.50))]  # Mais flexível
    else:
        data = data[(data['strike'] < spot_price) & (data['delta_eff'].between(-0.50, -0.05))]  # Mais flexível
    data = data[data['strike'].apply(lambda s: _within_price_band(s, spot_price, 0.30))]
    if debug:
        print(f"DEBUG: {option_type} - Após delta e faixa preço: {len(data)} opções")
    if data.empty:
        return data

    # Liquidez: volume médio diário ≥ 50 (mais flexível)
    data['eff_volume'] = data.apply(_get_effective_volume, axis=1)
    data = data[data['eff_volume'] >= 50]  # Reduzido de 100 para 50
    if debug:
        print(f"DEBUG: {option_type} - Após volume (≥50): {len(data)} opções")
    if data.empty:
        return data

    # Spread bid-ask ≤ 10% (mais flexível)
    data['spread_pct'] = data.apply(_calc_spread_pct, axis=1)
    data = data[(~data['spread_pct'].isna()) & (data['spread_pct'] <= 0.10)]  # Aumentado de 5% para 10%
    if debug:
        print(f"DEBUG: {option_type} - Após spread (≤10%): {len(data)} opções")
    if data.empty:
        return data

    # Probabilidade de sucesso
    data['prob_exercicio'] = data.apply(
        lambda r: calculate_exercise_probability(r['delta_eff'], option_type), axis=1
    )
    data['prob_sucesso'] = 100.0 - data['prob_exercicio']
    data = data[data['prob_sucesso'] >= (min_prob_success * 100.0)]
    if debug:
        print(f"DEBUG: {option_type} - Após prob sucesso (≥{min_prob_success*100:.0f}%): {len(data)} opções")
    if data.empty:
        return data

    # Prêmio: usar bid como referência; descartar anômalos e abaixo do mínimo
    if 'bid' not in data.columns:
        data['bid'] = float('nan')
    data = data[pd.to_numeric(data['bid'], errors='coerce').fillna(0) >= float(min_premium)]
    if debug:
        print(f"DEBUG: {option_type} - Após prêmio mínimo (≥{min_premium}): {len(data)} opções")
    if data.empty:
        return data

    # Remove prêmios anômalos (>30% do spot - mais flexível)
    data = data[data['bid'] <= (0.30 * spot_price)]  # Aumentado de 20% para 30%
    if debug:
        print(f"DEBUG: {option_type} - Após prêmio anômalo (≤30% spot): {len(data)} opções")

    return data


def _select_candidates(
    df: pd.DataFrame,
    option_type: str,
    spot_price: float,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Seleciona melhores strikes priorizando maior bid e proximidade do spot no lado correto.
    """
    if df.empty:
        return df

    data = df.copy()

    if option_type == 'CALL':
        data = data[data['strike'] > spot_price]
        if data.empty:
            return data
        data = data.sort_values(['bid', 'strike'], ascending=[False, True])
    else:
        data = data[data['strike'] < spot_price]
        if data.empty:
            return data
        data = data.sort_values(['bid', 'strike'], ascending=[False, False])

    return data.head(top_n)


def suggest_covered_strangle_for_ticker(
    client: OpLabClient,
    ticker: str,
    quantity_available: int,
    min_prob_success: float = 0.65,
    min_premium: float = 0.20,
    dte_min: int = 21,
    dte_max: int = 35,
    cash_available: Optional[float] = None,
    lot_size: int = 1,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Gera sugestão de Covered Strangle para um único ativo.
    Retorna None se não houver combinação que atenda aos filtros.
    """
    try:
        spot_price = float(client.get_underlying_price(ticker))
        chain = client.get_option_chain(ticker)
        if chain.empty:
            if debug:
                print(f"DEBUG: {ticker} - Chain vazio")
            return None

        calls = chain[chain['option_type'] == 'CALL']
        puts = chain[chain['option_type'] == 'PUT']
        
        if debug:
            print(f"DEBUG: {ticker} - CALLs: {len(calls)}, PUTs: {len(puts)}")

        calls_f = _compute_probabilities_and_filters(
            calls, spot_price, dte_min, dte_max, min_prob_success, min_premium, 'CALL', debug
        )
        puts_f = _compute_probabilities_and_filters(
            puts, spot_price, dte_min, dte_max, min_prob_success, min_premium, 'PUT', debug
        )

        calls_cand = _select_candidates(calls_f, 'CALL', spot_price)
        puts_cand = _select_candidates(puts_f, 'PUT', spot_price)

        if debug:
            print(f"DEBUG: {ticker} - CALLs candidatas: {len(calls_cand)}, PUTs candidatas: {len(puts_cand)}")

        if calls_cand.empty or puts_cand.empty:
            return None

        # Busca o melhor par maximizando prêmio total (bid_call + bid_put) e garantindo put_strike < call_strike
        best_pair: Optional[Tuple[pd.Series, pd.Series]] = None
        best_premio = -1.0
        for _, c in calls_cand.iterrows():
            for _, p in puts_cand.iterrows():
                if p['strike'] < c['strike']:
                    premio_total = float(c.get('bid', 0)) + float(p.get('bid', 0))
                    if premio_total > best_premio:
                        best_premio = premio_total
                        best_pair = (c, p)

        if best_pair is None:
            if debug:
                print(f"DEBUG: {ticker} - Nenhum par válido encontrado")
            return None

        call_row, put_row = best_pair

        # Tamanho dos contratos respeitando ações em carteira e caixa disponível
        max_call_contracts = max(0, int(quantity_available // lot_size))
        if cash_available is not None:
            max_put_contracts = int(max(0, math.floor(cash_available / (float(put_row['strike']) * lot_size))))
        else:
            max_put_contracts = max_call_contracts
        contracts = int(max(0, min(max_call_contracts, max_put_contracts)))

        prob_sucesso_call = float(call_row['prob_sucesso'])
        prob_sucesso_put = float(put_row['prob_sucesso'])

        # Flag qualificada se todos critérios foram atendidos (já filtrados) e contratos > 0
        qualificada = contracts > 0

        premio_call = float(call_row.get('bid', 0.0))
        premio_put = float(put_row.get('bid', 0.0))
        premio_total = premio_call + premio_put
        caixa_utilizado = float(put_row['strike']) * lot_size * max(contracts, 1)
        lucro_estimado = premio_total * lot_size * contracts
        retorno_caixa_pct = (lucro_estimado / caixa_utilizado * 100.0) if caixa_utilizado > 0 else 0.0

        result: Dict[str, Any] = {
            "ticker": ticker,
            "preco_atual": round(spot_price, 2),
            "call_symbol": str(call_row.get('symbol', '')),
            "put_symbol": str(put_row.get('symbol', '')),
            "call_strike": float(call_row['strike']),
            "put_strike": float(put_row['strike']),
            "premio_call": premio_call,
            "premio_put": premio_put,
            "premio_total": round(premio_total, 4),
            "prob_sucesso_call": round(prob_sucesso_call / 100.0, 4),
            "prob_sucesso_put": round(prob_sucesso_put / 100.0, 4),
            "probabilidade_call": round(prob_sucesso_call / 100.0, 4),
            "probabilidade_put": round(prob_sucesso_put / 100.0, 4),
            "range_sucesso": [float(put_row['strike']), float(call_row['strike'])],
            "dte": int(call_row['days_to_exp']),
            "contratos_sugeridos": contracts,
            "retorno_efetivo_pct": round((premio_total / spot_price) * 100.0, 3),
            "cash_necessario_put": round(float(put_row['strike']) * lot_size * max(contracts, 1), 2),
            "caixa_utilizado": round(caixa_utilizado, 2),
            "lucro_estimado": round(lucro_estimado, 2),
            "retorno_caixa_pct": round(retorno_caixa_pct, 2),
            "qualificada": qualificada,
        }
        return result
    except Exception as e:
        if debug:
            print(f"DEBUG: {ticker} - Erro: {e}")
        return None


def generate_covered_strangle(
    portfolio: List[Dict[str, Any]],
    client: OpLabClient,
    min_prob_success: float = 0.65,
    min_premium: float = 0.20,
    dte_min: int = 21,
    dte_max: int = 35,
    lot_size: int = 1,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Gera recomendações de Covered Strangle para uma lista de ativos.

    portfolio: lista de dicts com chaves: ticker, quantidade, caixa (opcional).
    """
    results: List[Dict[str, Any]] = []
    for item in portfolio:
        ticker = str(item.get('ticker', '')).strip().upper()
        if not ticker:
            continue
        qty = int(item.get('quantidade', 0))
        cash = item.get('caixa', None)
        cash_val = float(cash) if cash is not None and cash != "" else None

        rec = suggest_covered_strangle_for_ticker(
            client=client,
            ticker=ticker,
            quantity_available=qty,
            min_prob_success=min_prob_success,
            min_premium=min_premium,
            dte_min=dte_min,
            dte_max=dte_max,
            cash_available=cash_val,
            lot_size=lot_size,
            debug=debug,
        )
        if rec is not None:
            results.append(rec)
    return results
