#!/usr/bin/env python3
"""
Script para testar e identificar o erro "The truth value of a Series is ambiguous"
"""
import pandas as pd
import numpy as np
from params import HORIZON_PRESETS
from src.core.data.data import get_price_history
from indicators_simple import compute_indicators_simple
from decision import direction_signal

def test_data():
    print("=== TESTANDO DATA ===")
    try:
        ticker = "PETR4"
        p = HORIZON_PRESETS["3-6 meses"]
        
        print(f"Buscando dados históricos para {ticker}...")
        historical_data = get_price_history(ticker, p.history_days)
        print(f"✅ Dados obtidos: shape={historical_data.shape}")
        print(f"✅ Colunas: {historical_data.columns.tolist()}")
        print(f"✅ Tipos: {historical_data.dtypes}")
        return historical_data, p
        
    except Exception as e:
        print(f"❌ ERRO em test_data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_indicators(historical_data, p):
    print("\n=== TESTANDO INDICATORS ===")
    try:
        print("Calculando indicadores...")
        volume_data = historical_data['volume'] if 'volume' in historical_data.columns else None
        indicators = compute_indicators_simple(historical_data['close'], p, volume_data)
        print(f"✅ Indicadores calculados: {indicators}")
        return indicators
        
    except Exception as e:
        print(f"❌ ERRO em test_indicators: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_decision(indicators, p):
    print("\n=== TESTANDO DECISION ===")
    try:
        print("Determinando sinal de direção...")
        direction, confidence, score, reason = direction_signal(indicators, p.weights)
        print(f"✅ Sinal determinado: {direction}, confiança: {confidence}, score: {score}")
        return direction, confidence, score, reason
        
    except Exception as e:
        print(f"❌ ERRO em test_decision: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def main():
    print("Iniciando teste isolado do erro...")
    
    # Teste 1: Dados
    historical_data, p = test_data()
    if historical_data is None:
        print("❌ FALHOU no teste de dados")
        return
    
    # Teste 2: Indicadores
    indicators = test_indicators(historical_data, p)
    if indicators is None:
        print("❌ FALHOU no teste de indicadores")
        return
    
    # Teste 3: Decisão
    direction, confidence, score, reason = test_decision(indicators, p)
    if direction is None:
        print("❌ FALHOU no teste de decisão")
        return
    
    print(f"\n✅ TODOS OS TESTES PASSARAM!")
    print(f"Resultado final: {direction} (confiança: {confidence}%)")

if __name__ == "__main__":
    main()
