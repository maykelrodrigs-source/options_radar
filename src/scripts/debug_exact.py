#!/usr/bin/env python3
"""
Executa EXATAMENTE o mesmo código que o Streamlit, linha por linha
"""
import pandas as pd
import numpy as np
from src.core.decision.params import HORIZON_PRESETS
from src.core.data.data import get_price_history
from src.core.professional.indicators_simple import compute_indicators_simple as compute_indicators
from src.core.decision.decision import direction_signal, Direction
from src.core.data.oplab_client import OpLabClient

def debug_exact_flow():
    print("=== EXECUTANDO FLUXO EXATO DO STREAMLIT ===")
    
    try:
        # LINHA POR LINHA IGUAL AO STREAMLIT
        ticker = "PETR4"
        horizon = "3-6 meses"
        
        print(f"DEBUG: Starting analysis for {ticker} with horizon {horizon}")
        
        # Busca dados
        client = OpLabClient()
        current_price = client.get_underlying_price(ticker.strip().upper())
        print(f"DEBUG: Got current price: {current_price}")
        
        # Parâmetros do horizonte selecionado
        p = HORIZON_PRESETS[horizon]
        
        # Dados históricos com cache baseado no horizonte
        historical_data = get_price_history(ticker.strip().upper(), p.history_days)
        print(f"DEBUG: Got price history, df shape: {historical_data.shape}")
        
        # Calcula indicadores com janelas dinâmicas (inclui volume se disponível)
        volume_data = historical_data['volume'] if 'volume' in historical_data.columns else None
        print(f"DEBUG: About to compute indicators")
        indicators = compute_indicators(historical_data['close'], p, volume_data)
        print(f"DEBUG: Computed indicators: {list(indicators.keys())}")
        
        # Adiciona períodos para o motivo
        indicators['sma_short_period'] = p.sma_short
        indicators['sma_long_period'] = p.sma_long
        
        # Determina sinal de direção
        print(f"DEBUG: About to call direction_signal")
        direction, confidence, score, reason = direction_signal(indicators, p.weights)
        print(f"DEBUG: Got direction signal: {direction}, confidence: {confidence}, score: {score}")
        
        # Debug temporário - conversão segura para float
        try:
            price_val = float(indicators['price']) if indicators['price'] is not None else 0.0
            sma_long_val = float(indicators['sma_long']) if indicators['sma_long'] is not None else 0.0
            rsi_val = float(indicators['rsi']) if indicators['rsi'] is not None else 0.0
            macd_val = float(indicators['macd_hist']) if indicators['macd_hist'] is not None else 0.0
            debug_info = f"🔧 Debug - {ticker} ({horizon}): Preço={price_val:.2f}, SMA{p.sma_long}={sma_long_val:.2f}, RSI={rsi_val:.1f}, MACD={macd_val:.3f}"
        except Exception as debug_e:
            debug_info = f"🔧 Debug - Erro na formatação: {debug_e}"
            
        if indicators.get('adx') is not None:
            adx_val = float(indicators['adx'])
            debug_info += f", ADX={adx_val:.1f}"
        if indicators.get('obv') is not None:
            obv_val = float(indicators['obv'])
            debug_info += f", OBV={obv_val:.1f}"
        debug_info += f" → {direction} ({confidence}%)"
        
        print(f"DEBUG INFO: {debug_info}")
        
        # Simula o que o Streamlit faria
        print(f"✅ ANÁLISE CONCLUÍDA: {direction} (Confiança: {confidence}%)")
        print(f"Motivo: {reason}")
        
        # Testa comparação com Direction enum
        if direction != Direction.NEUTRAL.value:
            print(f"📈 Direção: {direction} - Busca de opções seria executada")
        else:
            print("🔍 Sem sinal forte. Recomendaria apenas Dividendos Sintéticos.")
            
        print("\n✅ FLUXO COMPLETO EXECUTADO SEM ERROS!")
        
    except Exception as e:
        print(f"❌ ERRO ENCONTRADO: {e}")
        import traceback
        traceback.print_exc()
        
        # Vamos investigar cada variável
        print("\n=== INVESTIGANDO VARIÁVEIS ===")
        try:
            print(f"ticker type: {type(ticker)}")
            print(f"horizon type: {type(horizon)}")
            print(f"current_price type: {type(current_price)}")
            print(f"p type: {type(p)}")
            print(f"historical_data type: {type(historical_data)}")
            print(f"volume_data type: {type(volume_data)}")
            print(f"indicators type: {type(indicators)}")
            if 'indicators' in locals():
                for key, value in indicators.items():
                    print(f"  indicators['{key}'] type: {type(value)}, value: {value}")
            if 'direction' in locals():
                print(f"direction type: {type(direction)}, value: {direction}")
            if 'confidence' in locals():
                print(f"confidence type: {type(confidence)}, value: {confidence}")
        except Exception as debug_e:
            print(f"Erro no debug: {debug_e}")

if __name__ == "__main__":
    debug_exact_flow()
