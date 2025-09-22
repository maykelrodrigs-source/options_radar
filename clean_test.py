import streamlit as st
import pandas as pd
import numpy as np
import traceback

# Desabilita cache para evitar interferências
st.set_page_config(page_title="Clean Test")

st.title("Teste Limpo - Zero Cache")

# Função simples que sabemos que funciona
def simple_calculation():
    # Cria dados simples
    data = {
        'close': [30.0, 31.0, 32.0, 33.0, 34.0, 35.0] * 50,  # 300 pontos
        'volume': [1000, 1100, 1200, 1300, 1400, 1500] * 50
    }
    df = pd.DataFrame(data)
    
    # Cálculos básicos
    sma_short = df['close'].rolling(window=5).mean()
    sma_long = df['close'].rolling(window=20).mean()
    
    current_price = float(df['close'].iloc[-1])
    sma_short_val = float(sma_short.iloc[-1])
    sma_long_val = float(sma_long.iloc[-1])
    
    # Comparações simples
    if current_price > sma_long_val:
        direction = "CALL"
    else:
        direction = "PUT"
        
    return {
        'price': current_price,
        'sma_short': sma_short_val,
        'sma_long': sma_long_val,
        'direction': direction
    }

if st.button("Testar Função Simples"):
    try:
        st.write("Executando cálculo simples...")
        result = simple_calculation()
        st.success(f"✅ Resultado: {result}")
        
    except Exception as e:
        st.error(f"❌ Erro: {e}")
        st.code(traceback.format_exc())

# Agora testa com nossos módulos
if st.button("Testar Nossos Módulos"):
    try:
        st.write("Importando módulos...")
        
        # Importa um por vez
        from params import HORIZON_PRESETS
        st.write("✅ params importado")
        
        # Testa sem cache
        import os
        os.environ['DISABLE_STREAMLIT_CACHE'] = '1'
        
        # Dados históricos simulados (sem API)
        dates = pd.date_range('2024-01-01', periods=260, freq='D')
        prices = np.random.normal(35, 2, 260).cumsum() + 30
        volumes = np.random.randint(1000, 5000, 260)
        
        historical_data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': volumes
        })
        
        st.write(f"✅ Dados simulados: {historical_data.shape}")
        
        # Testa indicadores
        from indicators_simple import compute_indicators_simple
        p = HORIZON_PRESETS["3-6 meses"]
        
        indicators = compute_indicators_simple(
            historical_data['close'], 
            p, 
            historical_data['volume']
        )
        
        st.write(f"✅ Indicadores: {indicators}")
        
        # Testa decisão
        from decision import direction_signal
        direction, confidence, score, reason = direction_signal(indicators, p.weights)
        
        st.success(f"✅ SUCESSO: {direction} ({confidence}%)")
        
    except Exception as e:
        st.error(f"❌ Erro nos módulos: {e}")
        st.code(traceback.format_exc())
