import streamlit as st
import pandas as pd
import numpy as np
from params import HORIZON_PRESETS
from data import get_price_history

st.title("Teste Minimal")

if st.button("Testar"):
    try:
        st.write("Passo 1: Buscando dados...")
        ticker = "PETR4"
        p = HORIZON_PRESETS["3-6 meses"]
        
        historical_data = get_price_history(ticker, p.history_days)
        st.write(f"✅ Dados obtidos: {historical_data.shape}")
        
        st.write("Passo 2: Testando SMAs básicas...")
        close_series = historical_data['close']
        st.write(f"Close series type: {type(close_series)}")
        
        sma_short = close_series.rolling(window=20).mean()
        st.write(f"SMA short calculada")
        
        sma_long = close_series.rolling(window=100).mean()
        st.write(f"SMA long calculada")
        
        current_price = float(close_series.iloc[-1])
        st.write(f"Preço atual: {current_price}")
        
        sma_short_val = float(sma_short.iloc[-1])
        st.write(f"SMA 20: {sma_short_val}")
        
        sma_long_val = float(sma_long.iloc[-1])
        st.write(f"SMA 100: {sma_long_val}")
        
        st.write("Passo 3: Testando comparação simples...")
        if current_price > sma_long_val:
            st.write("✅ Preço acima da SMA longa")
        else:
            st.write("❌ Preço abaixo da SMA longa")
            
        st.success("✅ Teste completo sem erros!")
        
    except Exception as e:
        import traceback
        st.error(f"Erro: {e}")
        st.code(traceback.format_exc())
