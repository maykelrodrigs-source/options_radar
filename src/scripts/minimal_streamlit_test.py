import streamlit as st
import pandas as pd
import traceback
import sys

st.title("Teste Mínimo - Identificar Erro Exato")

if st.button("Executar Teste Linha por Linha"):
    try:
        st.write("Passo 1: Criando Series simples...")
        series = pd.Series([1, 2, 3, 4, 5])
        st.write(f"Series criada: {type(series)}")
        
        st.write("Passo 2: Testando comparação direta...")
        if series.iloc[-1] > 3:
            st.write("✅ Comparação direta funcionou")
        
        st.write("Passo 3: Testando comparação com float...")
        val = float(series.iloc[-1])
        if val > 3:
            st.write("✅ Comparação com float funcionou")
            
        st.write("Passo 4: Importando módulos do projeto...")
        from params import HORIZON_PRESETS
        p = HORIZON_PRESETS["3-6 meses"]
        st.write(f"✅ Parâmetros carregados: {type(p)}")
        
        st.write("Passo 5: Importando data...")
        from src.core.data.data import get_price_history
        st.write("✅ Módulo data importado")
        
        st.write("Passo 6: Buscando dados históricos...")
        historical_data = get_price_history("PETR4", p.history_days)
        st.write(f"✅ Dados obtidos: {historical_data.shape}")
        
        st.write("Passo 7: Extraindo close series...")
        close = historical_data['close']
        st.write(f"✅ Close series: {type(close)}")
        
        st.write("Passo 8: Importando indicators_simple...")
        from indicators_simple import compute_indicators_simple
        st.write("✅ Módulo indicators_simple importado")
        
        st.write("Passo 9: Calculando indicadores...")
        volume_data = historical_data['volume'] if 'volume' in historical_data.columns else None
        indicators = compute_indicators_simple(close, p, volume_data)
        st.write(f"✅ Indicadores calculados: {type(indicators)}")
        st.write(f"Keys: {list(indicators.keys())}")
        
        st.write("Passo 10: Verificando tipos dos indicadores...")
        for key, value in indicators.items():
            st.write(f"  {key}: {type(value)} = {value}")
            
        st.write("Passo 11: Importando decision...")
        from decision import direction_signal
        st.write("✅ Módulo decision importado")
        
        st.write("Passo 12: Adicionando períodos...")
        indicators['sma_short_period'] = p.sma_short
        indicators['sma_long_period'] = p.sma_long
        st.write("✅ Períodos adicionados")
        
        st.write("Passo 13: Chamando direction_signal...")
        direction, confidence, score, reason = direction_signal(indicators, p.weights)
        st.write(f"✅ Sinal calculado: {direction} ({confidence}%)")
        
        st.success("🎉 TODOS OS PASSOS EXECUTADOS COM SUCESSO!")
        
    except Exception as e:
        st.error(f"❌ ERRO NO PASSO: {e}")
        st.code(traceback.format_exc())
        
        # Informações do sistema
        st.write("=== INFORMAÇÕES DO SISTEMA ===")
        st.write(f"Python: {sys.version}")
        st.write(f"Pandas: {pd.__version__}")
        st.write(f"Streamlit: {st.__version__}")
        
        # Informações das variáveis
        st.write("=== VARIÁVEIS NO ESCOPO ===")
        frame = sys.exc_info()[2].tb_frame
        while frame:
            st.write(f"Frame: {frame.f_code.co_filename}:{frame.f_lineno}")
            for var_name, var_value in frame.f_locals.items():
                if not var_name.startswith('_'):
                    st.write(f"  {var_name}: {type(var_value)}")
            frame = frame.f_back
