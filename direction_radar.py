"""
Página do Radar de Direção - Análise técnica com horizonte temporal.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional

from oplab_client import OpLabClient
from synthetic_dividends import find_synthetic_dividend_options
from technical_analysis import RiskProfile, get_option_parameters_by_direction
from params import HORIZON_PRESETS
from data import get_price_history
from indicators_simple import compute_indicators_simple as compute_indicators
from decision import direction_signal, Direction


def render_direction_radar_page():
    """Renderiza a página do Radar de Direção."""
    st.title("🎯 Radar de Direção")
    st.markdown("Análise técnica para decisão CALL/PUT baseada em indicadores")
    
    # Inputs simples
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input("Ticker", value="PETR4", help="Ex: PETR4, VALE3, ITUB4", key="direction_ticker")
    
    with col2:
        horizon = st.selectbox(
            "Horizonte de Prazo",
            options=list(HORIZON_PRESETS.keys()),
            index=1,  # Default: "3-6 meses"
            help="Define o horizonte temporal para análise técnica",
            key="direction_horizon"
        )
    
    # Botão para análise
    if st.button("🔍 Analisar Direção", type="primary", key="direction_analyze"):
        if not ticker.strip():
            st.error("Digite um ticker válido")
            return
            
        try:
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
            print(f"DEBUG: Computed indicators: {indicators}")
            
            # Adiciona períodos para o motivo
            indicators['sma_short_period'] = p.sma_short
            indicators['sma_long_period'] = p.sma_long
            
            # Determina sinal de direção
            print(f"DEBUG: About to call direction_signal")
            direction, confidence, score, reason = direction_signal(indicators, p.weights)
            print(f"DEBUG: Got direction signal: {direction}, confidence: {confidence}, score: {score}")
            
            # Renderiza resultados
            st.success(f"✅ ANÁLISE CONCLUÍDA: {direction} (Confiança: {confidence}%)")
            st.write(f"Motivo: {reason}")
            # render_analysis_results(indicators, direction, confidence, reason, ticker.strip().upper(), p)  # COMENTADO TEMPORARIAMENTE
            
            # Busca opções se há direção definida
            if direction != Direction.NEUTRAL.value:
                st.info(f"📈 Direção: {direction} - Busca de opções temporariamente desabilitada para debug")
                # direction_enum = Direction(direction)
                # render_option_recommendations(direction_enum, ticker.strip().upper(), horizon, client, indicators['price'])
            else:
                st.info("🔍 Sem sinal forte. Recomendo apenas Dividendos Sintéticos.")
                
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            st.error(f"Erro: {e}")
            st.code(error_traceback, language="python")
            print(f"ERROR: {e}")
            print(f"TRACEBACK: {error_traceback}")


def render_analysis_results(indicators: dict, direction: str, confidence: int, reason: str, ticker: str, p):
    """Renderiza os resultados da análise técnica."""
    st.subheader("📈 Análise Técnica")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price_val = float(indicators['price']) if indicators['price'] is not None else 0.0
        st.metric("Preço Atual", f"R$ {price_val:.2f}")
    
    with col2:
        if indicators['sma_long'] is not None:
            sma_long_val = float(indicators['sma_long'])
            price_val = float(indicators['price'])
            sma_diff = ((price_val - sma_long_val) / sma_long_val) * 100
            st.metric(f"SMA{p.sma_long}", f"R$ {sma_long_val:.2f}", f"{sma_diff:+.1f}%")
        else:
            st.metric(f"SMA{p.sma_long}", "N/A")
    
    with col3:
        if indicators['rsi'] is not None:
            rsi_val = float(indicators['rsi'])
            rsi_color = "🔴" if rsi_val > 70 else "🟡" if rsi_val > 50 else "🟢"
            st.metric(f"RSI({p.rsi_len})", f"{rsi_val:.1f}", rsi_color)
        else:
            st.metric(f"RSI({p.rsi_len})", "N/A")
    
    with col4:
        if indicators['macd_hist'] is not None:
            macd_val = float(indicators['macd_hist'])
            macd_color = "🟢" if macd_val > 0 else "🔴"
            st.metric("MACD Hist", f"{macd_val:.3f}", macd_color)
        else:
            st.metric("MACD Hist", "N/A")
    
    # Sinal de direção
    st.markdown("---")
    
    if direction == Direction.CALL.value:
        st.success(f"📈 **TENDÊNCIA DE ALTA** (Confiança: {confidence}%)")
        st.info("💡 **Recomendação:** CALLs OTM para aproveitar movimento de alta")
        st.caption(f"**Motivo:** {reason}")
        
        # Sugestões de estratégia para CALL
        with st.expander("💡 Estratégias Sugeridas para CALL"):
            st.write("""
            **🎯 Venda de CALL coberta (Renda):**
            - Venda CALL OTM com delta 0.15-0.30
            - Receba prêmio e mantenha ações como colateral
            - Risco: exercício se preço subir acima do strike
            
            **🚀 Compra de CALL (Alavancagem):**
            - Compre CALL OTM com delta 0.20-0.40
            - Aproveite movimentos de alta com capital limitado
            - Risco: perda total do prêmio se preço não subir
            """)
            
    elif direction == Direction.PUT.value:
        st.error(f"📉 **TENDÊNCIA DE BAIXA** (Confiança: {confidence}%)")
        st.info("💡 **Recomendação:** PUTs OTM para proteção contra queda")
        st.caption(f"**Motivo:** {reason}")
        
        # Sugestões de estratégia para PUT
        with st.expander("💡 Estratégias Sugeridas para PUT"):
            st.write("""
            **💰 Venda de PUT OTM (Renda):**
            - Venda PUT OTM com delta 0.15-0.30
            - Receba prêmio e fique pronto para comprar ações mais baratas
            - Risco: exercício se preço cair abaixo do strike
            
            **🛡️ Compra de PUT (Proteção):**
            - Compre PUT OTM com delta 0.20-0.40
            - Proteja carteira contra quedas significativas
            - Risco: perda total do prêmio se preço não cair
            """)
            
    else:
        st.warning(f"⚖️ **SINAL NEUTRO** (Confiança: {confidence}%)")
        st.info("💡 **Recomendação:** Apenas Dividendos Sintéticos")
        st.caption(f"**Motivo:** {reason}")
    
    # Detalhes da análise
    with st.expander("🔍 Detalhes da Análise"):
        st.write(f"""
        **Horizonte:** {list(HORIZON_PRESETS.keys())[list(HORIZON_PRESETS.values()).index(p)]}
        **Período histórico:** {p.history_days} dias
        **Médias móveis:** {p.sma_short}/{p.sma_long} períodos
        **RSI:** {p.rsi_len} períodos
        **MACD:** {p.macd_fast}/{p.macd_slow}/{p.macd_signal}
        **Pesos:** Tendência={p.weights['trend']:.0%}, Momentum={p.weights['momentum']:.0%}, Volume={p.weights['volume']:.0%}
        """)
        
        # Mostra indicadores específicos por horizonte
        if indicators.get('adx') is not None:
            adx_val = float(indicators['adx'])
            adx_desc = '(Tendência forte)' if adx_val > 25 else '(Tendência fraca)' if adx_val < 15 else '(Tendência moderada)'
            st.write(f"**ADX:** {adx_val:.1f} {adx_desc}")
        
        if indicators.get('obv') is not None:
            obv_val = float(indicators['obv'])
            obv_desc = 'Acumulação' if obv_val > 0 else 'Distribuição'
            st.write(f"**OBV:** {obv_desc}")
        
        if indicators.get('vol_ratio') is not None:
            vol_val = float(indicators['vol_ratio'])
            vol_desc = '(Alto)' if vol_val > 1.2 else '(Baixo)' if vol_val < 0.8 else '(Normal)'
            st.write(f"**Volume Ratio:** {vol_val:.1f}x {vol_desc}")




def render_option_recommendations(direction: Direction, ticker: str, horizon: str, client: OpLabClient, current_price: float):
    """Renderiza recomendações de opções."""
    st.subheader("🎯 Opções Recomendadas")
    
    # Mapeia horizonte para RiskProfile (compatibilidade)
    horizon_to_risk = {
        "1-4 semanas": RiskProfile.SHORT_TERM,
        "3-6 meses": RiskProfile.MEDIUM_TERM,
        "6-12 meses": RiskProfile.LONG_TERM,
    }
    
    risk_profile = horizon_to_risk.get(horizon, RiskProfile.MEDIUM_TERM)
    
    # Obtém parâmetros baseados no perfil
    option_params = get_option_parameters_by_direction(
        direction, current_price, risk_profile
    )
    
    if not option_params:
        st.warning("Não foi possível gerar parâmetros para busca de opções.")
        return
    
    # Mostra parâmetros sendo usados
    distance = option_params.get('call_min_distance_pct', option_params.get('put_max_distance_pct', 0))
    delta = option_params.get('call_max_delta', option_params.get('put_min_delta', 0))
    min_days = option_params.get('min_days', 0)
    max_days = option_params.get('max_days', 0)
    min_volume = option_params.get('min_volume', 0)
    
    st.caption(f"🔧 Parâmetros: Distância: {distance}% | Delta: {delta} | Prazo: {min_days}-{max_days}d | Volume: {min_volume}")
    
    # Debug: mostra todos os parâmetros
    with st.expander("🔍 Debug - Parâmetros de Busca"):
        st.write("**Parâmetros completos:**")
        for key, value in option_params.items():
            st.write(f"- {key}: {value}")
        st.write(f"**Direção:** {direction.value}")
        st.write(f"**Preço atual:** R$ {current_price:.2f}")
    
    # Busca opções
    try:
        st.caption("🔍 Buscando opções...")
        
        # Debug: mostra parâmetros antes da busca
        st.caption(f"🔧 Buscando com: min_volume={option_params.get('min_volume', 10)}, min_days={option_params.get('min_days', 15)}, max_days={option_params.get('max_days', 90)}")
        st.caption(f"🔧 CALL: distance_pct={option_params.get('call_min_distance_pct', 0)}, delta={option_params.get('call_max_delta', 0)}")
        st.caption(f"🔧 PUT: distance_pct={option_params.get('put_max_distance_pct', 0)}, delta={option_params.get('put_min_delta', 0)}")
        
        # Converte parâmetros antigos para novos
        max_exercise_prob = 20.0  # Default
        if 'call_max_delta' in option_params and 'put_min_delta' in option_params:
            # Usa a menor probabilidade entre CALL e PUT
            call_prob = abs(option_params.get('call_max_delta', 0)) * 100
            put_prob = abs(option_params.get('put_min_delta', 0)) * 100
            max_exercise_prob = min(call_prob, put_prob) if call_prob > 0 and put_prob > 0 else 20.0
        
        df = find_synthetic_dividend_options(
            ticker,
            client=client,
            min_volume=option_params.get("min_volume", 10),
            min_days=option_params.get("min_days", 15),
            max_days=option_params.get("max_days", 90),
            max_exercise_prob=max_exercise_prob,
            option_types="Ambas (CALL + PUT)",
        )
        
        st.caption(f"📊 Total de opções encontradas: {len(df)}")
        
        # Debug: mostra opções antes do filtro
        if not df.empty:
            with st.expander("🔍 Debug - Opções antes do filtro"):
                st.write(f"**Opções disponíveis:** {len(df)}")
                if len(df) > 0:
                    st.write("**Estratégias encontradas:**", df["Estratégia"].unique())
                    st.write("**Primeiras 5 opções:**")
                    st.dataframe(df.head()[["Opção", "Estratégia", "Strike", "Validade", "Prêmio (R$)"]])
        
        # Filtra apenas o tipo da direção
        if direction == Direction.CALL:
            df = df[df["Estratégia"] == "CALL"]
            st.caption(f"📈 CALLs após filtro: {len(df)}")
        else:
            df = df[df["Estratégia"] == "PUT"]
            st.caption(f"📉 PUTs após filtro: {len(df)}")
        
        if df.empty:
            st.warning(f"Nenhuma opção {direction.value} encontrada nos critérios do horizonte {horizon}.")
            
            # Tentativa com parâmetros mais flexíveis
            st.caption("🔄 Tentando com parâmetros mais flexíveis...")
            
            # Parâmetros mais flexíveis para fallback
            fallback_params = {
                "min_volume": 5,  # Volume ainda menor
                "min_days": 7,    # Prazo mínimo menor
                "max_days": 60,   # Prazo máximo maior
            }
            
            # Ajusta parâmetros baseado na direção
            if direction == Direction.CALL:
                fallback_params.update({
                    "call_min_distance_pct": 3.0,  # Distância menor
                    "call_max_delta": 0.60,        # Delta maior
                    "put_max_distance_pct": 0,
                    "put_min_delta": 0,
                })
            else:
                fallback_params.update({
                    "call_min_distance_pct": 0,
                    "call_max_delta": 0,
                    "put_max_distance_pct": -3.0,  # Distância menor
                    "put_min_delta": -0.60,        # Delta maior
                })
            
            # Busca com parâmetros flexíveis
            df_fallback = find_synthetic_dividend_options(
                ticker,
                client=client,
                **fallback_params
            )
            
            # Filtra pelo tipo da direção
            if direction == Direction.CALL:
                df_fallback = df_fallback[df_fallback["Estratégia"] == "CALL"]
            else:
                df_fallback = df_fallback[df_fallback["Estratégia"] == "PUT"]
            
            if df_fallback.empty:
                st.info("💡 Nenhuma opção encontrada mesmo com parâmetros flexíveis.")
                st.caption("Sugestões:")
                st.caption("• Tente outro ticker com mais liquidez (ex: PETR4, VALE3)")
                st.caption("• Use outro horizonte temporal")
                st.caption("• Verifique se há opções disponíveis no mercado")
                return
            else:
                st.success(f"✅ Encontradas {len(df_fallback)} opções com parâmetros flexíveis!")
                df = df_fallback
        
        # Mostra tabela simplificada
        display_df = df[[
            "Opção", "Strike", "Validade", "Prêmio (R$)", 
            "Retorno (%)", "Retorno a.a. (%)", "Prob. Exercício (%)"
        ]].copy()
        
        # Ordena por melhor retorno anualizado
        display_df = display_df.sort_values("Retorno a.a. (%)", ascending=False)
        
        # Configuração das colunas
        column_config = {
            "Strike": st.column_config.NumberColumn("Strike", format="R$ %.2f"),
            "Prêmio (R$)": st.column_config.NumberColumn("Prêmio (R$)", format="R$ %.2f"),
            "Retorno (%)": st.column_config.NumberColumn("Retorno (%)", format="%.1f%%"),
            "Retorno a.a. (%)": st.column_config.NumberColumn("Retorno a.a. (%)", format="%.1f%%"),
            "Prob. Exercício (%)": st.column_config.NumberColumn("Prob. Exercício (%)", format="%.0f%%"),
        }
        
        st.dataframe(display_df, use_container_width=True, column_config=column_config, hide_index=True)
        
        # Resumo
        st.info(f"""
        **📊 Resumo:**
        - **Direção:** {direction.value}
        - **Horizonte:** {horizon}
        - **Opções encontradas:** {len(df)}
        """)
        
    except Exception as e:
        st.error(f"Erro ao buscar opções: {e}")


if __name__ == "__main__":
    render_direction_radar_page()