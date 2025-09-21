"""
Página do Radar de Direção - Análise técnica para decisão CALL/PUT.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

from oplab_client import OpLabClient
from synthetic_dividends import find_synthetic_dividend_options
from technical_analysis import (
    TechnicalAnalyzer, Direction, RiskProfile, 
    get_option_parameters_by_direction, generate_direction_justification
)
from historical_data import historical_provider


def render_direction_radar_page():
    """Renderiza a página do Radar de Direção."""
    st.title("🎯 Radar de Direção")
    st.markdown("Análise técnica para decisão CALL/PUT baseada em indicadores")
    
    # Formulário de parâmetros
    submitted, params = render_direction_form()
    
    if submitted:
        try:
            # Busca dados
            client = OpLabClient()
            current_price = client.get_underlying_price(params["ticker"])
            
            # Dados históricos (simulados)
            historical_data = historical_provider.get_historical_data(
                params["ticker"], current_price
            )
            
            # Análise técnica
            analyzer = TechnicalAnalyzer()
            signal = analyzer.calculate_technical_indicators(historical_data)
            
            # Renderiza resultados
            render_technical_analysis(signal, params)
            
            # Se há direção definida, busca opções
            if signal.direction != Direction.NEUTRAL:
                render_direction_options(params, signal, client, current_price)
            else:
                st.info("🔍 Sem sinal forte. Recomendo apenas Dividendos Sintéticos.")
                
        except Exception as e:
            st.error(f"Erro: {e}")


def render_direction_form() -> tuple[bool, dict]:
    """Renderiza formulário de parâmetros para Radar de Direção."""
    with st.form("direction_form"):
        st.subheader("📊 Parâmetros de Análise")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ticker = st.text_input("Ticker", value="PETR4", help="Ex: PETR4, VALE3, ITUB4")
        
        with col2:
            risk_profile = st.selectbox(
                "Perfil de Risco",
                options=[RiskProfile.CONSERVATIVE, RiskProfile.MODERATE, RiskProfile.AGGRESSIVE],
                format_func=lambda x: x.value.title(),
                help="Define distância do strike e prazo das opções"
            )
        
        with col3:
            st.markdown("**Macro (Simulado)**")
            st.caption("SELIC: 10.75% | USD/BRL: 5.20 | IPCA: 4.62%")
        
        submitted = st.form_submit_button("🔍 Analisar Direção")
    
    params = {
        "ticker": ticker.strip().upper(),
        "risk_profile": risk_profile
    }
    
    return submitted, params


def render_technical_analysis(signal, params: dict):
    """Renderiza análise técnica e sinal de direção."""
    st.subheader("📈 Análise Técnica")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Preço Atual", f"R$ {signal.current_price:.2f}")
    
    with col2:
        sma_200_diff = ((signal.current_price - signal.sma_200) / signal.sma_200) * 100
        st.metric("SMA200", f"R$ {signal.sma_200:.2f}", f"{sma_200_diff:+.1f}%")
    
    with col3:
        st.metric("RSI(14)", f"{signal.rsi_14:.1f}", 
                 "🔴" if signal.rsi_14 > 70 else "🟡" if signal.rsi_14 > 50 else "🟢")
    
    with col4:
        macd_status = "🟢" if signal.macd_line > signal.macd_signal else "🔴"
        st.metric("MACD", f"{signal.macd_line:.3f}", macd_status)
    
    # Sinal de direção
    st.subheader("🎯 Sinal de Direção")
    
    if signal.direction == Direction.CALL:
        st.success(f"📈 **TENDÊNCIA DE ALTA** (Confiança: {signal.confidence:.0%})")
        st.info("💡 **Recomendação:** CALLs OTM para aproveitar movimento de alta")
    elif signal.direction == Direction.PUT:
        st.error(f"📉 **TENDÊNCIA DE BAIXA** (Confiança: {signal.confidence:.0%})")
        st.info("💡 **Recomendação:** PUTs OTM para proteção contra queda")
    else:
        st.warning("⚖️ **SINAL NEUTRO** (Confiança: {signal.confidence:.0%})")
        st.info("💡 **Recomendação:** Apenas Dividendos Sintéticos")
    
    # Justificativa
    with st.expander("🔍 Detalhes da Análise"):
        st.write(signal.reasoning)
        
        # Tabela de indicadores
        indicators_df = pd.DataFrame({
            "Indicador": ["Preço", "SMA50", "SMA200", "RSI(14)", "MACD", "MACD Signal", "Volume Ratio"],
            "Valor": [
                f"R$ {signal.current_price:.2f}",
                f"R$ {signal.sma_50:.2f}",
                f"R$ {signal.sma_200:.2f}",
                f"{signal.rsi_14:.1f}",
                f"{signal.macd_line:.3f}",
                f"{signal.macd_signal:.3f}",
                f"{signal.volume_ratio_20d:.1f}x"
            ],
            "Interpretação": [
                "Preço atual",
                "Média 50 dias",
                "Média 200 dias (tendência)",
                "Momentum (30=sobrevendido, 70=sobrecomprado)",
                "Convergência/Divergência",
                "Sinal MACD",
                "Volume vs média 20d"
            ]
        })
        st.dataframe(indicators_df, use_container_width=True, hide_index=True)


def render_direction_options(params: dict, signal, client: OpLabClient, current_price: float):
    """Renderiza opções recomendadas baseadas na direção."""
    st.subheader("🎯 Opções Recomendadas")
    
    # Obtém parâmetros de busca baseados na direção
    option_params = get_option_parameters_by_direction(
        signal.direction, current_price, params["risk_profile"]
    )
    
    if not option_params:
        st.warning("Não foi possível gerar parâmetros para busca de opções.")
        return
    
    # Busca opções com critérios específicos
    df = find_synthetic_dividend_options(
        params["ticker"],
        client=client,
        min_volume=option_params["min_volume"],
        min_days=option_params["min_days"],
        max_days=option_params["max_days"],
        call_min_distance_pct=option_params["call_min_distance_pct"],
        call_max_delta=option_params["call_max_delta"],
        put_max_distance_pct=option_params["put_max_distance_pct"],
        put_min_delta=option_params["put_min_delta"],
    )
    
    # Se não encontrou opções, tenta critérios mais flexíveis
    if df.empty:
        st.warning("⚠️ Nenhuma opção encontrada nos critérios específicos. Buscando com critérios mais flexíveis...")
        
        # Critérios mais flexíveis
        fallback_params = {
            "min_volume": 5,  # Volume muito baixo
            "min_days": 7,    # Prazo mínimo
            "max_days": 180,  # Prazo máximo amplo
        }
        
        if signal.direction == Direction.CALL:
            fallback_params.update({
                "call_min_distance_pct": 5.0,  # 5% mínimo
                "call_max_delta": 0.60,        # Delta mais alto
                "put_max_distance_pct": 0,
                "put_min_delta": 0,
            })
        else:  # PUT
            fallback_params.update({
                "call_min_distance_pct": 0,
                "call_max_delta": 0,
                "put_max_distance_pct": 5.0,   # 5% mínimo
                "put_min_delta": 0.10,         # Delta mais baixo
            })
        
        df = find_synthetic_dividend_options(
            params["ticker"],
            client=client,
            **fallback_params
        )
        
        if df.empty:
            st.info("Nenhuma opção encontrada mesmo com critérios flexíveis.")
            return
        else:
            st.success("✅ Encontradas opções com critérios flexíveis!")
    
    # Filtra apenas o tipo de opção da direção
    if signal.direction == Direction.CALL:
        df = df[df["Estratégia"] == "CALL"]
    else:
        df = df[df["Estratégia"] == "PUT"]
    
    if df.empty:
        st.info(f"Nenhuma opção {signal.direction.value} encontrada nos critérios.")
        return
    
    # Adiciona justificativas específicas da direção
    df["Justificativa Direção"] = df.apply(
        lambda row: generate_direction_justification(
            signal.direction, row["Estratégia"], row["Strike"], 
            row["Prêmio (R$)"], current_price, 
            (pd.to_datetime(row["Validade"]) - pd.Timestamp.now()).days,
            signal.confidence
        ), axis=1
    )
    
    # Seleciona colunas para exibição
    display_df = df[[
        "Opção", "Estratégia", "Strike", "Validade", "Prêmio (R$)", 
        "Retorno (%)", "Prob. Exercício (%)", "Justificativa Direção"
    ]].copy()
    
    # Configuração das colunas
    column_config = {
        "Opção": st.column_config.TextColumn("Opção", help="Código da opção"),
        "Estratégia": st.column_config.TextColumn("Tipo", help="CALL ou PUT"),
        "Strike": st.column_config.NumberColumn("Strike", format="R$ %.2f", help="Preço de exercício"),
        "Validade": st.column_config.DateColumn("Vencimento", help="Data de vencimento"),
        "Prêmio (R$)": st.column_config.NumberColumn("Prêmio (R$)", format="R$ %.2f", help="Prêmio da opção"),
        "Retorno (%)": st.column_config.NumberColumn("Retorno (%)", format="%.1f%%", help="Retorno no período"),
        "Prob. Exercício (%)": st.column_config.NumberColumn("Prob. Exercício (%)", format="%.0f%%", help="Probabilidade de exercício"),
        "Justificativa Direção": st.column_config.TextColumn("Justificativa", help="Justificativa baseada na direção técnica")
    }
    
    st.dataframe(display_df, use_container_width=True, column_config=column_config, hide_index=True)
    
    # Resumo da recomendação
    st.info(f"""
    **📊 Resumo da Recomendação:**
    - **Direção:** {signal.direction.value}
    - **Perfil:** {params['risk_profile'].value.title()}
    - **Opções encontradas:** {len(df)}
    - **Confiança técnica:** {signal.confidence:.0%}
    - **Critérios utilizados:** {'Flexíveis' if 'fallback' in locals() else 'Específicos do perfil'}
    """)


if __name__ == "__main__":
    render_direction_radar_page()
