import os
import streamlit as st
import pandas as pd

from oplab_client import OpLabClient
from synthetic_dividends import find_synthetic_dividend_options
from direction_radar import render_direction_radar_page


st.set_page_config(page_title="Options Radar", layout="wide")
st.title("🎯 Options Radar")

# Cria abas
tab1, tab2 = st.tabs(["💰 Dividendos Sintéticos", "📈 Radar de Direção"])

with tab1:
    st.markdown("### Estratégia de Dividendos Sintéticos")
    st.markdown("Busca opções para estratégia de dividendos sintéticos")

    def render_params_form():
        with st.form("params_form"):
            st.subheader("Parâmetros de busca")
            ticker = st.text_input("Ticker (ex.: PETR4)", value="PETR4").strip().upper()

            col1, col2, col3 = st.columns(3)
            with col1:
                min_volume = st.number_input("Contratos ativos (min)", min_value=0, value=10, step=5, help="Quantidade mínima de contratos ativos para filtro de liquidez")
            with col2:
                min_days = st.number_input("Prazo mínimo (dias)", min_value=7, max_value=365, value=15, step=1, help="Mínimo 7 dias para evitar opções muito próximas do vencimento")
            with col3:
                max_days = st.number_input("Prazo máximo (dias)", min_value=1, max_value=365, value=45, step=1)
            

            st.markdown("CALL coberta")
            c1, c2 = st.columns(2)
            with c1:
                call_min_distance = st.number_input("CALL: distância mínima (%)", min_value=0.0, value=15.0, step=0.5)
            with c2:
                call_max_delta = st.number_input("CALL: delta máximo", min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")

            st.markdown("PUT coberta")
            p1, p2 = st.columns(2)
            with p1:
                put_max_distance = st.number_input("PUT: distância máxima (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5, help="Distância máxima abaixo do preço atual")
            with p2:
                put_min_delta = st.number_input("PUT: delta mínimo", min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f", help="Delta mínimo em valor absoluto")

            submitted = st.form_submit_button("Buscar sugestões")

        params = {
            "ticker": ticker,
            "min_volume": int(min_volume),
            "min_days": int(min_days),
            "max_days": int(max_days),
            "call_min_distance_pct": float(call_min_distance),
            "call_max_delta": float(call_max_delta),
            "put_max_distance_pct": -float(put_max_distance),  # Converter para negativo
            "put_min_delta": -float(put_min_delta),  # Converter para negativo
        }
        return submitted, params


    def render_results(ticker: str, df: pd.DataFrame) -> None:
        if df.empty:
            st.info("Nenhuma opção encontrada nos critérios atuais.")
        else:
            st.subheader(f"Sugestões para {ticker}")
            
            # Configuração de colunas com tooltips
            column_config = {
                "Prob. Exercício (%)": st.column_config.NumberColumn(
                    "Prob. Exercício (%)",
                    help="Probabilidade de exercício: <5% baixo risco, 5-15% moderado, >15% alto risco",
                    format="%.0f%%"
                ),
                "Retorno (%)": st.column_config.NumberColumn(
                    "Retorno (%)",
                    help="Retorno no período. CALL: sobre preço da ação | PUT: sobre strike",
                    format="%.2f%%"
                ),
                "Retorno a.a. (%)": st.column_config.NumberColumn(
                    "Retorno a.a. (%)",
                    help="Retorno anualizado para comparar com dividendos reais",
                    format="%.1f%%"
                ),
                "Prêmio (R$)": st.column_config.NumberColumn(
                    "Prêmio (R$)",
                    help="Valor recebido pela venda da opção",
                    format="R$ %.2f"
                ),
                "Strike": st.column_config.NumberColumn(
                    "Strike",
                    help="Preço de exercício da opção",
                    format="R$ %.2f"
                )
            }
            
            st.dataframe(df, use_container_width=True, column_config=column_config)


    # Lógica principal da aba Dividendos Sintéticos
    submitted, params = render_params_form()
    if not submitted:
        st.caption("Configure as variáveis OPLAB_API_* no ambiente antes de buscar.")
    else:
        try:
            client = OpLabClient()
            
            # Primeiro verifica se o ticker tem opções disponíveis
            option_chain = client.get_option_chain(params["ticker"])
            if option_chain.empty:
                st.warning(f"⚠️ {params['ticker']} não possui opções negociadas na B3.")
                st.info("💡 Tente tickers com maior liquidez como: PETR4, VALE3, ITUB4, BBDC4, BBAS3")
            else:
                df = find_synthetic_dividend_options(
                    params["ticker"],
                    client=client,
                    min_volume=params["min_volume"],
                    min_days=params["min_days"],
                    max_days=params["max_days"],
                    call_min_distance_pct=params["call_min_distance_pct"],
                    call_max_delta=params["call_max_delta"],
                    put_max_distance_pct=params["put_max_distance_pct"],
                    put_min_delta=params["put_min_delta"],
                )
                render_results(params["ticker"], df)
        except Exception as e:
            st.error(f"Erro: {e}")

with tab2:
    render_direction_radar_page()