import os
import sys
import streamlit as st
import pandas as pd

# Garante que o pacote 'src' é importável quando rodado diretamente
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.data.oplab_client import OpLabClient
from src.features.income.synthetic_dividends import find_synthetic_dividend_options
from src.app.pages.professional_radar import render_professional_radar_page
from src.features.income.income_opportunities import render_income_opportunities_page
from src.app.pages.run_backtest import render_backtest_page


st.set_page_config(page_title="Options Radar", layout="wide")
st.title("🎯 Options Radar")

# Cria abas
tab1, tab2, tab3, tab4 = st.tabs(["💰 Dividendos Sintéticos", "🎯 Radar Profissional", "💸 Oportunidades de Renda", "🔬 Backtest"])

with tab1:
    st.markdown("### Estratégia de Dividendos Sintéticos")
    st.markdown("Busca opções para estratégia de dividendos sintéticos")

    def render_params_form():
        with st.form("params_form"):
            st.subheader("Parâmetros de busca")
            
            # Primeira linha - ticker e tipo
            ticker_col, type_col = st.columns([2, 1])
            with ticker_col:
                ticker = st.text_input("Ticker (ex.: PETR4)", value="PETR4", key="dividends_ticker").strip().upper()
            with type_col:
                option_types = st.selectbox(
                    "Tipo de opções",
                    options=["Ambas (CALL + PUT)", "Apenas CALL", "Apenas PUT"],
                    index=0,
                    help="Escolha quais estratégias incluir na busca",
                    key="dividends_option_types"
                )

            # Segunda linha de parâmetros
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                min_volume = st.number_input("Contratos ativos (min)", min_value=0, value=10, step=5, help="Quantidade mínima de contratos ativos para filtro de liquidez", key="dividends_min_volume")
            with col2:
                min_days = st.number_input("Prazo mínimo (dias)", min_value=5, max_value=365, value=15, step=1, help="Mínimo 5 dias para evitar opções muito próximas do vencimento", key="dividends_min_days")
            with col3:
                max_days = st.number_input("Prazo máximo (dias)", min_value=1, max_value=365, value=45, step=1, key="dividends_max_days")
            with col4:
                max_exercise_prob = st.number_input("Prob. máx. exercício (%)", min_value=1.0, max_value=50.0, value=20.0, step=1.0, help="Probabilidade máxima de exercício da opção", key="dividends_exercise_prob")

            submitted = st.form_submit_button("Buscar sugestões")

        params = {
            "ticker": ticker,
            "min_volume": int(min_volume),
            "min_days": int(min_days),
            "max_days": int(max_days),
            "max_exercise_prob": float(max_exercise_prob),
            "option_types": option_types,
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
                    max_exercise_prob=params["max_exercise_prob"],
                    option_types=params["option_types"],
                )
                render_results(params["ticker"], df)
        except Exception as e:
            st.error(f"Erro: {e}")

with tab2:
    render_professional_radar_page()

with tab3:
    render_income_opportunities_page()

with tab4:
    render_backtest_page()