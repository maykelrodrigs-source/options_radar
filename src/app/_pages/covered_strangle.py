"""
Página: Covered Strangle
"""

import streamlit as st
import pandas as pd
import json
from typing import List, Dict, Any

from src.core.data.oplab_client import OpLabClient
from src.features.income.covered_strangle import generate_covered_strangle


def render_covered_strangle_page():
    st.title("🧲 Covered Strangle")
    st.markdown("Estratégia combinando CALL coberta + PUT coberta em caixa")

    st.markdown("**⚙️ Configurações**")
    col1, col2, col3 = st.columns(3)
    with col1:
        min_prob = st.slider(
            "Prob. mínima de sucesso",
            min_value=0.55,
            max_value=0.85,
            value=0.65,
            step=0.01,
            help="Probabilidade mínima para cada ponta não ser exercida"
        )
    with col2:
        min_premium = st.number_input(
            "Prêmio mínimo (R$)",
            min_value=0.05,
            max_value=2.00,
            value=0.20,
            step=0.05,
            help="Prêmio mínimo por opção (CALL e PUT)"
        )
    with col3:
        dte_range = st.select_slider(
            "DTE alvo (dias)",
            options=list(range(14, 61)),
            value=(21, 35)
        )

    st.markdown("**📦 Carteira**")
    st.caption("Informe os ativos em carteira (quantidade) e caixa disponível opcional para PUT.")
    example = pd.DataFrame([
        {"ticker": "BBAS3", "quantidade": 200, "caixa": 5000.0},
        {"ticker": "PETR4", "quantidade": 100, "caixa": 3000.0},
    ])

    portfolio_df = st.data_editor(
        example,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "quantidade": st.column_config.NumberColumn("Qtd. ações", min_value=0, step=100),
            "caixa": st.column_config.NumberColumn("Caixa (R$)", min_value=0.0, step=100.0, format="R$ %.2f"),
        }
    )

    if st.button("🔍 Buscar", type="primary"):
        try:
            client = OpLabClient()
            portfolio_list: List[Dict[str, Any]] = portfolio_df.fillna(0).to_dict(orient="records")
            results = generate_covered_strangle(
                portfolio=portfolio_list,
                client=client,
                min_prob_success=min_prob,
                min_premium=min_premium,
                dte_min=int(dte_range[0]),
                dte_max=int(dte_range[1]),
                debug=True,  # Ativar debug para identificar problema
            )

            if not results:
                st.warning("Nenhuma combinação encontrada com os filtros atuais.")
                return

            st.success(f"✅ {len(results)} recomendações geradas")

            # Tabela resumida amigável
            df = pd.DataFrame(results)
            friendly = pd.DataFrame({
                "Ativo": df["ticker"],
                "Preço Atual (R$)": df["preco_atual"],
                "CALL (símbolo)": df.get("call_symbol", ""),
                "CALL Strike": df["call_strike"],
                "CALL Preço (Bid)": df["premio_call"],
                "PUT (símbolo)": df.get("put_symbol", ""),
                "PUT Strike": df["put_strike"],
                "PUT Preço (Bid)": df["premio_put"],
                "Prêmio Total (R$)": df["premio_total"],
                "Prob. Sucesso CALL": (df["probabilidade_call"] * 100).round(1).astype(str) + "%",
                "Prob. Sucesso PUT": (df["probabilidade_put"] * 100).round(1).astype(str) + "%",
                "Range de Sucesso": df["range_sucesso"],
                "DTE (dias)": df["dte"],
                "Contratos Sugeridos": df["contratos_sugeridos"],
                "Caixa Utilizado (R$)": df["caixa_utilizado"],
                "Lucro Estimado (R$)": df["lucro_estimado"],
                "Retorno sobre Caixa (%)": df["retorno_caixa_pct"],
                "Retorno Efetivo (%)": df["retorno_efetivo_pct"],
                "Qualificada": df["qualificada"],
            })

            column_config = {
                "Preço Atual (R$)": st.column_config.NumberColumn("Preço Atual (R$)", format="R$ %.2f"),
                "CALL Strike": st.column_config.NumberColumn("CALL Strike", format="R$ %.2f"),
                "CALL Preço (Bid)": st.column_config.NumberColumn("CALL Preço (Bid)", format="R$ %.2f"),
                "PUT Strike": st.column_config.NumberColumn("PUT Strike", format="R$ %.2f"),
                "PUT Preço (Bid)": st.column_config.NumberColumn("PUT Preço (Bid)", format="R$ %.2f"),
                "Prêmio Total (R$)": st.column_config.NumberColumn("Prêmio Total (R$)", format="R$ %.2f"),
                "Retorno Efetivo (%)": st.column_config.NumberColumn("Retorno Efetivo (%)", format="%.2f%%"),
                "Retorno sobre Caixa (%)": st.column_config.NumberColumn("Retorno sobre Caixa (%)", format="%.2f%%"),
                "Caixa Utilizado (R$)": st.column_config.NumberColumn("Caixa Utilizado (R$)", format="R$ %.2f"),
                "Lucro Estimado (R$)": st.column_config.NumberColumn("Lucro Estimado (R$)", format="R$ %.2f"),
                "DTE (dias)": st.column_config.NumberColumn("DTE (dias)", format="%d"),
                "Contratos Sugeridos": st.column_config.NumberColumn("Contratos Sugeridos", format="%d"),
            }

            st.dataframe(friendly, use_container_width=True, hide_index=True, column_config=column_config)

            with st.expander("💾 JSON por ativo"):
                st.code(json.dumps(results, indent=2, ensure_ascii=False), language="json")
        except Exception as e:
            st.error(f"Erro: {e}")


if __name__ == "__main__":
    render_covered_strangle_page()
