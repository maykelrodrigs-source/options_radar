"""
Página de Análise de Fundamentos - Valuation de Ações Brasileiras
"""

import streamlit as st
import pandas as pd
import json
from typing import List

from src.features.fundamentals.valuation import (
    analyze_fundamentals,
    analyze_multiple_tickers,
    get_real_fundamental_data,
    get_sample_fundamental_data,
    FundamentalData,
    ValuationResult
)
from src.core.data.oplab_client import OpLabClient


def render_fundamentals_page():
    """Renderiza a página de análise de fundamentos."""
    
    st.markdown("### 📊 Análise de Fundamentos")
    st.markdown("Cálculo de preço justo usando múltiplos métodos de valuation com **dados reais** via OpLab API")
    
    # Configurações
    with st.expander("⚙️ Configurações", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            yield_min = st.number_input(
                "Yield mínimo desejado (%)",
                min_value=1.0,
                max_value=20.0,
                value=6.0,
                step=0.5,
                help="Dividend yield mínimo para cálculo de preço teto"
            )
            
        with col2:
            taxa_desconto = st.number_input(
                "Taxa de desconto Bazin (%)",
                min_value=1.0,
                max_value=20.0,
                value=6.0,
                step=0.5,
                help="Taxa de desconto para fórmula de Bazin"
            )
            
        with col3:
            st.markdown("**P/Ls Alvo:**")
            pl10 = st.number_input("P/L 10", value=10, min_value=1, max_value=50, key="pl10")
            pl12 = st.number_input("P/L 12", value=12, min_value=1, max_value=50, key="pl12")
            pl15 = st.number_input("P/L 15", value=15, min_value=1, max_value=50, key="pl15")
    
    # Modo de análise
    modo = st.radio(
        "Modo de análise:",
        ["Análise Individual", "Análise Múltipla"],
        horizontal=True
    )
    
    if modo == "Análise Individual":
        render_individual_analysis(yield_min, taxa_desconto, [pl10, pl12, pl15])
    else:
        render_multiple_analysis(yield_min, taxa_desconto, [pl10, pl12, pl15])


def render_individual_analysis(yield_min: float, taxa_desconto: float, pl_targets: List[float]):
    """Renderiza análise individual de um ticker."""
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Dados de Entrada")
        
        # Ticker
        ticker = st.text_input(
            "Ticker",
            value="BBAS3",
            help="Código da ação (ex: BBAS3, PETR4, VALE3)"
        ).strip().upper()
        
        # Verificar se o ticker mudou e buscar dados automaticamente
        if 'last_ticker' not in st.session_state or st.session_state.last_ticker != ticker:
            st.session_state.last_ticker = ticker
            # Buscar dados reais para preencher os campos
            try:
                client = OpLabClient()
                fundamental_data = get_real_fundamental_data(ticker, client)
                st.session_state.fundamental_data = fundamental_data
                st.success(f"✅ Dados carregados para {ticker}")
            except Exception as e:
                st.session_state.fundamental_data = None
                st.warning(f"⚠️ Erro ao buscar dados para {ticker}: {e}")
        
        # Botão para atualizar dados manualmente
        col_refresh1, col_refresh2 = st.columns([1, 1])
        with col_refresh1:
            if st.button("🔄 Atualizar Dados", help="Força a atualização dos dados do ticker"):
                try:
                    client = OpLabClient()
                    fundamental_data = get_real_fundamental_data(ticker, client)
                    st.session_state.fundamental_data = fundamental_data
                    st.session_state.last_ticker = ticker  # Atualizar para evitar recarregamento
                    st.success(f"✅ Dados atualizados para {ticker}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erro ao atualizar dados: {e}")
        
        with col_refresh2:
            if st.button("🗑️ Limpar Cache", help="Limpa os dados em cache"):
                if 'fundamental_data' in st.session_state:
                    del st.session_state.fundamental_data
                if 'last_ticker' in st.session_state:
                    del st.session_state.last_ticker
                st.success("✅ Cache limpo")
                st.rerun()
        
        # Dados fundamentais
        with st.form("fundamental_data_form"):
            st.markdown("**Dados Fundamentais:**")
            
            # Usar dados da sessão se disponíveis
            if st.session_state.fundamental_data:
                data = st.session_state.fundamental_data
                preco_default = data.preco_atual if data.preco_atual > 0 else 22.16
                lpa_default = data.lpa if data.lpa > 0 else 2.85
                vpa_default = data.vpa if data.vpa > 0 else 18.50
                dps_default = data.dps if data.dps > 0 else 1.20
                crescimento_default = data.crescimento_esperado if data.crescimento_esperado > 0 else 8.0
            else:
                preco_default = 22.16
                lpa_default = 2.85
                vpa_default = 18.50
                dps_default = 1.20
                crescimento_default = 8.0
            
            preco_atual = st.number_input(
                "Preço Atual (R$)",
                min_value=0.01,
                value=preco_default,
                step=0.01,
                format="%.2f"
            )
            
            lpa = st.number_input(
                "LPA - Lucro por Ação (R$)",
                min_value=0.0,
                value=lpa_default,
                step=0.01,
                format="%.2f"
            )
            
            vpa = st.number_input(
                "VPA - Valor Patrimonial por Ação (R$)",
                min_value=0.0,
                value=vpa_default,
                step=0.01,
                format="%.2f"
            )
            
            dps = st.number_input(
                "DPS - Dividendos por Ação (R$)",
                min_value=0.0,
                value=dps_default,
                step=0.01,
                format="%.2f"
            )
            
            crescimento_esperado = st.number_input(
                "Crescimento Esperado (%)",
                min_value=0.0,
                max_value=100.0,
                value=crescimento_default,
                step=0.5
            )
            
            submitted = st.form_submit_button("🔍 Analisar", type="primary")
    
    with col2:
        if submitted:
            # Usar dados da sessão se disponíveis, senão buscar
            if st.session_state.fundamental_data:
                fundamental_data = st.session_state.fundamental_data
                st.info(f"📡 Usando dados já carregados para {ticker}")
            else:
                # Buscar dados reais se não estiverem na sessão
                try:
                    client = OpLabClient()
                    fundamental_data = get_real_fundamental_data(ticker, client)
                    st.success(f"✅ Dados carregados para {ticker}")
                except Exception as e:
                    st.error(f"❌ Erro ao buscar dados reais: {e}")
                    st.info("Verifique se o ticker existe e se as configurações OPLAB_API_* estão corretas.")
                    return
            
            # Atualizar com dados do formulário se fornecidos
            fundamental_data.preco_atual = preco_atual
            if lpa > 0:
                fundamental_data.lpa = lpa
            if vpa > 0:
                fundamental_data.vpa = vpa
            if dps > 0:
                fundamental_data.dps = dps
            if crescimento_esperado > 0:
                fundamental_data.crescimento_esperado = crescimento_esperado
            
            # Recalcular métricas
            if fundamental_data.lpa > 0:
                fundamental_data.pl = fundamental_data.preco_atual / fundamental_data.lpa
            if fundamental_data.vpa > 0:
                fundamental_data.pvp = fundamental_data.preco_atual / fundamental_data.vpa
            if fundamental_data.dps > 0:
                fundamental_data.dividend_yield = (fundamental_data.dps / fundamental_data.preco_atual) * 100
            if fundamental_data.lpa > 0 and fundamental_data.dps > 0:
                fundamental_data.payout = (fundamental_data.dps / fundamental_data.lpa) * 100
            if fundamental_data.vpa > 0 and fundamental_data.lpa > 0:
                fundamental_data.roe = (fundamental_data.lpa / fundamental_data.vpa) * 100
            if fundamental_data.crescimento_esperado > 0:
                fundamental_data.peg_ratio = fundamental_data.pl / fundamental_data.crescimento_esperado
            
            st.success(f"✅ Preço atual de {ticker}: R$ {fundamental_data.preco_atual:.2f}")
            
            # Análise
            result = analyze_fundamentals(
                ticker, fundamental_data, yield_min, pl_targets, [1.0, 1.5], taxa_desconto
            )
            
            render_valuation_results(result)


def render_multiple_analysis(yield_min: float, taxa_desconto: float, pl_targets: List[float]):
    """Renderiza análise múltipla de vários tickers."""
    
    st.subheader("📋 Análise Múltipla")
    
    # Lista de tickers
    tickers_input = st.text_area(
        "Tickers para análise (um por linha):",
        value="BBAS3\nPETR4\nVALE3\nITUB4\nBBDC4",
        help="Digite um ticker por linha"
    )
    
    if st.button("🔍 Analisar Múltiplos", type="primary"):
        tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
        
        if not tickers:
            st.warning("Digite pelo menos um ticker.")
            return
        
        # Mostrar status dos dados
        st.info("📡 Buscando preços atuais via OpLab API...")
        
        # Análise múltipla
        try:
            client = OpLabClient()
            results = analyze_multiple_tickers(tickers, yield_min, pl_targets, [1.0, 1.5], taxa_desconto, client)
            
            if not results:
                st.warning("Nenhum resultado encontrado.")
                return
            
            # Tabela resumo
            render_summary_table(results)
            
            # Resultados detalhados
            st.subheader("📊 Resultados Detalhados")
            for result in results:
                with st.expander(f"{result.ticker} - Margem: {result.margem_seguranca['media']:.1f}%"):
                    render_valuation_results(result)
                    
        except Exception as e:
            st.error(f"Erro na análise: {e}")
            st.info("Verifique se os tickers existem e se as configurações OPLAB_API_* estão corretas.")


def render_valuation_results(result: ValuationResult):
    """Renderiza os resultados de valuation de forma detalhada."""
    
    # Cards de resumo
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Preço Atual",
            f"R$ {result.preco_atual:.2f}",
            delta=f"{result.margem_seguranca['media']:.1f}%",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Preço Justo Médio",
            f"R$ {result.media_precos_justos:.2f}",
            delta="Média dos métodos"
        )
    
    with col3:
        status = "🟢 Desconto" if result.desconto else "🔴 Caro"
        st.metric("Status", status)
    
    with col4:
        st.metric("PEG Ratio", f"{result.peg_ratio:.2f}")
    
    # Tabela de preços justos
    st.subheader("💰 Preços Justos por Método")
    
    precos_data = {
        "Método": [
            "Graham",
            "Dividend Yield",
            "P/L 10",
            "P/L 12", 
            "P/L 15",
            "P/VPA 1.0",
            "P/VPA 1.5",
            "Bazin",
            "**Média**"
        ],
        "Preço Justo (R$)": [
            f"{result.preco_graham:.2f}",
            f"{result.preco_dividendos:.2f}",
            f"{result.preco_pl10:.2f}",
            f"{result.preco_pl12:.2f}",
            f"{result.preco_pl15:.2f}",
            f"{result.preco_pvp1:.2f}",
            f"{result.preco_pvp1_5:.2f}",
            f"{result.preco_bazin:.2f}",
            f"**{result.media_precos_justos:.2f}**"
        ],
        "Margem Segurança (%)": [
            f"{result.margem_seguranca['graham']:.1f}%",
            f"{result.margem_seguranca['dividendos']:.1f}%",
            f"{result.margem_seguranca['pl10']:.1f}%",
            f"{result.margem_seguranca['pl12']:.1f}%",
            f"{result.margem_seguranca['pl15']:.1f}%",
            f"{result.margem_seguranca['pvp1']:.1f}%",
            f"{result.margem_seguranca['pvp1_5']:.1f}%",
            f"{result.margem_seguranca['bazin']:.1f}%",
            f"**{result.margem_seguranca['media']:.1f}%**"
        ]
    }
    
    df_precos = pd.DataFrame(precos_data)
    st.dataframe(df_precos, use_container_width=True, hide_index=True)
    
    # JSON de saída
    st.subheader("📄 JSON de Saída")
    
    json_output = {
        "ticker": result.ticker,
        "preco_atual": result.preco_atual,
        "preco_graham": result.preco_graham,
        "preco_dividendos": result.preco_dividendos,
        "preco_pl10": result.preco_pl10,
        "preco_pl12": result.preco_pl12,
        "preco_pl15": result.preco_pl15,
        "preco_pvp1": result.preco_pvp1,
        "preco_pvp1_5": result.preco_pvp1_5,
        "preco_bazin": result.preco_bazin,
        "peg_ratio": result.peg_ratio,
        "media_precos_justos": result.media_precos_justos,
        "desconto": result.desconto,
        "caro": result.caro,
        "margem_seguranca": result.margem_seguranca
    }
    
    st.code(json.dumps(json_output, indent=2, ensure_ascii=False), language="json")


def render_summary_table(results: List[ValuationResult]):
    """Renderiza tabela resumo para análise múltipla."""
    
    st.subheader("📈 Resumo Comparativo")
    
    summary_data = []
    for result in results:
        summary_data.append({
            "Ticker": result.ticker,
            "Preço Atual": f"R$ {result.preco_atual:.2f}",
            "Preço Justo": f"R$ {result.media_precos_justos:.2f}",
            "Margem Segurança": f"{result.margem_seguranca['media']:.1f}%",
            "Status": "🟢 Desconto" if result.desconto else "🔴 Caro",
            "PEG": f"{result.peg_ratio:.2f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Configuração de colunas
    column_config = {
        "Margem Segurança": st.column_config.NumberColumn(
            "Margem Segurança",
            help="Margem de segurança em relação ao preço justo médio",
            format="%.1f%%"
        ),
        "PEG": st.column_config.NumberColumn(
            "PEG",
            help="PEG ratio (P/L dividido pelo crescimento)",
            format="%.2f"
        )
    }
    
    st.dataframe(df_summary, use_container_width=True, hide_index=True, column_config=column_config)
