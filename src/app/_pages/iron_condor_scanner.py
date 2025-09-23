"""
Página do Scanner de Iron Condor
Busca oportunidades de Iron Condor baseadas em probabilidade.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any
import json

from src.core.data.oplab_client import OpLabClient
from src.features.radar.iron_condor_scanner import scan_iron_condor_opportunities


def render_iron_condor_scanner_page():
    """Renderiza a página do Scanner de Iron Condor."""
    st.title("🦅 Scanner de Iron Condor")
    st.markdown("**Busca oportunidades de Iron Condor baseadas em probabilidade de sucesso**")
    
    # Explicação da estratégia
    with st.expander("📖 O que é Iron Condor?", expanded=False):
        st.markdown("""
        **Iron Condor** é uma estratégia neutra de opções que:
        
        🎯 **Objetivo:** Lucrar quando o ativo permanece dentro de um range de preços
        
        📊 **Estrutura:**
        - **Vende CALL OTM** + **Compra CALL mais alta** (trava de alta)
        - **Vende PUT OTM** + **Compra PUT mais baixa** (trava de baixa)
        
        💰 **Resultado:**
        - **Lucro máximo:** Prêmio líquido recebido (quando preço fica no range)
        - **Perda máxima:** Diferença entre strikes menos prêmio recebido
        
        ⚡ **Vantagens:**
        - Estratégia neutra (não precisa de direção)
        - Lucro limitado e previsível
        - Probabilidade de sucesso alta (75%+)
        
        ⚠️ **Riscos:**
        - Perda limitada mas potencialmente significativa
        - Requer gestão de risco adequada
        """)
    
    # Configurações
    st.markdown("**⚙️ Configurações**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_probability = st.slider(
            "Prob. Mínima",
            min_value=0.50,
            max_value=0.85,
            value=0.60,
            step=0.05,
            help="Probabilidade mínima de sucesso (recomendado: 60%)",
            key="iron_condor_prob"
        )
        
        min_premium_risk_ratio = st.slider(
            "Relação Prêmio/Risco",
            min_value=0.1,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Relação mínima prêmio/risco (recomendado: 0.2)",
            key="iron_condor_ratio"
        )
    
    with col2:
        max_days = st.selectbox(
            "Vencimento Máx",
            options=[30, 45, 60, 90],
            index=2,  # Default: 60 dias
            help="Vencimento máximo em dias (recomendado: 45-60 dias)",
            key="iron_condor_days"
        )
        
        min_iv_rank = st.slider(
            "IV Rank Mínimo",
            min_value=0,
            max_value=80,
            value=10,
            step=5,
            help="IV Rank mínimo (recomendado: 10+)",
            key="iron_condor_iv"
        )
    
    with col3:
        min_premium = st.number_input(
            "💰 Prêmio Mín (R$)", 
            0.01, 2.00, 0.10, 0.01,
            help="Prêmio líquido mínimo (R$ 0,10+ para qualidade)",
            key="iron_condor_premium"
        )
    
    # Botão para busca
    if st.button("🔍 Buscar", type="primary", key="iron_condor_search"):
        try:
            # Inicializa cliente OpLab
            with st.spinner("Conectando com OpLab..."):
                client = OpLabClient()
            
            # Busca oportunidades nos 10 tickers mais líquidos
            with st.spinner("🔍 Buscando oportunidades Iron Condor nos 10 tickers mais líquidos..."):
                opportunities = scan_iron_condor_opportunities(
                    client=client,
                    min_probability=min_probability,
                    max_days=max_days,
                    min_premium=min_premium,
                    min_premium_risk_ratio=min_premium_risk_ratio,
                    min_iv_rank=min_iv_rank,
                    apply_quality_filters=True
                )
            
            # Renderiza resultados
            if opportunities:
                render_iron_condor_results(opportunities, "Top 10 Tickers", min_probability, min_premium, min_premium_risk_ratio, min_iv_rank)
            else:
                st.warning(f"""
                ❌ **Nenhuma oportunidade encontrada nos 10 tickers mais líquidos**
                
                **Possíveis causas:**
                - Probabilidade mínima muito alta ({min_probability:.0%})
                - Vencimento muito curto (≤{max_days} dias)
                - Prêmio mínimo muito alto (R$ {min_premium:.2f})
                - Condições de mercado desfavoráveis
                
                **Sugestões:**
                - Reduza a probabilidade mínima (60-65%)
                - Aumente o vencimento máximo (75-90 dias)
                - Reduza o prêmio mínimo (R$ 0,10)
                - Tente novamente em outro horário
                """)
                
        except Exception as e:
            st.error(f"Erro na busca: {e}")
            st.info("Verifique se as configurações do OpLab estão corretas.")


def render_iron_condor_results(opportunities: List[Dict[str, Any]], ticker: str, min_probability: float, min_premium: float, min_premium_risk_ratio: float, min_iv_rank: float):
    """Renderiza os resultados das oportunidades Iron Condor."""
    
    st.success(f"✅ **{len(opportunities)} oportunidades encontradas para {ticker}**")
    
    # Resumo estatístico
    if opportunities:
        render_summary_metrics(opportunities)
    
    # Seção das 10 melhores oportunidades (acima dos filtros)
    render_top_opportunities_section(opportunities, min_probability, min_premium, min_premium_risk_ratio, min_iv_rank)
    
    # Tabela com seleção por checkbox
    st.subheader("📊 Oportunidades Encontradas - Selecione uma para Análise Detalhada")
    
    # Inicializa estado para seleção única
    if 'selected_opportunity_idx' not in st.session_state:
        st.session_state.selected_opportunity_idx = None
    
    # Converte para DataFrame para exibição
    df = pd.DataFrame(opportunities)
    
    # Seleciona colunas para exibição
    display_columns = [
        "underlying", "vencimento", "dte", "probabilidade_sucesso",
        "premio_liquido", "risco_maximo", "relacao_premio_risco",
        "EV", "iv_rank", "qualificada"
    ]
    
    display_df = df[display_columns].copy()
    
    # Formata colunas
    display_df["probabilidade_sucesso"] = display_df["probabilidade_sucesso"].apply(
        lambda x: f"{x:.1%}"
    )
    display_df["premio_liquido"] = display_df["premio_liquido"].apply(
        lambda x: f"R$ {x:.2f}"
    )
    display_df["risco_maximo"] = display_df["risco_maximo"].apply(
        lambda x: f"R$ {x:.2f}"
    )
    display_df["relacao_premio_risco"] = display_df["relacao_premio_risco"].apply(
        lambda x: f"{x:.2f}"
    )
    display_df["EV"] = display_df["EV"].apply(
        lambda x: f"R$ {x:.2f}"
    )
    display_df["iv_rank"] = display_df["iv_rank"].apply(
        lambda x: f"{x:.1f}%"
    )
    display_df["qualificada"] = display_df["qualificada"].apply(
        lambda x: "✅ Sim" if x else "❌ Não"
    )
    
    # Renomeia colunas
    display_df.columns = [
        "Ativo", "Vencimento", "DTE", "Prob. Sucesso",
        "Prêmio Líquido", "Risco Máximo", "Prêmio/Risco",
        "EV", "IV Rank", "Qualificada"
    ]
    
    # Configuração das colunas
    column_config = {
        "Ativo": st.column_config.TextColumn("Ativo", width="small"),
        "Vencimento": st.column_config.TextColumn("Vencimento", width="small"),
        "DTE": st.column_config.NumberColumn("DTE", format="%d", width="small"),
        "Prob. Sucesso": st.column_config.TextColumn("Prob. Sucesso", width="small"),
        "Prêmio Líquido": st.column_config.TextColumn("Prêmio Líquido", width="small"),
        "Risco Máximo": st.column_config.TextColumn("Risco Máximo", width="small"),
        "Prêmio/Risco": st.column_config.TextColumn("Prêmio/Risco", width="small"),
        "EV": st.column_config.TextColumn("EV", width="small"),
        "IV Rank": st.column_config.TextColumn("IV Rank", width="small"),
        "Qualificada": st.column_config.TextColumn("Qualificada", width="small"),
    }
    
    # Exibe a tabela
    st.dataframe(display_df, use_container_width=True, column_config=column_config, hide_index=True)
    
    # Análise automática da melhor oportunidade
    if opportunities:
        # Mostra automaticamente a melhor oportunidade (primeira da lista ordenada)
        best_opportunity = opportunities[0]  # Já ordenado por EV
        
        st.divider()
        st.subheader(f"📋 Análise Detalhada - {best_opportunity.get('underlying', 'N/A')} (Melhor Oportunidade)")
        
        # Renderiza análise detalhada da melhor oportunidade
        render_detailed_opportunity_analysis(best_opportunity)
        
        # Seção para ver outras oportunidades
        if len(opportunities) > 1:
            st.divider()
            st.subheader("🔍 Outras Oportunidades Encontradas")
            
            # Mostra as próximas 5 melhores oportunidades
            other_opportunities = opportunities[1:6]  # Pega as próximas 5
            
            for i, opp in enumerate(other_opportunities, 2):
                underlying = opp.get('underlying', 'N/A')
                vencimento = opp.get('vencimento', 'N/A')
                prob = opp.get('probabilidade_sucesso', 0)
                premio = opp.get('premio_liquido', 0)
                ev = opp.get('EV', 0)
                qualificada = opp.get('qualificada', False)
                
                status = "✅" if qualificada else "⚠️"
                
                with st.expander(f"{i}. {status} {underlying} | {vencimento} | {prob:.1%} | R$ {premio:.2f} | EV: R$ {ev:.2f}", expanded=False):
                    render_detailed_opportunity_analysis(opp)
    else:
        st.info("💡 **Nenhuma oportunidade encontrada para análise**")
    
    # Exportação JSON
    with st.expander("💾 Exportar Dados", expanded=False):
        json_data = json.dumps(opportunities, indent=2, default=str, ensure_ascii=False)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"iron_condor_{ticker}_{len(opportunities)}_oportunidades.json",
            mime="application/json"
        )
        
        st.code(json_data, language="json")


def render_top_opportunities_section(opportunities: List[Dict[str, Any]], min_probability: float, min_premium: float, min_premium_risk_ratio: float, min_iv_rank: float):
    """Renderiza as 10 melhores oportunidades que superam os filtros selecionados."""
    
    if not opportunities:
        return
    
    st.subheader("🏆 Top 10 Melhores Oportunidades (Acima dos Filtros)")
    
    # Cria diferentes categorias de "melhores" oportunidades
    categories = {
        "🔥 Maior Prêmio": {
            "key": "premio_liquido",
            "desc": f"Prêmios acima de R$ {min_premium:.2f}",
            "icon": "💰"
        },
        "📈 Maior Probabilidade": {
            "key": "probabilidade_sucesso", 
            "desc": f"Probabilidades acima de {min_probability:.0%}",
            "icon": "🎯"
        },
        "⚡ Melhor EV": {
            "key": "EV",
            "desc": "Maiores valores esperados",
            "icon": "💎"
        },
        "🛡️ Melhor Relação Prêmio/Risco": {
            "key": "relacao_premio_risco",
            "desc": f"Relações acima de {min_premium_risk_ratio:.1f}",
            "icon": "⚖️"
        },
        "📊 Maior IV Rank": {
            "key": "iv_rank",
            "desc": f"IV Rank acima de {min_iv_rank:.0f}%",
            "icon": "📈"
        }
    }
    
    # Para cada categoria, encontra as melhores oportunidades
    for category_name, category_info in categories.items():
        key = category_info["key"]
        desc = category_info["desc"]
        icon = category_info["icon"]
        
        # Filtra oportunidades que superam o filtro mínimo
        if key == "premio_liquido":
            filtered_opps = [opp for opp in opportunities if opp.get(key, 0) > min_premium]
        elif key == "probabilidade_sucesso":
            filtered_opps = [opp for opp in opportunities if opp.get(key, 0) > min_probability]
        elif key == "relacao_premio_risco":
            filtered_opps = [opp for opp in opportunities if opp.get(key, 0) > min_premium_risk_ratio]
        elif key == "iv_rank":
            filtered_opps = [opp for opp in opportunities if opp.get(key, 0) > min_iv_rank]
        else:  # EV - pega todas (pode ser negativo)
            filtered_opps = opportunities
        
        # Ordena por descrescente (maior valor primeiro)
        filtered_opps.sort(key=lambda x: x.get(key, 0), reverse=True)
        
        # Pega as top 3 de cada categoria
        top_opportunities = filtered_opps[:3]
        
        if top_opportunities:
            with st.expander(f"{icon} **{category_name}** - {desc}", expanded=False):
                for i, opp in enumerate(top_opportunities, 1):
                    underlying = opp.get('underlying', 'N/A')
                    vencimento = opp.get('vencimento', 'N/A')
                    prob = opp.get('probabilidade_sucesso', 0)
                    premio = opp.get('premio_liquido', 0)
                    ev = opp.get('EV', 0)
                    rr_ratio = opp.get('relacao_premio_risco', 0)
                    ivr = opp.get('iv_rank', 0)
                    qualificada = opp.get('qualificada', False)
                    
                    status = "✅" if qualificada else "⚠️"
                    value = opp.get(key, 0)
                    
                    # Formata o valor de acordo com a categoria
                    if key == "premio_liquido":
                        value_str = f"R$ {value:.2f}"
                    elif key == "probabilidade_sucesso":
                        value_str = f"{value:.1%}"
                    elif key == "EV":
                        value_str = f"R$ {value:.2f}"
                    elif key == "relacao_premio_risco":
                        value_str = f"{value:.2f}"
                    elif key == "iv_rank":
                        value_str = f"{value:.1f}%"
                    
                    st.markdown(f"""
                    **{i}. {status} {underlying}** | {vencimento}
                    - **{category_name}**: {value_str}
                    - **Probabilidade**: {prob:.1%} | **Prêmio**: R$ {premio:.2f}
                    - **EV**: R$ {ev:.2f} | **Prêmio/Risco**: {rr_ratio:.2f} | **IV Rank**: {ivr:.1f}%
                    """)
    


def render_detailed_opportunity_analysis(opportunity: Dict[str, Any]):
    """Renderiza análise detalhada de uma oportunidade selecionada."""
    
    # Status da oportunidade
    qualificada = opportunity.get('qualificada', False)
    status_color = "🟢" if qualificada else "🟡"
    status_text = "QUALIFICADA" if qualificada else "NÃO QUALIFICADA"
    
    st.markdown(f"**Status:** {status_color} {status_text}")
    
    # Métricas principais em colunas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Probabilidade de Sucesso",
            f"{opportunity.get('probabilidade_sucesso', 0):.1%}",
            help="Chance do preço ficar dentro do range"
        )
    
    with col2:
        st.metric(
            "Prêmio Líquido",
            f"R$ {opportunity.get('premio_liquido', 0):.2f}",
            help="Valor líquido recebido na operação"
        )
    
    with col3:
        st.metric(
            "Risco Máximo",
            f"R$ {opportunity.get('risco_maximo', 0):.2f}",
            help="Perda máxima possível"
        )
    
    with col4:
        ev = opportunity.get('EV', 0)
        ev_color = "normal" if ev >= 0 else "inverse"
        st.metric(
            "Valor Esperado (EV)",
            f"R$ {ev:.2f}",
            help="Valor esperado considerando probabilidades",
            delta=None
        )
    
    # Informações detalhadas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Informações da Operação:**")
        st.info(f"""
        **Ativo:** {opportunity.get('underlying', 'N/A')}  
        **Vencimento:** {opportunity.get('vencimento', 'N/A')}  
        **Dias até Vencimento:** {opportunity.get('dte', 'N/A')}  
        **IV Rank:** {opportunity.get('iv_rank', 0):.1f}%  
        **Prêmio/Risco:** {opportunity.get('relacao_premio_risco', 0):.2f}
        """)
    
    with col2:
        st.markdown("**🎯 Range de Sucesso:**")
        range_sucesso = opportunity.get('range_sucesso', [0, 0])
        st.success(f"""
        **Preço deve ficar entre:**  
        **R$ {range_sucesso[0]:.2f}** e **R$ {range_sucesso[1]:.2f}**
        
        **Delta CALL:** {opportunity.get('delta_calls', 0):.2f}  
        **Delta PUT:** {opportunity.get('delta_puts', 0):.2f}
        """)
    
    # Simulação de investimento
    st.markdown("**💰 Simulação de Investimento (R$ 1.000):**")
    
    premio_por_contrato = opportunity.get('premio_liquido', 0)
    risco_por_contrato = opportunity.get('risco_maximo', 0)
    contratos_simulacao = int(1000 / risco_por_contrato) if risco_por_contrato > 0 else 0
    
    if contratos_simulacao > 0:
        capital_utilizado = risco_por_contrato * contratos_simulacao
        premio_total = premio_por_contrato * contratos_simulacao
        
        lucro_maximo = premio_total
        prejuizo_maximo = capital_utilizado - premio_total
        
        prob_sucesso = opportunity.get('probabilidade_sucesso', 0)
        prob_perda = 1 - prob_sucesso
        retorno_esperado = (prob_sucesso * lucro_maximo) - (prob_perda * prejuizo_maximo)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "📈 Lucro Máximo", 
                f"R$ {lucro_maximo:.2f}", 
                help=f"Se o preço ficar no range ({contratos_simulacao} contratos)"
            )
        with col2:
            st.metric(
                "📉 Prejuízo Máximo", 
                f"R$ {prejuizo_maximo:.2f}", 
                help=f"Se o preço sair do range ({contratos_simulacao} contratos)"
            )
        with col3:
            st.metric(
                "🎯 Retorno Esperado", 
                f"R$ {retorno_esperado:.2f}", 
                help=f"Valor esperado com {prob_sucesso:.1%} de chance de sucesso"
            )
    else:
        st.warning("⚠️ **Capital insuficiente** para operar com R$ 1.000 (risco muito alto)")
    
    # Guia detalhado da operação
    st.markdown("**📋 Guia Completo da Operação:**")
    strikes = opportunity.get('strikes', {})
    precos = opportunity.get('precos_opcoes', {})
    ticker = opportunity.get('underlying', 'N/A')
    vencimento = opportunity.get('vencimento', 'N/A')
    
    st.info(f"""
    **🎯 IRON CONDOR - {ticker}**
    - **Vencimento:** {vencimento}
    - **Contratos por perna:** {contratos_simulacao}
    - **Capital necessário:** R$ {capital_utilizado:.2f}
    """)
    
    st.markdown("**📝 Instruções Passo a Passo:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🟢 CALLs (Trava de Alta) - VENDA:**")
        st.code(f"""
1️⃣ VENDA {contratos_simulacao} CALLs
   Ticker: {ticker}CA{strikes.get('call_short', 0):.0f}{vencimento.replace('-', '')}
   Strike: R$ {strikes.get('call_short', 0):.2f}
   Prêmio: R$ {precos.get('call_short', 0):.2f} por contrato
   Total recebido: R$ {precos.get('call_short', 0) * contratos_simulacao:.2f}

2️⃣ COMPRE {contratos_simulacao} CALLs
   Ticker: {ticker}CA{strikes.get('call_long', 0):.0f}{vencimento.replace('-', '')}
   Strike: R$ {strikes.get('call_long', 0):.2f}
   Prêmio: R$ {precos.get('call_long', 0):.2f} por contrato
   Total pago: R$ {precos.get('call_long', 0) * contratos_simulacao:.2f}

💰 Resultado CALLs: R$ {(precos.get('call_short', 0) - precos.get('call_long', 0)) * contratos_simulacao:.2f}
        """)
    with col2:
        st.markdown("**🔴 PUTs (Trava de Baixa) - VENDA:**")
        st.code(f"""
1️⃣ VENDA {contratos_simulacao} PUTs
   Ticker: {ticker}PU{strikes.get('put_short', 0):.0f}{vencimento.replace('-', '')}
   Strike: R$ {strikes.get('put_short', 0):.2f}
   Prêmio: R$ {precos.get('put_short', 0):.2f} por contrato
   Total recebido: R$ {precos.get('put_short', 0) * contratos_simulacao:.2f}

2️⃣ COMPRE {contratos_simulacao} PUTs
   Ticker: {ticker}PU{strikes.get('put_long', 0):.0f}{vencimento.replace('-', '')}
   Strike: R$ {strikes.get('put_long', 0):.2f}
   Prêmio: R$ {precos.get('put_long', 0):.2f} por contrato
   Total pago: R$ {precos.get('put_long', 0) * contratos_simulacao:.2f}

💰 Resultado PUTs: R$ {(precos.get('put_short', 0) - precos.get('put_long', 0)) * contratos_simulacao:.2f}
        """)
    
    # Resumo financeiro
    st.markdown("**💰 Resumo Financeiro:**")
    premio_call = (precos.get('call_short', 0) - precos.get('call_long', 0)) * contratos_simulacao
    premio_put = (precos.get('put_short', 0) - precos.get('put_long', 0)) * contratos_simulacao
    premio_total_confirmado = premio_call + premio_put
    col1, col2, col3 = st.columns(3)
    with col1: 
        st.metric("Prêmio CALLs", f"R$ {premio_call:.2f}")
    with col2: 
        st.metric("Prêmio PUTs", f"R$ {premio_put:.2f}")
    with col3: 
        st.metric("Prêmio Total", f"R$ {premio_total_confirmado:.2f}")
    
    # Condições de sucesso
    st.markdown("**🎯 Condições de Sucesso:**")
    range_sucesso = opportunity.get('range_sucesso', [0, 0])
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"""
        **✅ LUCRO MÁXIMO**
        Preço entre: R$ {range_sucesso[0]:.2f} e R$ {range_sucesso[1]:.2f}
        Probabilidade: {opportunity.get('probabilidade_sucesso', 0):.1%}
        Lucro: R$ {lucro_maximo:.2f}
        """)
    with col2:
        st.error(f"""
        **❌ PREJUÍZO MÁXIMO**
        Preço < R$ {range_sucesso[0]:.2f} OU > R$ {range_sucesso[1]:.2f}
        Probabilidade: {1 - opportunity.get('probabilidade_sucesso', 0):.1%}
        Prejuízo: R$ {prejuizo_maximo:.2f}
        """)
    
    # Gráfico de payoff
    st.markdown("**📈 Gráfico de Payoff:**")
    try:
        render_payoff_chart(opportunity)
    except Exception as e:
        st.error(f"Erro ao gerar gráfico de payoff: {e}")


def render_summary_metrics(opportunities: List[Dict[str, Any]]):
    """Renderiza métricas resumo das oportunidades."""
    
    # Calcula estatísticas
    probabilities = [op["probabilidade_sucesso"] for op in opportunities]
    returns = [op["retorno_esperado_pct"] for op in opportunities]
    premiums = [op["premio_liquido"] for op in opportunities]
    risks = [op["risco_maximo"] for op in opportunities]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Prob. Média", 
            f"{sum(probabilities)/len(probabilities):.1%}",
            help="Probabilidade média de sucesso"
        )
    
    with col2:
        st.metric(
            "Ret. Esperado Médio", 
            f"{sum(returns)/len(returns):.1f}%",
            help="Retorno esperado médio anualizado"
        )
    
    with col3:
        st.metric(
            "Prêmio Médio", 
            f"R$ {sum(premiums)/len(premiums):.2f}",
            help="Prêmio líquido médio recebido"
        )
    
    with col4:
        st.metric(
            "Risco Médio", 
            f"R$ {sum(risks)/len(risks):.2f}",
            help="Risco máximo médio"
        )


def render_opportunity_details(opportunity: Dict[str, Any]):
    """Renderiza detalhes de uma oportunidade específica."""
    
    st.subheader("🎯 Melhor Oportunidade")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Probabilidade Sucesso",
            f"{opportunity['probabilidade_sucesso']:.1%}",
            help="Chance do preço ficar no range"
        )
    
    with col2:
        st.metric(
            "Retorno Esperado",
            f"{opportunity['retorno_esperado_pct']:.1f}%",
            help="Retorno esperado anualizado"
        )
    
    with col3:
        st.metric(
            "Prêmio Líquido",
            f"R$ {opportunity['premio_liquido']:.2f}",
            help="Valor recebido líquido"
        )
    
    with col4:
        st.metric(
            "Risco Máximo",
            f"R$ {opportunity['risco_maximo']:.2f}",
            help="Perda máxima possível"
        )
    
    # Simulação de investimento
    st.markdown("**💰 Simulação de Investimento (R$ 1.000):**")
    
    # Calcula quantos contratos podem ser operados com R$ 1.000
    premio_por_contrato = opportunity['premio_liquido']
    risco_por_contrato = opportunity['risco_maximo']
    
    # O capital necessário é sempre o risco máximo por contrato
    # Calcula quantos contratos cabem em R$ 1.000 baseado no risco
    contratos_simulacao = int(1000 / risco_por_contrato) if risco_por_contrato > 0 else 0
    
    if contratos_simulacao > 0:
        # Calcula cenários com precisão
        capital_utilizado = risco_por_contrato * contratos_simulacao  # Capital necessário
        premio_total = premio_por_contrato * contratos_simulacao      # Prêmio recebido
        
        # Lucro máximo = prêmio total recebido
        lucro_maximo = premio_total
        
        # Prejuízo máximo = capital necessário - prêmio recebido
        prejuizo_maximo = capital_utilizado - premio_total
        
        # Retorno esperado em valor
        prob_sucesso = opportunity['probabilidade_sucesso']
        prob_perda = 1 - prob_sucesso
        retorno_esperado = (prob_sucesso * lucro_maximo) - (prob_perda * prejuizo_maximo)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📈 **Lucro Máximo**",
                f"R$ {lucro_maximo:.2f}",
                help=f"Se o preço ficar no range ({contratos_simulacao} contratos)"
            )
        
        with col2:
            st.metric(
                "📉 **Prejuízo Máximo**",
                f"R$ {prejuizo_maximo:.2f}",
                help=f"Se o preço sair do range ({contratos_simulacao} contratos)"
            )
        
        with col3:
            st.metric(
                "🎯 **Retorno Esperado**",
                f"R$ {retorno_esperado:.2f}",
                help=f"Valor esperado com {prob_sucesso:.1%} de chance de sucesso"
            )
    else:
        st.warning("⚠️ **Capital insuficiente** para operar com R$ 1.000 (risco muito alto)")
    
    # Guia detalhado da operação
    st.markdown("**📋 Guia Completo da Operação:**")
    
    strikes = opportunity['strikes']
    precos = opportunity['precos_opcoes']
    ticker = opportunity['underlying']
    vencimento = opportunity['vencimento']
    
    # Cabeçalho da operação
    st.info(f"""
    **🎯 IRON CONDOR - {ticker}**
    - **Vencimento:** {vencimento}
    - **Contratos por perna:** {contratos_simulacao}
    - **Capital necessário:** R$ {capital_utilizado:.2f}
    """)
    
    # Instruções passo a passo
    st.markdown("**📝 Instruções Passo a Passo:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 CALLs (Trava de Alta) - VENDA:**")
        st.code(f"""
1️⃣ VENDA {contratos_simulacao} CALLs
   Ticker: {ticker}CA{strikes['call_short']:.0f}{vencimento.replace('-', '')}
   Strike: R$ {strikes['call_short']:.2f}
   Prêmio: R$ {precos['call_short']:.2f} por contrato
   Total recebido: R$ {precos['call_short'] * contratos_simulacao:.2f}

2️⃣ COMPRE {contratos_simulacao} CALLs
   Ticker: {ticker}CA{strikes['call_long']:.0f}{vencimento.replace('-', '')}
   Strike: R$ {strikes['call_long']:.2f}
   Prêmio: R$ {precos['call_long']:.2f} por contrato
   Total pago: R$ {precos['call_long'] * contratos_simulacao:.2f}

💰 Resultado CALLs: R$ {(precos['call_short'] - precos['call_long']) * contratos_simulacao:.2f}
        """)
    
    with col2:
        st.markdown("**🔴 PUTs (Trava de Baixa) - VENDA:**")
        st.code(f"""
1️⃣ VENDA {contratos_simulacao} PUTs
   Ticker: {ticker}PU{strikes['put_short']:.0f}{vencimento.replace('-', '')}
   Strike: R$ {strikes['put_short']:.2f}
   Prêmio: R$ {precos['put_short']:.2f} por contrato
   Total recebido: R$ {precos['put_short'] * contratos_simulacao:.2f}

2️⃣ COMPRE {contratos_simulacao} PUTs
   Ticker: {ticker}PU{strikes['put_long']:.0f}{vencimento.replace('-', '')}
   Strike: R$ {strikes['put_long']:.2f}
   Prêmio: R$ {precos['put_long']:.2f} por contrato
   Total pago: R$ {precos['put_long'] * contratos_simulacao:.2f}

💰 Resultado PUTs: R$ {(precos['put_short'] - precos['put_long']) * contratos_simulacao:.2f}
        """)
    
    # Resumo financeiro
    st.markdown("**💰 Resumo Financeiro:**")
    
    premio_call = (precos['call_short'] - precos['call_long']) * contratos_simulacao
    premio_put = (precos['put_short'] - precos['put_long']) * contratos_simulacao
    premio_total_confirmado = premio_call + premio_put
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Prêmio CALLs", f"R$ {premio_call:.2f}")
    with col2:
        st.metric("Prêmio PUTs", f"R$ {premio_put:.2f}")
    with col3:
        st.metric("Prêmio Total", f"R$ {premio_total_confirmado:.2f}")
    
    # Range de sucesso
    st.markdown("**🎯 Condições de Sucesso:**")
    range_sucesso = opportunity['range_sucesso']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"""
        **✅ LUCRO MÁXIMO**
        Preço entre: R$ {range_sucesso[0]:.2f} e R$ {range_sucesso[1]:.2f}
        Probabilidade: {opportunity['probabilidade_sucesso']:.1%}
        Lucro: R$ {lucro_maximo:.2f}
        """)
    
    with col2:
        st.error(f"""
        **❌ PREJUÍZO MÁXIMO**
        Preço < R$ {range_sucesso[0]:.2f} OU > R$ {range_sucesso[1]:.2f}
        Probabilidade: {1 - opportunity['probabilidade_sucesso']:.1%}
        Prejuízo: R$ {prejuizo_maximo:.2f}
        """)
    
    # Visualização do payoff
    render_payoff_chart(opportunity)


def render_payoff_chart(opportunity: Dict[str, Any]):
    """Renderiza gráfico de payoff da estrutura Iron Condor."""
    
    try:
        st.markdown("**📈 Gráfico de Payoff:**")
        
        strikes = opportunity['strikes']
        premio_liquido = opportunity['premio_liquido']
        
        # Verifica se os dados necessários estão presentes
        if not strikes or 'call_short' not in strikes or 'call_long' not in strikes or 'put_short' not in strikes or 'put_long' not in strikes:
            st.warning("⚠️ Dados insuficientes para gerar o gráfico de payoff.")
            return
        
        # Define pontos para o gráfico
        price_points = []
        payoff_points = []
        
        # Preço mínimo (abaixo do PUT long)
        min_price = strikes['put_long'] * 0.95
        # Preço máximo (acima do CALL long)
        max_price = strikes['call_long'] * 1.05
        
        # Gera pontos
        prices = np.linspace(min_price, max_price, 100)
        
        for price in prices:
            payoff = calculate_iron_condor_payoff(
                price, strikes, premio_liquido
            )
            price_points.append(price)
            payoff_points.append(payoff)
        
        # Cria gráfico
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=price_points,
            y=payoff_points,
            mode='lines',
            name='Payoff',
            line=dict(color='blue', width=2),
            hovertemplate='Preço: R$ %{x:.2f}<br>Payoff: R$ %{y:.2f}<extra></extra>'
        ))
        
        # Adiciona linhas de referência
        fig.add_vline(x=strikes['put_long'], line_dash="dash", line_color="red", 
                      annotation_text="PUT Long")
        fig.add_vline(x=strikes['put_short'], line_dash="dash", line_color="orange", 
                      annotation_text="PUT Short")
        fig.add_vline(x=strikes['call_short'], line_dash="dash", line_color="orange", 
                      annotation_text="CALL Short")
        fig.add_vline(x=strikes['call_long'], line_dash="dash", line_color="red", 
                      annotation_text="CALL Long")
        
        # Linha de payoff zero
        fig.add_hline(y=0, line_dash="dot", line_color="gray", 
                      annotation_text="Break-even")
        
        # Configuração do gráfico
        fig.update_layout(
            title="Payoff da Estrutura Iron Condor",
            xaxis_title="Preço do Ativo",
            yaxis_title="Payoff (R$)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
        
    except Exception as e:
        st.error(f"❌ Erro ao gerar gráfico de payoff: {e}")
        st.info("💡 Verifique se os dados da oportunidade estão completos.")


def calculate_iron_condor_payoff(price: float, strikes: Dict[str, float], premio_liquido: float) -> float:
    """Calcula o payoff de uma estrutura Iron Condor para um preço dado."""
    
    # CALL spread payoff
    call_short_payoff = max(0, price - strikes['call_short'])
    call_long_payoff = max(0, price - strikes['call_long'])
    call_spread_payoff = call_long_payoff - call_short_payoff
    
    # PUT spread payoff
    put_short_payoff = max(0, strikes['put_short'] - price)
    put_long_payoff = max(0, strikes['put_long'] - price)
    put_spread_payoff = put_long_payoff - put_short_payoff
    
    # Payoff total (soma dos spreads + prêmio recebido)
    total_payoff = call_spread_payoff + put_spread_payoff + premio_liquido
    
    return total_payoff


if __name__ == "__main__":
    render_iron_condor_scanner_page()


