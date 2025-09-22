"""
Página do Radar de Direção Profissional - Análise em 6 camadas.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
import datetime

from src.core.data.oplab_client import OpLabClient
from src.features.income.synthetic_dividends import find_synthetic_dividend_options
from src.core.professional.professional_analysis import ProfessionalAnalyzer, Direction
from src.core.data.data import get_price_history


def render_professional_radar_page():
    """Renderiza a página do Radar de Direção Profissional."""
    st.title("🎯 Radar de Direção Profissional")
    st.markdown("**Análise institucional em 6 camadas: Tendência, Momentum, Volume, Sentimento, Macro e Decisão**")
    
    # Inputs
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input("Ticker", value="PETR4", help="Ex: PETR4, VALE3, ITUB4", key="professional_ticker")
    
    with col2:
        horizon = st.selectbox(
            "Horizonte de Análise",
            options=["curto", "médio", "longo"],
            index=1,  # Default: "médio"
            help="Adapta parâmetros técnicos ao horizonte temporal",
            key="professional_horizon"
        )
    
    # Botão para análise
    if st.button("🔍 Análise Profissional", type="primary", key="professional_analyze"):
        if not ticker.strip():
            st.error("Digite um ticker válido")
            return
            
        try:
            # Busca dados
            client = OpLabClient()
            
            # Mapeamento de horizontes para dias históricos
            history_days_map = {
                "curto": 200,
                "médio": 400,
                "longo": 600
            }
            
            # Dados históricos
            historical_data = get_price_history(ticker.strip().upper(), history_days_map[horizon])
            
            # Validação de dados
            if historical_data.empty:
                st.error("❌ **Dados históricos não disponíveis**")
                st.warning("⚠️ **A API OpLab não fornece dados históricos**")
                st.info("💡 **Análise técnica profissional requer dados históricos reais**")
                st.info("🔧 **Soluções possíveis:**")
                st.info("   • Integrar com Yahoo Finance (yfinance)")
                st.info("   • Usar outro provedor de dados históricos")
                st.info("   • Aguardar OpLab implementar endpoint de histórico")
                return
            elif len(historical_data) < 50:
                st.error(f"❌ **Dados insuficientes para análise profissional**")
                st.info(f"📊 Recebido: {len(historical_data)} registros, mínimo: 50")
                return
            
            # Analisador profissional com horizonte adaptativo
            analyzer = ProfessionalAnalyzer(client, horizon)
            
            # Executa análise completa
            with st.spinner("Executando análise profissional..."):
                analysis = analyzer.analyze(ticker.strip().upper(), historical_data)
            
            # Renderiza resultados
            render_professional_analysis(analysis, horizon)
            
            # Busca opções se há direção definida
            if analysis.direction != Direction.NEUTRAL:
                render_professional_options(analysis, client)
            else:
                st.info("🔍 **SINAL NEUTRO** - Recomendo apenas Dividendos Sintéticos ou estratégias de volatilidade.")
                
        except Exception as e:
            st.error(f"Erro na análise: {e}")


def render_professional_analysis(analysis, horizon: str):
    """Renderiza os resultados da análise profissional."""
    
    # Header com resultado principal
    st.markdown("---")
    
    if analysis.direction == Direction.CALL:
        st.success(f"📈 **TENDÊNCIA DE ALTA** (Confiança: {analysis.confidence:.0f}%)")
        color = "green"
    elif analysis.direction == Direction.PUT:
        st.error(f"📉 **TENDÊNCIA DE BAIXA** (Confiança: {analysis.confidence:.0f}%)")
        color = "red"
    else:
        st.warning(f"⚖️ **SINAL NEUTRO** (Confiança: {analysis.confidence:.0f}%)")
        color = "orange"
    
    # Score final
    score_color = "green" if analysis.final_score > 0 else "red" if analysis.final_score < 0 else "gray"
    st.metric(
        "Score Final", 
        f"{analysis.final_score:+.2f}", 
        help="Score normalizado de -1 (forte baixa) a +1 (forte alta)"
    )
    
    # Drivers principais
    if analysis.key_drivers:
        st.markdown("**🎯 Drivers Principais:**")
        for driver in analysis.key_drivers:
            st.write(f"• {driver}")
    
    # Estratégia recomendada
    st.info(f"💡 **Estratégia:** {analysis.strategy_recommendation}")
    
    # Análises por camada
    st.subheader("📊 Análise por Camadas")
    
    # Camada 1: Tendência
    with st.expander("🏗️ Camada 1: Tendência Estrutural", expanded=True):
        render_trend_analysis(analysis.trend)
    
    # Camada 2: Momentum
    with st.expander("⚡ Camada 2: Momentum e Força"):
        render_momentum_analysis(analysis.momentum)
    
    # Camada 3: Volume e Fluxo
    with st.expander("📈 Camada 3: Volume e Fluxo"):
        render_volume_analysis(analysis.volume_flow)
    
    # Camada 4: Sentimento via Opções
    with st.expander("🎭 Camada 4: Sentimento do Mercado"):
        render_sentiment_analysis(analysis.options_sentiment)
    
    # Camada 5: Contexto Macro
    with st.expander("🌍 Camada 5: Contexto Macro e Setorial"):
        render_macro_analysis(analysis.macro_context)
    
    # Camada 6: Decisão Final
    with st.expander("🎯 Camada 6: Modelo de Decisão"):
        render_decision_analysis(analysis)


def render_trend_analysis(trend):
    """Renderiza análise de tendência."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("SMA 10", f"R$ {trend.sma_10:.2f}")
        st.metric("SMA 50", f"R$ {trend.sma_50:.2f}")
    
    with col2:
        st.metric("SMA 100", f"R$ {trend.sma_100:.2f}")
        st.metric("SMA 200", f"R$ {trend.sma_200:.2f}")
    
    with col3:
        cross_status = "🟢 Golden Cross" if trend.golden_cross else "🔴 Death Cross" if trend.death_cross else "⚪ Neutro"
        st.metric("Cruzamento", cross_status)
        
        vol_regime = {"ALTA": "🔴", "NORMAL": "🟡", "BAIXA": "🟢"}[trend.volatility_regime]
        st.metric("Volatilidade", f"{vol_regime} {trend.volatility_regime}")
    
    # Score de tendência
    trend_color = "green" if trend.trend_score > 0 else "red" if trend.trend_score < 0 else "gray"
    st.metric("Score Tendência", f"{trend.trend_score:+.2f}", help="Força da tendência estrutural")
    
    st.write(f"**Força:** {trend.trend_strength.value}")
    st.write(f"**ATR Ratio:** {trend.atr_ratio:.2f}x")


def render_momentum_analysis(momentum):
    """Renderiza análise de momentum."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RSI 7", f"{momentum.rsi_7:.1f}")
        st.metric("RSI 14", f"{momentum.rsi_14:.1f}")
    
    with col2:
        st.metric("RSI 21", f"{momentum.rsi_21:.1f}")
        st.metric("MACD Hist", f"{momentum.macd_histogram:.3f}")
    
    with col3:
        st.metric("ADX", f"{momentum.adx:.1f}")
        st.metric("ROC 10d", f"{momentum.roc_10:+.1f}%")
    
    # Score de momentum
    momentum_color = "green" if momentum.momentum_score > 0 else "red" if momentum.momentum_score < 0 else "gray"
    st.metric("Score Momentum", f"{momentum.momentum_score:+.2f}", help="Força do momentum atual")
    
    st.write(f"**Força:** {momentum.momentum_strength}")
    st.write(f"**ROC 20d:** {momentum.roc_20:+.1f}%")


def render_volume_analysis(volume_flow):
    """Renderiza análise de volume e fluxo."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Volume 5d", f"{volume_flow.volume_ratio_5d:.1f}x")
        st.metric("Volume 20d", f"{volume_flow.volume_ratio_20d:.1f}x")
    
    with col2:
        st.metric("OBV Trend", f"{volume_flow.obv_trend:+.2f}")
        st.metric("A/D Trend", f"{volume_flow.accumulation_distribution:+.2f}")
    
    with col3:
        flow_emoji = {"ENTRADA": "📈", "SAÍDA": "📉", "NEUTRO": "➡️"}[volume_flow.flow_direction]
        st.metric("Fluxo", f"{flow_emoji} {volume_flow.flow_direction}")
    
    # Score de volume
    volume_color = "green" if volume_flow.volume_score > 0 else "red" if volume_flow.volume_score < 0 else "gray"
    st.metric("Score Volume", f"{volume_flow.volume_score:+.2f}", help="Confirmação por volume e fluxo")


def render_sentiment_analysis(sentiment):
    """Renderiza análise de sentimento."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Put/Call Ratio", f"{sentiment.put_call_ratio:.2f}")
        st.metric("Volatility Skew", f"{sentiment.volatility_skew:+.3f}")
    
    with col2:
        st.metric("Call Volume", f"{sentiment.call_volume_ratio:.1%}")
        st.metric("Put Volume", f"{sentiment.put_volume_ratio:.1%}")
    
    with col3:
        bias_emoji = {"BULLISH": "🐂", "BEARISH": "🐻", "NEUTRAL": "🐨"}[sentiment.market_bias]
        st.metric("Market Bias", f"{bias_emoji} {sentiment.market_bias}")
    
    # Score de sentimento
    sentiment_color = "green" if sentiment.sentiment_score > 0 else "red" if sentiment.sentiment_score < 0 else "gray"
    st.metric("Score Sentimento", f"{sentiment.sentiment_score:+.2f}", help="Sentimento do mercado via opções")


def render_macro_analysis(macro):
    """Renderiza análise macro e setorial."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Setor", f"{macro.sector_score:+.2f}")
    
    with col2:
        st.metric("Macro", f"{macro.macro_score:+.2f}")
    
    with col3:
        st.metric("Commodities", f"{macro.commodity_score:+.2f}")
    
    # Score geral
    context_color = "green" if macro.overall_context_score > 0 else "red" if macro.overall_context_score < 0 else "gray"
    st.metric("Score Contexto", f"{macro.overall_context_score:+.2f}", help="Contexto macro e setorial")
    
    st.write(f"**Bias:** {macro.context_bias}")


def render_decision_analysis(analysis):
    """Renderiza modelo de decisão."""
    st.markdown("**📊 Pesos das Camadas:**")
    
    weights = {
        'Tendência': 45,
        'Momentum': 25,
        'Volume': 15,
        'Sentimento': 10,
        'Macro': 5
    }
    
    for layer, weight in weights.items():
        st.write(f"• **{layer}:** {weight}%")
    
    st.markdown("**🎯 Critérios de Decisão:**")
    st.write("• **Score ≥ +0.3:** CALL (tendência de alta)")
    st.write("• **Score ≤ -0.3:** PUT (tendência de baixa)")
    st.write("• **Score entre -0.3 e +0.3:** NEUTRO (sem sinal claro)")
    
    # Score detalhado
    st.markdown("**📈 Contribuição de cada camada:**")
    contributions = {
        'Tendência': analysis.trend.trend_score * 0.45,
        'Momentum': analysis.momentum.momentum_score * 0.25,
        'Volume': analysis.volume_flow.volume_score * 0.15,
        'Sentimento': analysis.options_sentiment.sentiment_score * 0.10,
        'Macro': analysis.macro_context.overall_context_score * 0.05
    }
    
    for layer, contribution in contributions.items():
        color = "green" if contribution > 0 else "red" if contribution < 0 else "gray"
        st.write(f"• **{layer}:** {contribution:+.3f}")


def render_professional_options(analysis, client: OpLabClient):
    """Renderiza recomendações de opções profissionais."""
    st.subheader("🎯 Opções Recomendadas")
    
    # Mapeia direção para parâmetros de busca
    if analysis.direction == Direction.CALL:
        option_params = {
            "min_days": 15,
            "max_days": 60,
            "min_volume": 5,
            "max_exercise_prob": 10.0,  # Baixo risco para CALL
            "option_types": "Apenas CALL"
        }
    else:  # PUT
        option_params = {
            "min_days": 15,
            "max_days": 60,
            "min_volume": 5,
            "max_exercise_prob": 10.0,  # Baixo risco para PUT
            "option_types": "Apenas PUT"
        }
    
    # Busca opções
    try:
        df = find_synthetic_dividend_options(
            analysis.ticker,
            client=client,
            **option_params
        )
        
        # Filtra pela direção
        if analysis.direction == Direction.CALL:
            df = df[df["Estratégia"] == "CALL"]
        else:
            df = df[df["Estratégia"] == "PUT"]
        
        if df.empty:
            st.info(f"Nenhuma opção {analysis.direction.value} encontrada nos critérios profissionais.")
            return
        
        # Enriquece a tabela com métricas adicionais para melhor análise
        display_df = df.copy()
        
        # Calcula métricas adicionais para visualização (sem alterar modelo)
        current_price = analysis.current_price
        
        # 1. Distância do preço atual
        display_df["Distância (%)"] = ((display_df["Strike"] / current_price) - 1) * 100
        
        # 2. Dias restantes
        from datetime import datetime
        display_df["Dias Restantes"] = (pd.to_datetime(display_df["Validade"]) - datetime.now()).dt.days
        
        # 3. Adequação temporal (movimento necessário vs tempo disponível)
        display_df["Mov./Dia (%)"] = abs(display_df["Distância (%)"]) / display_df["Dias Restantes"]
        
        # 4. Classificação de adequação temporal
        def classify_time_adequacy(row):
            mov_per_day = row["Mov./Dia (%)"]
            if mov_per_day <= 0.3:
                return "🟢 Adequado"
            elif mov_per_day <= 0.5:
                return "🟡 Apertado"
            else:
                return "🔴 Arriscado"
        
        display_df["Adequação Temporal"] = display_df.apply(classify_time_adequacy, axis=1)
        
        # 5. Volume formatado
        display_df["Volume"] = display_df["Contratos ativos"]
        
        # Seleciona e ordena colunas para exibição
        display_columns = [
            "Opção", "Strike", "Distância (%)", "Validade", "Dias Restantes",
            "Prêmio (R$)", "Retorno a.a. (%)", "Prob. Exercício (%)",
            "Volume", "Mov./Dia (%)", "Adequação Temporal"
        ]
        
        display_df = display_df[display_columns].copy()
        
        # Ordena por melhor retorno anualizado
        display_df = display_df.sort_values("Retorno a.a. (%)", ascending=False)
        
        # Configuração das colunas
        column_config = {
            "Strike": st.column_config.NumberColumn("Strike", format="R$ %.2f"),
            "Distância (%)": st.column_config.NumberColumn("Distância (%)", format="%+.1f%%", help="Distância do strike em relação ao preço atual"),
            "Dias Restantes": st.column_config.NumberColumn("Dias", format="%d"),
            "Prêmio (R$)": st.column_config.NumberColumn("Prêmio (R$)", format="R$ %.2f"),
            "Retorno a.a. (%)": st.column_config.NumberColumn("Retorno a.a. (%)", format="%.1f%%"),
            "Prob. Exercício (%)": st.column_config.NumberColumn("Prob. Exercício (%)", format="%.0f%%"),
            "Volume": st.column_config.NumberColumn("Volume", format="%d"),
            "Mov./Dia (%)": st.column_config.NumberColumn("Mov./Dia (%)", format="%.2f%%", help="Movimento diário necessário para atingir o strike"),
            "Adequação Temporal": st.column_config.TextColumn("Adequação Temporal", help="Análise se o prazo é adequado para o movimento necessário"),
        }
        
        st.dataframe(display_df, use_container_width=True, column_config=column_config, hide_index=True)
        
        # Resumo profissional
        st.info(f"""
        **📊 Análise Profissional:**
        - **Direção:** {analysis.direction.value} (Confiança: {analysis.confidence:.0f}%)
        - **Score Final:** {analysis.final_score:+.2f}
        - **Estratégia:** {analysis.strategy_recommendation}
        - **Opções encontradas:** {len(df)}
        """)
        
        # Guia de interpretação das novas métricas
        with st.expander("📖 Como interpretar as métricas da tabela"):
            st.markdown("""
            **🎯 Distância (%):** Quanto o strike está acima/abaixo do preço atual
            - 🟢 **0% a ±5%**: Próximo ao dinheiro (ATM/ITM) - maior probabilidade
            - 🟡 **±5% a ±10%**: Moderadamente fora do dinheiro - equilibrado
            - 🔴 **>±10%**: Muito fora do dinheiro - alta alavancagem, alto risco
            
            **⏰ Adequação Temporal:** Análise do prazo vs movimento necessário
            - 🟢 **Adequado**: ≤0.3% movimento/dia - prazo confortável
            - 🟡 **Apertado**: 0.3-0.5% movimento/dia - requer atenção
            - 🔴 **Arriscado**: >0.5% movimento/dia - movimento muito exigente
            
            **💰 Mov./Dia (%):** Movimento diário médio necessário para atingir o strike
            - Baseado na distância atual dividida pelos dias restantes
            - Compare com a volatilidade histórica do ativo (~1-2% dia para ITSA4)
            
            **📊 Volume:** Número de contratos negociados
            - >10.000: Alta liquidez - facilita entrada/saída
            - 1.000-10.000: Liquidez moderada
            - <1.000: Baixa liquidez - pode ter spread alto
            """)
        
        # Alerta sobre adequação temporal se necessário
        risky_options = display_df[display_df["Adequação Temporal"] == "🔴 Arriscado"]
        if not risky_options.empty:
            st.warning(f"""
            ⚠️ **Atenção:** {len(risky_options)} opção(ões) classificada(s) como "🔴 Arriscado" 
            devido ao movimento diário exigente (>{0.5:.1f}% por dia).
            
            Considere que ITSA4 tem volatilidade histórica de ~1.2% ao dia. 
            Movimentos >0.5% por dia requerem tendência muito forte.
            """)
        
    except Exception as e:
        st.error(f"Erro ao buscar opções: {e}")


if __name__ == "__main__":
    render_professional_radar_page()
