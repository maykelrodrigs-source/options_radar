"""
Interface Streamlit para execução e visualização de backtests.
Permite configurar parâmetros e visualizar resultados interativamente.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import Optional

from src.core.backtesting.backtest import BacktestEngine, BacktestResult, BacktestProfessionalAnalyzer
from src.core.data.data import get_historical_data_cached
from src.core.backtesting.plots import BacktestPlotter, create_all_plots


def render_backtest_page():
    """Renderiza a página principal do backtest."""
    st.title("🔬 Backtest do ProfessionalAnalyzer")
    st.markdown("Valide a eficácia do analisador profissional em dados históricos")
    
    # Sidebar com parâmetros
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Parâmetros básicos
        ticker = st.text_input(
            "Ticker",
            value="ITUB4",
            help="Código do ativo para análise"
        ).strip().upper()
        
        # Horizonte temporal
        st.subheader("⏱️ Horizonte Temporal")
        horizon = st.selectbox(
            "Horizonte de análise técnica",
            options=["curto", "médio", "longo"],
            index=0,  # Default: "curto"
            help="Define os parâmetros dos indicadores técnicos (SMA, RSI, MACD)"
        )
        
        # Período do backtest
        st.subheader("📅 Período")
        
        col1, col2 = st.columns(2)
        with col1:
            years_back = st.selectbox(
                "Período",
                options=[1, 2, 3, 5],
                index=1,  # Default: 2 anos
                help="Anos de histórico para backtest"
            )
        
        with col2:
            end_date = st.date_input(
                "Data final",
                value=datetime.now().date(),
                help="Data final do backtest"
            )
        
        start_date = datetime.combine(end_date, datetime.min.time()) - timedelta(days=years_back * 365)
        
        # Parâmetros de análise
        st.subheader("🎯 Parâmetros")
        
        evaluation_days = st.slider(
            "Dias para avaliação",
            min_value=5,
            max_value=60,
            value=10,
            step=5,
            help="Dias úteis para verificar se sinal acertou"
        )
        
        rolling_window = st.slider(
            "Intervalo entre análises",
            min_value=1,
            max_value=15,
            value=5,
            step=1,
            help="Dias úteis entre cada análise"
        )
        
        success_threshold = st.slider(
            "Threshold de sucesso (%)",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Variação mínima para considerar acerto"
        )
        
        # Botões de execução
        col1, col2 = st.columns([2, 1])
        with col1:
            run_backtest = st.button(
                "🚀 Executar Backtest",
                type="primary",
                help="Inicia o backtest com os parâmetros configurados"
            )
        with col2:
            if st.button("🗑️ Limpar", help="Limpa resultado anterior"):
                if 'backtest_result' in st.session_state:
                    del st.session_state['backtest_result']
                if 'backtest_params' in st.session_state:
                    del st.session_state['backtest_params']
                st.rerun()
    
    # Verifica se deve executar novo backtest
    should_run_backtest = False
    
    if run_backtest:
        # Validações antes de executar
        if not ticker:
            st.error("❌ Digite um ticker válido")
            return
        
        if years_back < 1:
            st.error("❌ Período deve ser pelo menos 1 ano")
            return
            
        should_run_backtest = True
    
    # Verifica se já existe resultado na sessão
    has_existing_result = (
        'backtest_result' in st.session_state and 
        st.session_state['backtest_result'] is not None
    )
    
    # Área principal - decide o que mostrar
    if should_run_backtest:
        # Executa novo backtest
        with st.spinner(f"🔍 Executando backtest para {ticker}..."):
            try:
                # Configura analisador com defaults otimizados para curto prazo
                analyzer = BacktestProfessionalAnalyzer(horizon=horizon)
                analyzer.decision_threshold = 0.22
                analyzer.layer_weights = {
                    'trend': 0.30,
                    'momentum': 0.30,
                    'volume': 0.20,
                    'sentiment': 0.15,
                    'macro': 0.05
                }

                engine = BacktestEngine(
                    success_threshold=success_threshold,
                    horizon=horizon,
                    data_provider=get_historical_data_cached,
                    analyzer=analyzer
                )
                
                result = engine.run_backtest(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=datetime.combine(end_date, datetime.min.time()),
                    evaluation_days=evaluation_days,
                    rolling_window=rolling_window
                )
                
                # Salva resultado na sessão para reutilização
                st.session_state['backtest_result'] = result
                st.session_state['backtest_params'] = {
                    'ticker': ticker,
                    'horizon': horizon,
                    'start_date': start_date,
                    'end_date': end_date,
                    'evaluation_days': evaluation_days,
                    'rolling_window': rolling_window,
                    'success_threshold': success_threshold
                }
                
            except Exception as e:
                st.error(f"❌ Erro ao executar backtest: {e}")
                st.exception(e)
                return
        
        # Renderiza resultados do novo backtest
        render_results_section(result)
        
    elif has_existing_result:
        # Mostra resultado existente
        result = st.session_state['backtest_result']
        params = st.session_state.get('backtest_params', {})
        
        # Mostra informações do backtest existente
        ticker_info = params.get('ticker', 'N/A')
        horizon_info = params.get('horizon', 'N/A')
        start_info = params.get('start_date', datetime.now()).strftime('%d/%m/%Y') if params.get('start_date') else 'N/A'
        end_info = params.get('end_date', datetime.now()).strftime('%d/%m/%Y') if params.get('end_date') else 'N/A'
        
        st.info(f"📊 Mostrando resultado do backtest anterior: **{ticker_info}** ({horizon_info}) | {start_info} a {end_info}")
        st.caption("💡 Para executar novo backtest, altere os parâmetros e clique em 'Executar Backtest' ou clique em 'Limpar' para remover este resultado.")
        
        # Renderiza resultados existentes
        render_results_section(result)
        
    else:
        # Mostra tela de boas-vindas
        render_welcome_section(ticker, start_date, datetime.combine(end_date, datetime.min.time()), horizon)


def render_welcome_section(ticker: str, start_date: datetime, end_date: datetime, horizon: str = "médio"):
    """Renderiza seção de boas-vindas com preview dos parâmetros."""
    st.info("👋 Configure os parâmetros na barra lateral e clique em 'Executar Backtest'")
    
    # Preview dos parâmetros
    with st.expander("👀 Preview dos Parâmetros", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Ticker", ticker)
            st.metric("Horizonte", horizon.title())
        
        with col2:
            st.metric("Data Inicial", start_date.strftime('%d/%m/%Y'))
            st.metric("Data Final", end_date.strftime('%d/%m/%Y'))
        
        with col3:
            st.metric("Período", f"{(end_date - start_date).days} dias")
            st.metric("Threshold", "3.0% (padrão)")
    
    # Explicação do processo
    st.markdown("---")
    st.markdown("### 🔬 Como Funciona o Backtest")
    
    st.markdown("""
    **1. 📊 Coleta de Dados**
    - Busca histórico de preços do ticker selecionado
    - Simula dados baseados em padrões realistas de mercado
    
    **2. 🎯 Geração de Sinais**
    - Executa ProfessionalAnalyzer com horizonte configurado
    - Gera sinais CALL/PUT/NEUTRAL com níveis de confiança
    - Registra drivers principais de cada decisão
    
    **3. ✅ Validação**
    - **CALL**: Acerto se preço subiu > +{threshold}% no horizonte
    - **PUT**: Acerto se preço caiu > -{threshold}% no horizonte  
    - **NEUTRAL**: Acerto se preço ficou entre ±{threshold}%
    
    **4. 📈 Análise**
    - Calcula taxa de acerto geral e por tipo de sinal
    - Analisa performance por nível de confiança
    - Gera visualizações interativas dos resultados
    """.format(threshold="3.0"))
    
    # Informações sobre horizontes
    st.markdown("### ⏱️ Horizontes Temporais")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📈 Curto (1-3 meses)**
        - SMA: 5/20/50/100
        - RSI: 7/14/21
        - MACD: 8/17/9
        - Foco: Movimentos rápidos
        """)
    
    with col2:
        st.markdown("""
        **📊 Médio (3-6 meses)**
        - SMA: 10/50/100/200
        - RSI: 7/14/21
        - MACD: 12/26/9
        - Foco: Tendências estruturais
        """)
    
    with col3:
        st.markdown("""
        **📉 Longo (6+ meses)**
        - SMA: 20/50/100/200
        - RSI: 14/21/28
        - MACD: 12/26/9
        - Foco: Ciclos de mercado
        """)
    
    # Exemplos de resultados esperados
    st.markdown("### 📋 Exemplo de Resultados")
    
    sample_data = {
        'Data': ['01/05/2023', '15/06/2023', '10/07/2023', '01/08/2023'],
        'Sinal': ['CALL', 'PUT', 'NEUTRAL', 'CALL'],
        'Confiança (%)': ['75%', '68%', '45%', '82%'],
        'Preço Inicial': ['R$ 28,50', 'R$ 31,20', 'R$ 27,00', 'R$ 28,00'],
        'Preço +20d': ['R$ 29,90', 'R$ 30,10', 'R$ 27,40', 'R$ 26,50'],
        'Retorno (%)': ['+4.9%', '-3.5%', '+1.5%', '-5.4%'],
        'Resultado': ['✅ Acerto', '✅ Acerto', '✅ Acerto', '❌ Erro']
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, hide_index=True)


def render_results_section(result: BacktestResult):
    """Renderiza seção completa de resultados."""
    if result.total_signals == 0:
        st.warning("⚠️ Nenhum sinal foi gerado no período especificado")
        st.info("💡 Tente ajustar os parâmetros: período maior, threshold menor ou intervalo menor")
        return
    
    # Header com estatísticas principais
    render_stats_header(result)
    
    # Abas de visualização
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Resumo", 
        "📈 Sinais vs Preço", 
        "🎯 Análise Detalhada", 
        "📋 Tabela de Sinais",
        "📁 Exportar Dados"
    ])
    
    with tab1:
        render_summary_tab(result)
    
    with tab2:
        render_signals_chart_tab(result)
    
    with tab3:
        render_detailed_analysis_tab(result)
    
    with tab4:
        render_signals_table_tab(result)
    
    with tab5:
        render_export_tab(result)


def render_stats_header(result: BacktestResult):
    """Renderiza header com estatísticas principais."""
    st.success(f"✅ Backtest concluído para **{result.ticker}**")
    
    # Métricas principais
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total de Sinais",
            result.total_signals,
            help="Número total de sinais gerados"
        )
    
    with col2:
        accuracy_color = "normal"
        if result.overall_accuracy >= 60:
            accuracy_color = "inverse"
        elif result.overall_accuracy < 40:
            accuracy_color = "off"
            
        st.metric(
            "Acurácia Geral",
            f"{result.overall_accuracy:.1f}%",
            help="Percentual de sinais que acertaram a direção"
        )
    
    with col3:
        st.metric(
            "Período",
            f"{(result.end_date - result.start_date).days}d",
            help="Período total do backtest"
        )
    
    with col4:
        st.metric(
            "Avaliação",
            f"{result.evaluation_days}d",
            help="Dias úteis para validar cada sinal"
        )
    
    with col5:
        # Performance rating
        if result.overall_accuracy >= 60:
            performance = "🎉 Excelente"
        elif result.overall_accuracy >= 50:
            performance = "👍 Boa"
        elif result.overall_accuracy >= 40:
            performance = "⚠️ Regular"
        else:
            performance = "❌ Precisa Melhorar"
        
        st.metric(
            "Performance",
            performance,
            help="Avaliação qualitativa da performance"
        )


def render_summary_tab(result: BacktestResult):
    """Renderiza aba de resumo."""
    st.subheader("📊 Estatísticas Detalhadas")
    
    # Acurácia por tipo de sinal
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Acurácia por Sinal**")
        
        signal_data = {
            'Sinal': ['CALL', 'PUT', 'NEUTRAL'],
            'Quantidade': [result.call_signals, result.put_signals, result.neutral_signals],
            'Acurácia (%)': [f"{result.call_accuracy:.1f}%", f"{result.put_accuracy:.1f}%", f"{result.neutral_accuracy:.1f}%"]
        }
        
        signal_df = pd.DataFrame(signal_data)
        st.dataframe(signal_df, hide_index=True)
    
    with col2:
        st.markdown("**🎯 Acurácia por Confiança**")
        
        high_count = len([s for s in result.signals if s.confidence >= 70])
        medium_count = len([s for s in result.signals if 50 <= s.confidence < 70])
        low_count = len([s for s in result.signals if s.confidence < 50])
        
        conf_data = {
            'Nível': ['Alta (≥70%)', 'Média (50-69%)', 'Baixa (<50%)'],
            'Quantidade': [high_count, medium_count, low_count],
            'Acurácia (%)': [
                f"{result.high_confidence_accuracy:.1f}%",
                f"{result.medium_confidence_accuracy:.1f}%", 
                f"{result.low_confidence_accuracy:.1f}%"
            ]
        }
        
        conf_df = pd.DataFrame(conf_data)
        st.dataframe(conf_df, hide_index=True)
    
    # Insights automáticos
    st.markdown("---")
    st.subheader("🔍 Insights Automáticos")
    
    insights = generate_insights(result)
    for insight in insights:
        if insight['type'] == 'success':
            st.success(f"✅ {insight['message']}")
        elif insight['type'] == 'warning':
            st.warning(f"⚠️ {insight['message']}")
        elif insight['type'] == 'info':
            st.info(f"💡 {insight['message']}")


def render_signals_chart_tab(result: BacktestResult):
    """Renderiza aba com gráfico de sinais vs preço."""
    st.subheader("📈 Sinais vs Movimento do Preço")
    
    plotter = BacktestPlotter()
    fig = plotter.plot_price_with_signals(result)
    st.plotly_chart(fig, use_container_width=True)
    
    # Timeline de performance
    st.subheader("⏱️ Performance ao Longo do Tempo")
    timeline_fig = plotter.plot_timeline_performance(result)
    st.plotly_chart(timeline_fig, use_container_width=True)


def render_detailed_analysis_tab(result: BacktestResult):
    """Renderiza aba com análises detalhadas."""
    plotter = BacktestPlotter()
    
    # Breakdown de acurácia
    st.subheader("🎯 Breakdown de Acurácia")
    accuracy_fig = plotter.plot_accuracy_breakdown(result)
    st.plotly_chart(accuracy_fig, use_container_width=True)
    
    # Distribuição de retornos
    st.subheader("📊 Distribuição de Retornos")
    returns_fig = plotter.plot_returns_distribution(result)
    st.plotly_chart(returns_fig, use_container_width=True)
    
    # Confiança vs Acurácia
    st.subheader("🎯 Confiança vs Resultado")
    conf_fig = plotter.plot_confidence_vs_accuracy(result)
    st.plotly_chart(conf_fig, use_container_width=True)


def render_signals_table_tab(result: BacktestResult):
    """Renderiza aba com tabela de sinais."""
    st.subheader("📋 Tabela Completa de Sinais")
    
    # Filtros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal_filter = st.selectbox(
            "Filtrar por sinal",
            options=['Todos', 'CALL', 'PUT', 'NEUTRAL'],
            index=0
        )
    
    with col2:
        result_filter = st.selectbox(
            "Filtrar por resultado",
            options=['Todos', 'Acertos', 'Erros'],
            index=0
        )
    
    with col3:
        confidence_filter = st.selectbox(
            "Filtrar por confiança",
            options=['Todos', 'Alta (≥70%)', 'Média (50-69%)', 'Baixa (<50%)'],
            index=0
        )
    
    # Aplica filtros
    filtered_signals = result.signals.copy()
    
    if signal_filter != 'Todos':
        filtered_signals = [s for s in filtered_signals if s.signal.value == signal_filter]
    
    if result_filter == 'Acertos':
        filtered_signals = [s for s in filtered_signals if s.is_correct]
    elif result_filter == 'Erros':
        filtered_signals = [s for s in filtered_signals if not s.is_correct]
    
    if confidence_filter == 'Alta (≥70%)':
        filtered_signals = [s for s in filtered_signals if s.confidence >= 70]
    elif confidence_filter == 'Média (50-69%)':
        filtered_signals = [s for s in filtered_signals if 50 <= s.confidence < 70]
    elif confidence_filter == 'Baixa (<50%)':
        filtered_signals = [s for s in filtered_signals if s.confidence < 50]
    
    # Cria DataFrame
    if filtered_signals:
        data = []
        for signal in filtered_signals:
            data.append({
                'Data': signal.date.strftime('%d/%m/%Y'),
                'Sinal': signal.signal.value,
                'Confiança (%)': f"{signal.confidence:.0f}%",
                'Preço Inicial': f"R$ {signal.price_at_signal:.2f}",
                f'Preço +{result.evaluation_days}d': f"R$ {signal.price_after_horizon:.2f}",
                'Retorno (%)': f"{signal.return_pct:+.1f}%",
                'Resultado': "✅ Acerto" if signal.is_correct else "❌ Erro",
                'Drivers': ", ".join(signal.drivers[:3])  # Primeiros 3 drivers
            })
        
        df = pd.DataFrame(data)
        
        st.caption(f"Mostrando {len(df)} de {len(result.signals)} sinais")
        st.dataframe(df, hide_index=True)
    else:
        st.info("Nenhum sinal encontrado com os filtros aplicados")


def render_export_tab(result: BacktestResult):
    """Renderiza aba de exportação."""
    st.subheader("📁 Exportar Resultados")
    
    # Opções de exportação
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Dados**")
        
        if st.button("📋 Baixar Tabela (CSV)", type="secondary"):
            engine = BacktestEngine()
            df = engine.export_signals_to_dataframe(result)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"backtest_{result.ticker}_{result.start_date.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        st.markdown("**📈 Gráficos**")
        st.info("💡 Use o botão de câmera nos gráficos para salvar como PNG")
    
    # Resumo para relatório
    st.markdown("---")
    st.subheader("📄 Resumo Executivo")
    
    summary_text = generate_executive_summary(result)
    st.text_area(
        "Copie este resumo para seus relatórios:",
        value=summary_text,
        height=200,
        help="Resumo formatado para copiar e colar em relatórios"
    )


def generate_insights(result: BacktestResult) -> list:
    """Gera insights automáticos baseados nos resultados."""
    insights = []
    
    # Insight sobre performance geral
    if result.overall_accuracy >= 60:
        insights.append({
            'type': 'success',
            'message': f"Performance excelente com {result.overall_accuracy:.1f}% de acurácia. O modelo está funcionando bem!"
        })
    elif result.overall_accuracy >= 50:
        insights.append({
            'type': 'info',
            'message': f"Performance boa com {result.overall_accuracy:.1f}% de acurácia. Há espaço para otimização."
        })
    else:
        insights.append({
            'type': 'warning',
            'message': f"Performance de {result.overall_accuracy:.1f}% indica necessidade de ajustes no modelo."
        })
    
    # Insight sobre confiança
    if result.high_confidence_accuracy > result.overall_accuracy + 10:
        insights.append({
            'type': 'success',
            'message': "Sinais de alta confiança têm performance significativamente melhor. Considere usar apenas sinais ≥70%."
        })
    
    # Insight sobre tipos de sinal
    best_signal = max(['CALL', 'PUT', 'NEUTRAL'], 
                     key=lambda x: getattr(result, f"{x.lower()}_accuracy"))
    best_accuracy = getattr(result, f"{best_signal.lower()}_accuracy")
    
    if best_accuracy > result.overall_accuracy + 15:
        insights.append({
            'type': 'info',
            'message': f"Sinais {best_signal} têm melhor performance ({best_accuracy:.1f}%). Considere focar neste tipo."
        })
    
    # Insight sobre quantidade de sinais
    if result.total_signals < 10:
        insights.append({
            'type': 'warning',
            'message': "Poucos sinais gerados. Considere aumentar o período ou diminuir o intervalo entre análises."
        })
    
    return insights


def generate_executive_summary(result: BacktestResult) -> str:
    """Gera resumo executivo formatado."""
    return f"""
BACKTEST DO PROFESSIONALANALYZER - {result.ticker}
{'='*50}

PERÍODO: {result.start_date.strftime('%d/%m/%Y')} a {result.end_date.strftime('%d/%m/%Y')}
HORIZONTE: {result.evaluation_days} dias úteis
THRESHOLD: ±3.0%

RESULTADOS PRINCIPAIS:
• Total de sinais: {result.total_signals}
• Acurácia geral: {result.overall_accuracy:.1f}%
• CALL: {result.call_accuracy:.1f}% ({result.call_signals} sinais)
• PUT: {result.put_accuracy:.1f}% ({result.put_signals} sinais)
• NEUTRAL: {result.neutral_accuracy:.1f}% ({result.neutral_signals} sinais)

PERFORMANCE POR CONFIANÇA:
• Alta confiança (≥70%): {result.high_confidence_accuracy:.1f}%
• Média confiança (50-69%): {result.medium_confidence_accuracy:.1f}%
• Baixa confiança (<50%): {result.low_confidence_accuracy:.1f}%

CONCLUSÃO:
{'Modelo apresenta performance excelente' if result.overall_accuracy >= 60 
 else 'Modelo apresenta performance satisfatória' if result.overall_accuracy >= 50 
 else 'Modelo necessita ajustes para melhor performance'}

Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M')}
    """.strip()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Backtest ProfessionalAnalyzer",
        page_icon="🔬",
        layout="wide"
    )
    
    render_backtest_page()
