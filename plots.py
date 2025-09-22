"""
Módulo de visualização para resultados de backtest.
Gera gráficos interativos com Plotly para análise dos sinais.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from backtest import BacktestResult, BacktestSignal, Direction


class BacktestPlotter:
    """Classe para gerar visualizações dos resultados de backtest."""
    
    def __init__(self):
        # Cores padrão para os sinais
        self.signal_colors = {
            Direction.CALL: '#00C851',  # Verde
            Direction.PUT: '#FF4444',   # Vermelho
            Direction.NEUTRAL: '#9E9E9E'  # Cinza
        }
        
        # Símbolos para os marcadores
        self.signal_symbols = {
            Direction.CALL: 'triangle-up',
            Direction.PUT: 'triangle-down',
            Direction.NEUTRAL: 'circle'
        }
    
    def plot_price_with_signals(
        self, 
        result: BacktestResult,
        price_data: Optional[pd.DataFrame] = None,
        show_confidence: bool = True,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Plota preço do ativo com marcadores dos sinais.
        
        Args:
            result: Resultado do backtest
            price_data: DataFrame com dados de preço (opcional, será simulado se não fornecido)
            show_confidence: Se deve mostrar nível de confiança na cor/tamanho
            title: Título customizado do gráfico
            
        Returns:
            Figura Plotly interativa
        """
        # Simula dados de preço se não fornecidos
        if price_data is None:
            price_data = self._simulate_price_data(result)
        
        fig = go.Figure()
        
        # Adiciona linha de preço
        fig.add_trace(go.Scatter(
            x=price_data['date'],
            y=price_data['close'],
            mode='lines',
            name='Preço',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>%{y:.2f}</b><br>%{x}<extra></extra>'
        ))
        
        # Adiciona marcadores para cada tipo de sinal
        for signal_type in [Direction.CALL, Direction.PUT, Direction.NEUTRAL]:
            signals_of_type = [s for s in result.signals if s.signal == signal_type]
            
            if not signals_of_type:
                continue
            
            # Prepara dados dos sinais
            dates = [s.date for s in signals_of_type]
            prices = [s.price_at_signal for s in signals_of_type]
            confidences = [s.confidence for s in signals_of_type]
            results = ['✅' if s.is_correct else '❌' for s in signals_of_type]
            returns = [s.return_pct for s in signals_of_type]
            
            # Tamanho baseado na confiança se habilitado
            if show_confidence:
                sizes = [max(8, min(20, conf / 5)) for conf in confidences]
                opacity = [max(0.6, min(1.0, conf / 100)) for conf in confidences]
            else:
                sizes = [12] * len(signals_of_type)
                opacity = [0.8] * len(signals_of_type)
            
            # Hover customizado
            hover_text = [
                f"<b>{signal_type.value}</b><br>" +
                f"Data: {date.strftime('%d/%m/%Y')}<br>" +
                f"Preço: R$ {price:.2f}<br>" +
                f"Confiança: {conf:.0f}%<br>" +
                f"Retorno: {ret:+.1f}%<br>" +
                f"Resultado: {result}"
                for date, price, conf, ret, result in zip(dates, prices, confidences, returns, results)
            ]
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='markers',
                name=f'{signal_type.value} ({len(signals_of_type)})',
                marker=dict(
                    color=self.signal_colors[signal_type],
                    size=sizes,
                    symbol=self.signal_symbols[signal_type],
                    opacity=opacity,
                    line=dict(width=1, color='white')
                ),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_text
            ))
        
        # Layout
        title_text = title or f'Backtest {result.ticker} - Sinais vs Preço'
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title='Data',
            yaxis_title='Preço (R$)',
            hovermode='closest',
            showlegend=True,
            template='plotly_white',
            height=600,
            margin=dict(t=60, b=40, l=60, r=40)
        )
        
        # Adiciona anotação com estatísticas
        stats_text = (
            f"Total: {result.total_signals} sinais | "
            f"Acurácia: {result.overall_accuracy:.1f}% | "
            f"Período: {result.evaluation_days} dias"
        )
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.1,
            showarrow=False,
            font=dict(size=11, color="gray"),
            xanchor="center"
        )
        
        return fig
    
    def plot_accuracy_breakdown(self, result: BacktestResult) -> go.Figure:
        """Plota breakdown de acurácia por tipo de sinal e confiança."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Acurácia por Sinal', 'Acurácia por Confiança'],
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Gráfico 1: Acurácia por tipo de sinal
        signal_types = ['CALL', 'PUT', 'NEUTRAL']
        accuracies = [result.call_accuracy, result.put_accuracy, result.neutral_accuracy]
        counts = [result.call_signals, result.put_signals, result.neutral_signals]
        
        colors_list = ['#00C851', '#FF4444', '#9E9E9E']
        
        fig.add_trace(
            go.Bar(
                x=signal_types,
                y=accuracies,
                text=[f'{acc:.1f}%<br>({count} sinais)' for acc, count in zip(accuracies, counts)],
                textposition='auto',
                marker_color=colors_list,
                name='Acurácia por Sinal',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Gráfico 2: Acurácia por nível de confiança
        confidence_levels = ['Alta (≥70%)', 'Média (50-69%)', 'Baixa (<50%)']
        conf_accuracies = [
            result.high_confidence_accuracy,
            result.medium_confidence_accuracy,
            result.low_confidence_accuracy
        ]
        
        # Conta sinais por nível de confiança
        high_count = len([s for s in result.signals if s.confidence >= 70])
        medium_count = len([s for s in result.signals if 50 <= s.confidence < 70])
        low_count = len([s for s in result.signals if s.confidence < 50])
        conf_counts = [high_count, medium_count, low_count]
        
        fig.add_trace(
            go.Bar(
                x=confidence_levels,
                y=conf_accuracies,
                text=[f'{acc:.1f}%<br>({count} sinais)' for acc, count in zip(conf_accuracies, conf_counts)],
                textposition='auto',
                marker_color=['#4CAF50', '#FF9800', '#F44336'],
                name='Acurácia por Confiança',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Layout
        fig.update_layout(
            title=dict(
                text=f'Análise de Acurácia - {result.ticker}',
                x=0.5,
                font=dict(size=16)
            ),
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        # Atualiza eixos Y para mostrar percentual
        fig.update_yaxes(title_text="Acurácia (%)", range=[0, 100])
        
        # Adiciona linha de referência em 50%
        for col in [1, 2]:
            fig.add_hline(
                y=50, line_dash="dash", line_color="gray", 
                annotation_text="50% (aleatório)", 
                row=1, col=col
            )
        
        return fig
    
    def plot_returns_distribution(self, result: BacktestResult) -> go.Figure:
        """Plota distribuição dos retornos por tipo de sinal."""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['CALL', 'PUT', 'NEUTRAL'],
            specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # Separa retornos por tipo de sinal
        call_returns = [s.return_pct for s in result.signals if s.signal == Direction.CALL]
        put_returns = [s.return_pct for s in result.signals if s.signal == Direction.PUT]
        neutral_returns = [s.return_pct for s in result.signals if s.signal == Direction.NEUTRAL]
        
        returns_data = [
            (call_returns, '#00C851', 1),
            (put_returns, '#FF4444', 2),
            (neutral_returns, '#9E9E9E', 3)
        ]
        
        for returns, color, col in returns_data:
            if not returns:
                continue
                
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=15,
                    marker_color=color,
                    opacity=0.7,
                    name=f'Retornos',
                    showlegend=False
                ),
                row=1, col=col
            )
            
            # Adiciona linha vertical no threshold
            if col == 1:  # CALL
                threshold = 3.0
            elif col == 2:  # PUT
                threshold = -3.0
            else:  # NEUTRAL
                threshold = 0.0
            
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold:+.1f}%",
                row=1, col=col
            )
        
        # Layout
        fig.update_layout(
            title=dict(
                text=f'Distribuição de Retornos - {result.ticker}',
                x=0.5,
                font=dict(size=16)
            ),
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        # Atualiza eixos
        fig.update_xaxes(title_text="Retorno (%)")
        fig.update_yaxes(title_text="Frequência")
        
        return fig
    
    def plot_confidence_vs_accuracy(self, result: BacktestResult) -> go.Figure:
        """Scatter plot: Confiança vs Acerto (para identificar padrões)."""
        fig = go.Figure()
        
        # Separa por tipo de sinal
        for signal_type in [Direction.CALL, Direction.PUT, Direction.NEUTRAL]:
            signals_of_type = [s for s in result.signals if s.signal == signal_type]
            
            if not signals_of_type:
                continue
            
            confidences = [s.confidence for s in signals_of_type]
            accuracies = [1 if s.is_correct else 0 for s in signals_of_type]
            returns = [s.return_pct for s in signals_of_type]
            dates = [s.date.strftime('%d/%m/%Y') for s in signals_of_type]
            
            # Jitter no eixo Y para melhor visualização
            jittered_accuracies = [acc + np.random.uniform(-0.05, 0.05) for acc in accuracies]
            
            hover_text = [
                f"<b>{signal_type.value}</b><br>" +
                f"Data: {date}<br>" +
                f"Confiança: {conf:.0f}%<br>" +
                f"Retorno: {ret:+.1f}%<br>" +
                f"Resultado: {'✅ Acerto' if acc == 1 else '❌ Erro'}"
                for conf, acc, ret, date in zip(confidences, accuracies, returns, dates)
            ]
            
            fig.add_trace(go.Scatter(
                x=confidences,
                y=jittered_accuracies,
                mode='markers',
                name=signal_type.value,
                marker=dict(
                    color=self.signal_colors[signal_type],
                    size=8,
                    symbol=self.signal_symbols[signal_type],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_text
            ))
        
        # Layout
        fig.update_layout(
            title=dict(
                text=f'Confiança vs Resultado - {result.ticker}',
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title='Confiança (%)',
            yaxis_title='Resultado',
            yaxis=dict(
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['❌ Erro', '✅ Acerto']
            ),
            template='plotly_white',
            height=500,
            hovermode='closest'
        )
        
        # Adiciona linha de tendência se houver dados suficientes
        if len(result.signals) >= 10:
            all_confidences = [s.confidence for s in result.signals]
            all_accuracies = [1 if s.is_correct else 0 for s in result.signals]
            
            # Regressão linear simples
            z = np.polyfit(all_confidences, all_accuracies, 1)
            p = np.poly1d(z)
            
            x_trend = np.linspace(min(all_confidences), max(all_confidences), 100)
            y_trend = p(x_trend)
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                name='Tendência',
                line=dict(color='black', dash='dash', width=2),
                showlegend=True
            ))
        
        return fig
    
    def plot_timeline_performance(self, result: BacktestResult) -> go.Figure:
        """Plota performance ao longo do tempo (rolling accuracy)."""
        if len(result.signals) < 10:
            # Gráfico simples se poucos sinais
            return self._plot_simple_timeline(result)
        
        # Calcula acurácia móvel
        window = min(10, len(result.signals) // 3)
        rolling_data = self._calculate_rolling_accuracy(result.signals, window)
        
        fig = go.Figure()
        
        # Linha de acurácia móvel
        fig.add_trace(go.Scatter(
            x=rolling_data['dates'],
            y=rolling_data['accuracy'],
            mode='lines+markers',
            name=f'Acurácia Móvel ({window} sinais)',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{y:.1f}%</b><br>%{x}<extra></extra>'
        ))
        
        # Adiciona marcadores de sinais individuais
        signal_dates = [s.date for s in result.signals]
        signal_results = [100 if s.is_correct else 0 for s in result.signals]
        signal_types = [s.signal.value for s in result.signals]
        
        colors = [self.signal_colors[s.signal] for s in result.signals]
        
        fig.add_trace(go.Scatter(
            x=signal_dates,
            y=signal_results,
            mode='markers',
            name='Sinais Individuais',
            marker=dict(
                color=colors,
                size=8,
                opacity=0.6,
                line=dict(width=1, color='white')
            ),
            text=signal_types,
            hovertemplate='<b>%{text}</b><br>%{x}<br>%{y}<extra></extra>'
        ))
        
        # Layout
        fig.update_layout(
            title=dict(
                text=f'Performance ao Longo do Tempo - {result.ticker}',
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title='Data',
            yaxis_title='Acurácia (%)',
            yaxis=dict(range=[0, 100]),
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        # Linha de referência em 50%
        fig.add_hline(
            y=50, line_dash="dash", line_color="gray",
            annotation_text="50% (aleatório)"
        )
        
        return fig
    
    def _simulate_price_data(self, result: BacktestResult) -> pd.DataFrame:
        """Simula dados de preço para o período do backtest."""
        if not result.signals:
            return pd.DataFrame()
        
        # Usa datas dos sinais como referência
        start_date = min(s.date for s in result.signals) - timedelta(days=30)
        end_date = max(s.date for s in result.signals) + timedelta(days=30)
        
        # Gera datas
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]  # Remove fins de semana
        
        # Simula preços baseados nos preços dos sinais
        signal_prices = {s.date: s.price_at_signal for s in result.signals}
        
        prices = []
        for date in dates:
            # Encontra sinal mais próximo
            closest_signal_date = min(signal_prices.keys(), 
                                    key=lambda x: abs((x - date).days))
            base_price = signal_prices[closest_signal_date]
            
            # Adiciona ruído baseado na distância temporal
            days_diff = abs((date - closest_signal_date).days)
            noise_factor = min(0.1, days_diff * 0.005)  # Max 10% de ruído
            noise = np.random.normal(0, noise_factor)
            
            price = base_price * (1 + noise)
            prices.append(price)
        
        return pd.DataFrame({
            'date': dates,
            'close': prices
        })
    
    def _plot_simple_timeline(self, result: BacktestResult) -> go.Figure:
        """Gráfico simples para poucos sinais."""
        fig = go.Figure()
        
        dates = [s.date for s in result.signals]
        results = [100 if s.is_correct else 0 for s in result.signals]
        types = [s.signal.value for s in result.signals]
        colors = [self.signal_colors[s.signal] for s in result.signals]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=results,
            mode='markers+lines',
            name='Resultados',
            marker=dict(
                color=colors,
                size=12,
                line=dict(width=2, color='white')
            ),
            line=dict(color='gray', width=1, dash='dash'),
            text=types,
            hovertemplate='<b>%{text}</b><br>%{x}<br>%{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Timeline de Resultados - {result.ticker}',
            xaxis_title='Data',
            yaxis_title='Resultado',
            yaxis=dict(
                tickmode='array',
                tickvals=[0, 100],
                ticktext=['❌ Erro', '✅ Acerto'],
                range=[-10, 110]
            ),
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def _calculate_rolling_accuracy(
        self, 
        signals: List[BacktestSignal], 
        window: int
    ) -> Dict:
        """Calcula acurácia móvel."""
        if len(signals) < window:
            return {'dates': [], 'accuracy': []}
        
        # Ordena por data
        sorted_signals = sorted(signals, key=lambda s: s.date)
        
        dates = []
        accuracies = []
        
        for i in range(window - 1, len(sorted_signals)):
            window_signals = sorted_signals[i - window + 1:i + 1]
            accuracy = (sum(1 for s in window_signals if s.is_correct) / len(window_signals)) * 100
            
            dates.append(sorted_signals[i].date)
            accuracies.append(accuracy)
        
        return {'dates': dates, 'accuracy': accuracies}


def create_all_plots(result: BacktestResult) -> Dict[str, go.Figure]:
    """Cria todos os gráficos de uma vez."""
    plotter = BacktestPlotter()
    
    plots = {
        'price_signals': plotter.plot_price_with_signals(result),
        'accuracy_breakdown': plotter.plot_accuracy_breakdown(result),
        'returns_distribution': plotter.plot_returns_distribution(result),
        'confidence_accuracy': plotter.plot_confidence_vs_accuracy(result),
        'timeline_performance': plotter.plot_timeline_performance(result)
    }
    
    return plots
