"""
Módulo de backtest para validar o ProfessionalAnalyzer.
Testa sinais CALL/PUT/NEUTRAL em dados históricos e mede acurácia.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from src.core.professional.professional_analysis import ProfessionalAnalyzer, Direction, OptionsSentimentAnalysis
from src.core.data.data import get_price_history, get_historical_data


class BacktestProfessionalAnalyzer(ProfessionalAnalyzer):
    """
    Versão do ProfessionalAnalyzer para backtest que funciona APENAS com dados REAIS.
    Não usa OpLab - apenas dados históricos reais do Yahoo Finance.
    """
    
    def __init__(self, horizon: str = "médio"):
        """
        Inicializa analyzer para backtest sem OpLab.
        Usa apenas dados históricos REAIS do Yahoo Finance.
        """
        # Não chama super().__init__ para evitar criar OpLabClient
        self.client = None  # Sem OpLab
        self.horizon = horizon
        self.params = self._get_horizon_parameters(horizon)
        self.decision_threshold = 0.30
        
        # Pesos configuráveis das camadas (mesmo do ProfessionalAnalyzer original)
        self.layer_weights = {
            'trend': 0.30,      # 30% - tendência é fundamental
            'momentum': 0.25,   # 25% - momentum confirma tendência
            'volume': 0.20,     # 20% - volume confirma movimento
            'sentiment': 0.15,  # 15% - sentimento do mercado
            'macro': 0.10       # 10% - contexto macro
        }
    
    def analyze_options_sentiment(self, ticker: str, price_data: pd.DataFrame) -> OptionsSentimentAnalysis:
        """
        Para backtest, retorna sentimento neutro pois não temos dados de opções reais históricos.
        Isso é correto - no backtest histórico não tínhamos acesso aos dados de opções do passado.
        """
        return OptionsSentimentAnalysis(
            put_call_ratio=1.0,
            volatility_skew=0.0,
            call_volume_ratio=0.5,
            put_volume_ratio=0.5,
            sentiment_score=0.0,
            market_bias="NEUTRAL"
        )


@dataclass
class BacktestSignal:
    """Representa um sinal gerado durante o backtest."""
    date: datetime
    ticker: str
    signal: Direction
    confidence: float
    price_at_signal: float
    price_after_horizon: float
    return_pct: float
    is_correct: bool
    drivers: List[str]


@dataclass
class BenchmarkMetrics:
    """Métricas de benchmark para comparação."""
    buy_hold_return: float
    strategy_return: float
    random_accuracy: float
    strategy_max_drawdown: float
    buy_hold_max_drawdown: float
    strategy_sharpe: float
    buy_hold_sharpe: float
    strategy_wins: int
    strategy_losses: int
    avg_win_pct: float
    avg_loss_pct: float


@dataclass
class BacktestResult:
    """Resultado completo do backtest."""
    ticker: str
    start_date: datetime
    end_date: datetime
    evaluation_days: int
    total_signals: int
    signals: List[BacktestSignal]
    
    # Estatísticas gerais
    overall_accuracy: float
    call_accuracy: float
    put_accuracy: float
    neutral_accuracy: float
    
    # Contadores
    call_signals: int
    put_signals: int
    neutral_signals: int
    
    # Performance por confiança
    high_confidence_accuracy: float  # >= 70%
    medium_confidence_accuracy: float  # 50-69%
    low_confidence_accuracy: float  # < 50%
    
    # Métricas avançadas
    benchmark_metrics: Optional[BenchmarkMetrics] = None


class BacktestEngine:
    """Engine principal para execução de backtests."""
    
    def __init__(self, success_threshold: float = 3.0, analyzer=None, horizon: str = "médio"):
        """
        Args:
            success_threshold: Percentual mínimo para considerar acerto (ex: 3.0 = 3%)
            analyzer: ProfessionalAnalyzer instance (opcional, para testes)
            horizon: Horizonte temporal ("curto", "médio", "longo")
        """
        self.success_threshold = success_threshold
        self.horizon = horizon
        # Usa BacktestProfessionalAnalyzer que funciona APENAS com dados REAIS
        self.analyzer = analyzer if analyzer is not None else BacktestProfessionalAnalyzer(horizon=horizon)
    
    def run_backtest(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        evaluation_days: int = 20,
        rolling_window: int = 5,
        min_history_days: int = 200
    ) -> BacktestResult:
        """
        Executa backtest completo.
        
        Args:
            ticker: Código do ativo (ex: PETR4)
            start_date: Data inicial do backtest
            end_date: Data final do backtest
            evaluation_days: Dias úteis para avaliar se sinal acertou
            rolling_window: Intervalo entre análises (dias úteis)
            min_history_days: Mínimo de histórico para primeira análise
            
        Returns:
            BacktestResult com todos os sinais e estatísticas
        """
        print(f"🚀 Iniciando backtest para {ticker}")
        print(f"📅 Período: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
        print(f"⏱️ Avaliação: {evaluation_days} dias úteis")
        
        signals = []
        
        # Gera datas de análise (rolling window)
        analysis_dates = self._generate_analysis_dates(
            start_date, end_date, rolling_window
        )
        
        print(f"📊 Total de análises planejadas: {len(analysis_dates)}")
        
        for i, analysis_date in enumerate(analysis_dates):
            try:
                # Busca histórico até a data de análise
                historical_data = self._get_historical_data_until_date(
                    ticker, analysis_date, min_history_days
                )
                
                if historical_data.empty:
                    print(f"⚠️ Sem dados históricos para {analysis_date.strftime('%d/%m/%Y')}")
                    continue
                
                # Executa análise profissional
                if self.analyzer is None:
                    print(f"⚠️ Analyzer é None - pulando análise para {analysis_date.strftime('%d/%m/%Y')}")
                    continue
                    
                print(f"🔍 Executando análise para {analysis_date.strftime('%d/%m/%Y')} com {len(historical_data)} dias de histórico")
                analysis = self.analyzer.analyze(ticker, historical_data)
                print(f"✅ Análise executada: {analysis.direction.value} ({analysis.confidence:.0f}%)")
                
                # Usa horizonte dinâmico se disponível
                dynamic_evaluation_days = evaluation_days
                if analysis.dynamic_horizon:
                    dynamic_evaluation_days = analysis.dynamic_horizon.evaluation_days
                    print(f"🎯 Horizonte dinâmico: {dynamic_evaluation_days} dias (regime: {analysis.dynamic_horizon.regime})")
                
                # Busca preços para validação (entrada e saída)
                entry_price, exit_price = self._get_entry_and_exit_prices(
                    ticker, analysis_date, dynamic_evaluation_days
                )
                
                if entry_price is None or exit_price is None:
                    print(f"⚠️ Sem preços de entrada/saída para {analysis_date.strftime('%d/%m/%Y')}")
                    continue
                
                # Cria sinal usando preço de entrada correto
                current_price = historical_data['close'].iloc[-1]  # Preço no momento da decisão
                return_pct = ((exit_price - entry_price) / entry_price) * 100
                
                is_correct = self._evaluate_signal_accuracy(
                    analysis.direction, return_pct
                )
                
                signal = BacktestSignal(
                    date=analysis_date,
                    ticker=ticker,
                    signal=analysis.direction,
                    confidence=analysis.confidence,
                    price_at_signal=entry_price,  # Preço de entrada real (abertura do dia seguinte)
                    price_after_horizon=exit_price,  # Preço de saída
                    return_pct=return_pct,
                    is_correct=is_correct,
                    drivers=analysis.key_drivers
                )
                
                signals.append(signal)
                
                if (i + 1) % 10 == 0:
                    print(f"✅ Processadas {i + 1}/{len(analysis_dates)} análises")
                    
            except Exception as e:
                print(f"❌ Erro na análise de {analysis_date.strftime('%d/%m/%Y')}: {e}")
                continue
        
        # Calcula estatísticas finais
        result = self._calculate_statistics(
            ticker, start_date, end_date, evaluation_days, signals
        )
        
        # Calcula métricas de benchmark
        if signals:
            result.benchmark_metrics = self._calculate_benchmark_metrics(
                ticker, start_date, end_date, signals
            )
        
        print(f"🎯 Backtest concluído: {result.total_signals} sinais gerados")
        print(f"📈 Acurácia geral: {result.overall_accuracy:.1f}%")
        
        return result
    
    def _generate_analysis_dates(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        rolling_window: int
    ) -> List[datetime]:
        """Gera datas de análise com rolling window."""
        dates = []
        current_date = start_date
        
        while current_date <= end_date:
            # Pula fins de semana
            if current_date.weekday() < 5:
                dates.append(current_date)
            
            current_date += timedelta(days=rolling_window)
        
        return dates
    
    def _get_historical_data_until_date(
        self, 
        ticker: str, 
        until_date: datetime, 
        min_days: int
    ) -> pd.DataFrame:
        """
        Busca dados históricos REAIS até uma data específica via Yahoo Finance.
        NUNCA usa dados simulados - apenas dados reais.
        """
        try:
            # Calcula período necessário (adiciona buffer para fins de semana)
            extended_days = max(min_days + 100, 400)  # Garante pelo menos 400 dias
            start_date = until_date - timedelta(days=int(extended_days * 1.4))
            
            # Formata datas para Yahoo Finance
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = until_date.strftime('%Y-%m-%d')
            
            print(f"📊 Buscando dados REAIS para {ticker}: {start_str} a {end_str}")
            
            # Busca dados REAIS via Yahoo Finance
            df = get_historical_data(ticker, start_str, end_str)
            
            if df.empty:
                print(f"⚠️ Nenhum dado REAL encontrado para {ticker} no período")
                return pd.DataFrame()
            
            # Filtra até a data de análise (garante que não usa dados futuros)
            # Converte until_date para o mesmo timezone dos dados
            if not df.empty and df['date'].dt.tz is not None:
                until_date_tz = until_date.replace(tzinfo=df['date'].dt.tz)
                df = df[df['date'] <= until_date_tz]
            else:
                df = df[df['date'] <= until_date]
            
            # Garante mínimo de dados necessários
            if len(df) >= min_days:
                df = df.tail(min_days).reset_index(drop=True)
                print(f"✅ {len(df)} candles REAIS carregados para {ticker} até {until_date.strftime('%d/%m/%Y')}")
            else:
                print(f"⚠️ Apenas {len(df)} candles REAIS disponíveis para {ticker} (mínimo: {min_days})")
            
            return df
            
        except Exception as e:
            print(f"❌ Erro ao buscar dados REAIS para {ticker}: {e}")
            return pd.DataFrame()
    
    def _get_entry_and_exit_prices(
        self, 
        ticker: str, 
        analysis_date: datetime, 
        evaluation_days: int
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Busca preços REAIS de entrada e saída via Yahoo Finance.
        
        Entrada: Preço de ABERTURA do dia seguinte ao sinal (evita look-ahead bias)
        Saída: Preço de FECHAMENTO após evaluation_days
        
        NUNCA usa dados simulados - apenas dados reais.
        """
        try:
            # Calcula data futura (adiciona buffer para fins de semana)
            future_date = analysis_date + timedelta(days=int(evaluation_days * 1.4))
            
            # Formata datas para Yahoo Finance
            start_str = analysis_date.strftime('%Y-%m-%d')
            end_str = future_date.strftime('%Y-%m-%d')
            
            print(f"🔮 Buscando preços REAIS de entrada/saída para {ticker}: {start_str} a {end_str}")
            
            # Busca dados REAIS do período futuro via Yahoo Finance
            df = get_historical_data(ticker, start_str, end_str)
            
            if df.empty:
                print(f"⚠️ Sem dados REAIS futuros para {ticker} em {end_str}")
                return None, None
            
            # Remove o primeiro dia (data da análise) para evitar look-ahead bias
            # Converte analysis_date para o mesmo timezone dos dados
            if not df.empty and df['date'].dt.tz is not None:
                analysis_date_tz = analysis_date.replace(tzinfo=df['date'].dt.tz)
                df = df[df['date'] > analysis_date_tz]
            else:
                df = df[df['date'] > analysis_date]
            
            if df.empty:
                print(f"⚠️ Sem dados após filtrar data da análise para {ticker}")
                return None, None
            
            # PREÇO DE ENTRADA: Abertura do primeiro dia após o sinal
            entry_price = float(df.iloc[0]['open'])  # Abertura do dia seguinte
            
            # PREÇO DE SAÍDA: Fechamento após evaluation_days
            if len(df) < evaluation_days:
                print(f"⚠️ Apenas {len(df)} dias REAIS futuros disponíveis para {ticker} (esperado: {evaluation_days})")
            
            # Pega o preço de saída após evaluation_days (ou o último disponível)
            target_idx = min(evaluation_days - 1, len(df) - 1)
            exit_price = float(df.iloc[target_idx]['close'])
            
            print(f"✅ Entrada: {entry_price:.2f} (abertura D+1) | Saída: {exit_price:.2f} (fechamento D+{target_idx + 1})")
            
            return entry_price, exit_price
            
        except Exception as e:
            print(f"❌ Erro ao buscar preços REAIS para {ticker}: {e}")
            return None, None
    
    def _evaluate_signal_accuracy(
        self, 
        signal: Direction, 
        return_pct: float
    ) -> bool:
        """Avalia se o sinal acertou baseado no threshold."""
        if signal == Direction.CALL:
            return return_pct > self.success_threshold
        elif signal == Direction.PUT:
            return return_pct < -self.success_threshold
        else:  # NEUTRAL
            return abs(return_pct) <= self.success_threshold
    
    def _calculate_statistics(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        evaluation_days: int,
        signals: List[BacktestSignal]
    ) -> BacktestResult:
        """Calcula todas as estatísticas do backtest."""
        if not signals:
            return BacktestResult(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                evaluation_days=evaluation_days,
                total_signals=0,
                signals=[],
                overall_accuracy=0.0,
                call_accuracy=0.0,
                put_accuracy=0.0,
                neutral_accuracy=0.0,
                call_signals=0,
                put_signals=0,
                neutral_signals=0,
                high_confidence_accuracy=0.0,
                medium_confidence_accuracy=0.0,
                low_confidence_accuracy=0.0
            )
        
        # Separa sinais por tipo
        call_signals = [s for s in signals if s.signal == Direction.CALL]
        put_signals = [s for s in signals if s.signal == Direction.PUT]
        neutral_signals = [s for s in signals if s.signal == Direction.NEUTRAL]
        
        # Calcula acurácias por tipo
        overall_accuracy = (sum(1 for s in signals if s.is_correct) / len(signals)) * 100
        
        call_accuracy = (sum(1 for s in call_signals if s.is_correct) / len(call_signals) * 100) if call_signals else 0
        put_accuracy = (sum(1 for s in put_signals if s.is_correct) / len(put_signals) * 100) if put_signals else 0
        neutral_accuracy = (sum(1 for s in neutral_signals if s.is_correct) / len(neutral_signals) * 100) if neutral_signals else 0
        
        # Acurácia por nível de confiança
        high_conf = [s for s in signals if s.confidence >= 70]
        medium_conf = [s for s in signals if 50 <= s.confidence < 70]
        low_conf = [s for s in signals if s.confidence < 50]
        
        high_confidence_accuracy = (sum(1 for s in high_conf if s.is_correct) / len(high_conf) * 100) if high_conf else 0
        medium_confidence_accuracy = (sum(1 for s in medium_conf if s.is_correct) / len(medium_conf) * 100) if medium_conf else 0
        low_confidence_accuracy = (sum(1 for s in low_conf if s.is_correct) / len(low_conf) * 100) if low_conf else 0
        
        return BacktestResult(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            evaluation_days=evaluation_days,
            total_signals=len(signals),
            signals=signals,
            overall_accuracy=overall_accuracy,
            call_accuracy=call_accuracy,
            put_accuracy=put_accuracy,
            neutral_accuracy=neutral_accuracy,
            call_signals=len(call_signals),
            put_signals=len(put_signals),
            neutral_signals=len(neutral_signals),
            high_confidence_accuracy=high_confidence_accuracy,
            medium_confidence_accuracy=medium_confidence_accuracy,
            low_confidence_accuracy=low_confidence_accuracy
        )
    
    def _calculate_benchmark_metrics(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        signals: List[BacktestSignal]
    ) -> BenchmarkMetrics:
        """Calcula métricas de benchmark para comparação."""
        try:
            # 1. Buy & Hold benchmark
            buy_hold_return = self._calculate_buy_hold_return(ticker, start_date, end_date)
            
            # 2. Retorno da estratégia
            strategy_return = self._calculate_strategy_return(signals)
            
            # 3. Estratégia aleatória (baseline)
            random_accuracy = 50.0  # Estratégia aleatória teria ~50% de acerto
            
            # 4. Max Drawdown
            strategy_max_drawdown = self._calculate_max_drawdown(signals)
            buy_hold_max_drawdown = self._calculate_buy_hold_max_drawdown(ticker, start_date, end_date)
            
            # 5. Sharpe Ratio
            strategy_sharpe = self._calculate_strategy_sharpe(signals)
            buy_hold_sharpe = self._calculate_buy_hold_sharpe(ticker, start_date, end_date)
            
            # 6. Estatísticas de win/loss
            wins = [s for s in signals if s.return_pct > 0]
            losses = [s for s in signals if s.return_pct < 0]
            
            avg_win_pct = np.mean([s.return_pct for s in wins]) if wins else 0.0
            avg_loss_pct = np.mean([s.return_pct for s in losses]) if losses else 0.0
            
            return BenchmarkMetrics(
                buy_hold_return=buy_hold_return,
                strategy_return=strategy_return,
                random_accuracy=random_accuracy,
                strategy_max_drawdown=strategy_max_drawdown,
                buy_hold_max_drawdown=buy_hold_max_drawdown,
                strategy_sharpe=strategy_sharpe,
                buy_hold_sharpe=buy_hold_sharpe,
                strategy_wins=len(wins),
                strategy_losses=len(losses),
                avg_win_pct=avg_win_pct,
                avg_loss_pct=avg_loss_pct
            )
            
        except Exception as e:
            print(f"⚠️ Erro ao calcular métricas de benchmark: {e}")
            return BenchmarkMetrics(0, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_buy_hold_return(self, ticker: str, start_date: datetime, end_date: datetime) -> float:
        """Calcula retorno Buy & Hold do período."""
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            df = get_historical_data(ticker, start_str, end_str)
            if df.empty or len(df) < 2:
                return 0.0
            
            initial_price = df.iloc[0]['close']
            final_price = df.iloc[-1]['close']
            
            return ((final_price - initial_price) / initial_price) * 100
            
        except Exception as e:
            print(f"⚠️ Erro ao calcular Buy & Hold: {e}")
            return 0.0
    
    def _calculate_strategy_return(self, signals: List[BacktestSignal]) -> float:
        """Calcula retorno acumulado da estratégia."""
        if not signals:
            return 0.0
        
        # Retorno médio por operação
        returns = [s.return_pct / 100 for s in signals]  # Converte para decimal
        
        # Retorno acumulado (composto)
        cumulative_return = 1.0
        for ret in returns:
            cumulative_return *= (1 + ret)
        
        return (cumulative_return - 1) * 100  # Converte de volta para percentual
    
    def _calculate_max_drawdown(self, signals: List[BacktestSignal]) -> float:
        """Calcula maximum drawdown da estratégia."""
        if not signals:
            return 0.0
        
        returns = [s.return_pct / 100 for s in signals]
        cumulative = np.cumprod([1 + r for r in returns])
        
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        
        return float(drawdown.min() * 100)  # Converte para percentual
    
    def _calculate_buy_hold_max_drawdown(self, ticker: str, start_date: datetime, end_date: datetime) -> float:
        """Calcula maximum drawdown do Buy & Hold."""
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            df = get_historical_data(ticker, start_str, end_str)
            if df.empty:
                return 0.0
            
            prices = df['close'].values
            peak = np.maximum.accumulate(prices)
            drawdown = (prices - peak) / peak
            
            return float(drawdown.min() * 100)
            
        except Exception as e:
            print(f"⚠️ Erro ao calcular Max Drawdown Buy & Hold: {e}")
            return 0.0
    
    def _calculate_strategy_sharpe(self, signals: List[BacktestSignal]) -> float:
        """Calcula Sharpe ratio da estratégia."""
        if not signals:
            return 0.0
        
        returns = [s.return_pct / 100 for s in signals]
        
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Anualiza assumindo ~252 dias úteis / ano
        # Para sinais espaçados, ajusta frequência
        frequency = 252 / len(signals) if len(signals) > 0 else 1
        
        return float((mean_return / std_return) * np.sqrt(frequency))
    
    def _calculate_buy_hold_sharpe(self, ticker: str, start_date: datetime, end_date: datetime) -> float:
        """Calcula Sharpe ratio do Buy & Hold."""
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            df = get_historical_data(ticker, start_str, end_str)
            if df.empty or len(df) < 2:
                return 0.0
            
            # Calcula retornos diários
            prices = df['close'].values
            returns = np.diff(prices) / prices[:-1]
            
            if len(returns) < 2:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            
            if std_return == 0:
                return 0.0
            
            return float((mean_return / std_return) * np.sqrt(252))  # Anualizado
            
        except Exception as e:
            print(f"⚠️ Erro ao calcular Sharpe Buy & Hold: {e}")
            return 0.0
    
    def export_signals_to_dataframe(self, result: BacktestResult) -> pd.DataFrame:
        """Exporta sinais para DataFrame para análise."""
        data = []
        
        for signal in result.signals:
            data.append({
                'Data': signal.date.strftime('%d/%m/%Y'),
                'Sinal': signal.signal.value,
                'Confiança (%)': f"{signal.confidence:.0f}%",
                'Preço Inicial': f"R$ {signal.price_at_signal:.2f}",
                'Preço +{evaluation_days}d': f"R$ {signal.price_after_horizon:.2f}",
                'Retorno (%)': f"{signal.return_pct:+.1f}%",
                'Resultado': "✅ Acerto" if signal.is_correct else "❌ Erro",
                'Drivers': ", ".join(signal.drivers[:2])  # Primeiros 2 drivers
            })
        
        df = pd.DataFrame(data)
        return df
    
    def print_summary(self, result: BacktestResult) -> None:
        """Imprime resumo detalhado do backtest."""
        print("\n" + "="*60)
        print(f"📊 RESUMO DO BACKTEST - {result.ticker}")
        print("="*60)
        print(f"📅 Período: {result.start_date.strftime('%d/%m/%Y')} a {result.end_date.strftime('%d/%m/%Y')}")
        print(f"⏱️ Horizonte de avaliação: {result.evaluation_days} dias úteis")
        print(f"🎯 Threshold de sucesso: ±{self.success_threshold}%")
        print()
        
        print("📈 ESTATÍSTICAS GERAIS:")
        print(f"• Total de sinais: {result.total_signals}")
        print(f"• Acurácia geral: {result.overall_accuracy:.1f}%")
        print()
        
        print("📊 ACURÁCIA POR SINAL:")
        print(f"• CALL ({result.call_signals} sinais): {result.call_accuracy:.1f}%")
        print(f"• PUT ({result.put_signals} sinais): {result.put_accuracy:.1f}%")
        print(f"• NEUTRAL ({result.neutral_signals} sinais): {result.neutral_accuracy:.1f}%")
        print()
        
        print("🎯 ACURÁCIA POR CONFIANÇA:")
        high_count = len([s for s in result.signals if s.confidence >= 70])
        medium_count = len([s for s in result.signals if 50 <= s.confidence < 70])
        low_count = len([s for s in result.signals if s.confidence < 50])
        
        print(f"• Alta confiança (≥70%, {high_count} sinais): {result.high_confidence_accuracy:.1f}%")
        print(f"• Média confiança (50-69%, {medium_count} sinais): {result.medium_confidence_accuracy:.1f}%")
        print(f"• Baixa confiança (<50%, {low_count} sinais): {result.low_confidence_accuracy:.1f}%")
        print()
        
        # Métricas de benchmark
        if result.benchmark_metrics:
            bm = result.benchmark_metrics
            print("📊 COMPARAÇÃO COM BENCHMARKS:")
            print(f"• Estratégia: {bm.strategy_return:+.1f}% | Buy & Hold: {bm.buy_hold_return:+.1f}%")
            print(f"• Max Drawdown Estratégia: {bm.strategy_max_drawdown:.1f}%")
            print(f"• Max Drawdown Buy & Hold: {bm.buy_hold_max_drawdown:.1f}%")
            print(f"• Sharpe Estratégia: {bm.strategy_sharpe:.2f} | Buy & Hold: {bm.buy_hold_sharpe:.2f}")
            print(f"• Win/Loss: {bm.strategy_wins}W / {bm.strategy_losses}L")
            print(f"• Ganho médio: {bm.avg_win_pct:+.1f}% | Perda média: {bm.avg_loss_pct:+.1f}%")
            print()
        
        # Análise de performance
        if result.overall_accuracy >= 60:
            print("🎉 PERFORMANCE: EXCELENTE (≥60%)")
        elif result.overall_accuracy >= 50:
            print("👍 PERFORMANCE: BOA (50-59%)")
        elif result.overall_accuracy >= 40:
            print("⚠️ PERFORMANCE: REGULAR (40-49%)")
        else:
            print("❌ PERFORMANCE: PRECISA MELHORAR (<40%)")
        
        # Comparação com benchmarks
        if result.benchmark_metrics:
            bm = result.benchmark_metrics
            if bm.strategy_return > bm.buy_hold_return:
                print("✅ ESTRATÉGIA SUPERA BUY & HOLD")
            else:
                print("❌ BUY & HOLD SUPERA ESTRATÉGIA")
        
        print("="*60)


if __name__ == "__main__":
    # Exemplo de uso
    engine = BacktestEngine(success_threshold=3.0)
    
    # Backtest dos últimos 2 anos
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    result = engine.run_backtest(
        ticker="PETR4",
        start_date=start_date,
        end_date=end_date,
        evaluation_days=20,
        rolling_window=5
    )
    
    engine.print_summary(result)
    
    # Exporta para DataFrame
    df = engine.export_signals_to_dataframe(result)
    print(f"\n📋 Tabela de sinais exportada com {len(df)} registros")


def run_experiments(
    tickers: List[str] = ["PETR4", "VALE3", "ITUB4", "MGLU3"],
    evaluation_days_list: List[int] = [5, 10, 20],
    threshold_list: List[float] = [0.15, 0.30],
    weights_configs: List[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Executa experimentos variando parâmetros do backtest para comparar performance.
    
    Args:
        tickers: Lista de tickers para testar
        evaluation_days_list: Lista de janelas de avaliação (dias)
        threshold_list: Lista de thresholds de decisão
        weights_configs: Lista de configurações de pesos das camadas
        
    Returns:
        DataFrame com resultados comparativos
    """
    if weights_configs is None:
        weights_configs = [
            # Configuração padrão
            {'trend': 0.30, 'momentum': 0.25, 'volume': 0.20, 'sentiment': 0.15, 'macro': 0.10},
            # Configuração focada em tendência
            {'trend': 0.40, 'momentum': 0.30, 'volume': 0.15, 'sentiment': 0.10, 'macro': 0.05},
            # Configuração balanceada
            {'trend': 0.25, 'momentum': 0.25, 'volume': 0.25, 'sentiment': 0.15, 'macro': 0.10}
        ]
    
    print("🧪 EXECUTANDO EXPERIMENTOS DE BACKTEST")
    print("=" * 50)
    
    results = []
    experiment_id = 0
    
    for ticker in tickers:
        print(f"\n📊 Testando {ticker}...")
        
        for eval_days in evaluation_days_list:
            for threshold in threshold_list:
                for i, weights in enumerate(weights_configs):
                    experiment_id += 1
                    
                    try:
                        # Cria analyzer com configuração específica
                        analyzer = BacktestProfessionalAnalyzer(horizon="médio")
                        analyzer.decision_threshold = threshold
                        analyzer.layer_weights = weights
                        
                        # Configura engine
                        engine = BacktestEngine(
                            success_threshold=3.0,  # 3% fixo
                            analyzer=analyzer,
                            horizon="médio"
                        )
                        
                        # Executa backtest
                        result = engine.run_backtest(
                            ticker=ticker,
                            start_date=datetime(2024, 1, 1),
                            end_date=datetime(2024, 12, 31),
                            evaluation_days=eval_days,
                            rolling_window=10,
                            min_history_days=300
                        )
                        
                        # Adiciona resultado
                        results.append({
                            'experiment_id': experiment_id,
                            'ticker': ticker,
                            'eval_days': eval_days,
                            'threshold': threshold,
                            'weights_config': f"Config_{i+1}",
                            'weights_detail': str(weights),
                            'total_signals': result.total_signals,
                            'overall_accuracy': result.overall_accuracy,
                            'call_accuracy': result.call_accuracy,
                            'put_accuracy': result.put_accuracy,
                            'neutral_accuracy': result.neutral_accuracy,
                            'call_signals': result.call_signals,
                            'put_signals': result.put_signals,
                            'neutral_signals': result.neutral_signals,
                            'high_conf_accuracy': result.high_confidence_accuracy,
                            'medium_conf_accuracy': result.medium_confidence_accuracy,
                            'low_conf_accuracy': result.low_confidence_accuracy
                        })
                        
                        print(f"  ✅ Exp {experiment_id}: {eval_days}d, threshold {threshold}, config {i+1} - Acurácia: {result.overall_accuracy:.1f}%")
                        
                    except Exception as e:
                        print(f"  ❌ Exp {experiment_id}: Erro - {e}")
                        continue
    
    # Converte para DataFrame
    df_results = pd.DataFrame(results)
    
    if not df_results.empty:
        print(f"\n📋 RESUMO DOS EXPERIMENTOS ({len(df_results)} configurações testadas)")
        print("=" * 60)
        
        # Melhores configurações por métrica
        best_overall = df_results.loc[df_results['overall_accuracy'].idxmax()]
        best_call = df_results.loc[df_results['call_accuracy'].idxmax()]
        best_put = df_results.loc[df_results['put_accuracy'].idxmax()]
        
        print(f"🏆 MELHOR ACURÁCIA GERAL: {best_overall['overall_accuracy']:.1f}%")
        print(f"   Ticker: {best_overall['ticker']}, Janela: {best_overall['eval_days']}d, Threshold: {best_overall['threshold']}, {best_overall['weights_config']}")
        
        print(f"📈 MELHOR CALL: {best_call['call_accuracy']:.1f}%")
        print(f"   Ticker: {best_call['ticker']}, Janela: {best_call['eval_days']}d, Threshold: {best_call['threshold']}, {best_call['weights_config']}")
        
        print(f"📉 MELHOR PUT: {best_put['put_accuracy']:.1f}%")
        print(f"   Ticker: {best_put['ticker']}, Janela: {best_put['eval_days']}d, Threshold: {best_put['threshold']}, {best_put['weights_config']}")
        
        # Estatísticas por parâmetro
        print(f"\n📊 MÉDIAS POR THRESHOLD:")
        threshold_stats = df_results.groupby('threshold')['overall_accuracy'].agg(['mean', 'std', 'count'])
        for threshold, stats in threshold_stats.iterrows():
            print(f"   Threshold {threshold}: {stats['mean']:.1f}% ±{stats['std']:.1f}% ({stats['count']} testes)")
        
        print(f"\n📊 MÉDIAS POR JANELA DE AVALIAÇÃO:")
        eval_stats = df_results.groupby('eval_days')['overall_accuracy'].agg(['mean', 'std', 'count'])
        for days, stats in eval_stats.iterrows():
            print(f"   {days} dias: {stats['mean']:.1f}% ±{stats['std']:.1f}% ({stats['count']} testes)")
        
        print(f"\n📊 MÉDIAS POR CONFIGURAÇÃO DE PESOS:")
        weight_stats = df_results.groupby('weights_config')['overall_accuracy'].agg(['mean', 'std', 'count'])
        for config, stats in weight_stats.iterrows():
            print(f"   {config}: {stats['mean']:.1f}% ±{stats['std']:.1f}% ({stats['count']} testes)")
    
    return df_results


# Função auxiliar para análise detalhada
def analyze_experiment_results(df_results: pd.DataFrame) -> None:
    """Analisa resultados dos experimentos em detalhes."""
    if df_results.empty:
        print("❌ Nenhum resultado para analisar")
        return
    
    print("\n🔍 ANÁLISE DETALHADA DOS EXPERIMENTOS")
    print("=" * 50)
    
    # Correlações entre parâmetros e performance
    print("📈 CORRELAÇÃO ENTRE PARÂMETROS E ACURÁCIA:")
    correlations = df_results[['eval_days', 'threshold', 'overall_accuracy', 'call_accuracy', 'put_accuracy']].corr()['overall_accuracy'].sort_values(ascending=False)
    
    for param, corr in correlations.items():
        if param != 'overall_accuracy':
            direction = "📈" if corr > 0 else "📉"
            print(f"   {direction} {param}: {corr:.3f}")
    
    # Distribuição de sinais por configuração
    print(f"\n📊 DISTRIBUIÇÃO DE SINAIS:")
    signal_dist = df_results[['call_signals', 'put_signals', 'neutral_signals']].mean()
    total_avg = signal_dist.sum()
    
    for signal_type, avg_count in signal_dist.items():
        pct = (avg_count / total_avg) * 100
        print(f"   {signal_type}: {avg_count:.1f} ({pct:.1f}%)")
    
    # Top 5 configurações
    print(f"\n🏆 TOP 5 CONFIGURAÇÕES:")
    top_configs = df_results.nlargest(5, 'overall_accuracy')
    
    for i, (_, config) in enumerate(top_configs.iterrows(), 1):
        print(f"   {i}. {config['ticker']} - {config['eval_days']}d - T{config['threshold']} - {config['weights_config']}: {config['overall_accuracy']:.1f}%")


if __name__ == "__main__":
    # Exemplo de uso da função de experimentos
    print("🧪 Executando experimentos...")
    results_df = run_experiments(
        tickers=["PETR4", "ITUB4"], 
        evaluation_days_list=[10, 20],
        threshold_list=[0.15, 0.30]
    )
    
    if not results_df.empty:
        analyze_experiment_results(results_df)
