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

from professional_analysis import ProfessionalAnalyzer, Direction
from data import get_price_history, get_historical_data
from oplab_client import OpLabClient


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
        self.analyzer = analyzer if analyzer is not None else ProfessionalAnalyzer(horizon=horizon)
    
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
                
                # Busca preço futuro para validação
                future_price = self._get_future_price(
                    ticker, analysis_date, evaluation_days
                )
                
                if future_price is None:
                    print(f"⚠️ Sem preço futuro para {analysis_date.strftime('%d/%m/%Y')}")
                    continue
                
                # Cria sinal
                current_price = historical_data['close'].iloc[-1]
                return_pct = ((future_price - current_price) / current_price) * 100
                
                is_correct = self._evaluate_signal_accuracy(
                    analysis.direction, return_pct
                )
                
                signal = BacktestSignal(
                    date=analysis_date,
                    ticker=ticker,
                    signal=analysis.direction,
                    confidence=analysis.confidence,
                    price_at_signal=current_price,
                    price_after_horizon=future_price,
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
        Busca dados históricos reais até uma data específica via OpLab API.
        """
        try:
            # Calcula período necessário (adiciona buffer para fins de semana)
            extended_days = max(min_days + 100, 400)  # Garante pelo menos 400 dias
            start_date = until_date - timedelta(days=int(extended_days * 1.4))
            
            # Formata datas para API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = until_date.strftime('%Y-%m-%d')
            
            print(f"📊 Buscando dados reais para {ticker}: {start_str} a {end_str}")
            
            # Busca dados reais via OpLab
            df = get_historical_data(ticker, start_str, end_str)
            
            if df.empty:
                print(f"⚠️ Nenhum dado encontrado para {ticker} no período")
                return pd.DataFrame()
            
            # Filtra até a data de análise (garante que não usa dados futuros)
            df = df[df['date'] <= until_date]
            
            # Garante mínimo de dados necessários
            if len(df) >= min_days:
                df = df.tail(min_days).reset_index(drop=True)
                print(f"✅ {len(df)} candles carregados para {ticker} até {until_date.strftime('%d/%m/%Y')}")
            else:
                print(f"⚠️ Apenas {len(df)} candles disponíveis para {ticker} (mínimo: {min_days})")
            
            return df
            
        except Exception as e:
            print(f"❌ Erro ao buscar dados históricos para {ticker}: {e}")
            return pd.DataFrame()
    
    def _get_future_price(
        self, 
        ticker: str, 
        analysis_date: datetime, 
        evaluation_days: int
    ) -> Optional[float]:
        """
        Busca preço real após evaluation_days via OpLab API.
        """
        try:
            # Calcula data futura (adiciona buffer para fins de semana)
            future_date = analysis_date + timedelta(days=int(evaluation_days * 1.4))
            
            # Formata datas para API
            start_str = analysis_date.strftime('%Y-%m-%d')
            end_str = future_date.strftime('%Y-%m-%d')
            
            print(f"🔮 Buscando preço futuro para {ticker}: {start_str} a {end_str}")
            
            # Busca dados reais do período futuro
            df = get_historical_data(ticker, start_str, end_str)
            
            if df.empty:
                print(f"⚠️ Sem dados futuros para {ticker} em {end_str}")
                return None
            
            # Remove o primeiro dia (data da análise) para evitar look-ahead bias
            df = df[df['date'] > analysis_date]
            
            if len(df) < evaluation_days:
                print(f"⚠️ Apenas {len(df)} dias futuros disponíveis para {ticker} (esperado: {evaluation_days})")
            
            if df.empty:
                return None
            
            # Pega o preço após evaluation_days (ou o último disponível)
            target_idx = min(evaluation_days - 1, len(df) - 1)
            future_price = df.iloc[target_idx]['close']
            
            print(f"✅ Preço futuro para {ticker}: {future_price:.2f} após {target_idx + 1} dias")
            
            return float(future_price)
            
        except Exception as e:
            print(f"❌ Erro ao buscar preço futuro para {ticker}: {e}")
            return None
    
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
        
        # Análise de performance
        if result.overall_accuracy >= 60:
            print("🎉 PERFORMANCE: EXCELENTE (≥60%)")
        elif result.overall_accuracy >= 50:
            print("👍 PERFORMANCE: BOA (50-59%)")
        elif result.overall_accuracy >= 40:
            print("⚠️ PERFORMANCE: REGULAR (40-49%)")
        else:
            print("❌ PERFORMANCE: PRECISA MELHORAR (<40%)")
        
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
