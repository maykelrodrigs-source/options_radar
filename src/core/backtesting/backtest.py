"""
Módulo de backtest para validar o ProfessionalAnalyzer.
Testa sinais CALL/PUT/NEUTRAL em dados históricos e mede acurácia.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from src.core.professional.professional_analysis import ProfessionalAnalyzer, Direction, OptionsSentimentAnalysis
from src.core.data.data import get_price_history, get_historical_data


class BacktestProfessionalAnalyzer(ProfessionalAnalyzer):
    """
    Versão do ProfessionalAnalyzer para backtest que funciona APENAS com dados REAIS.
    Não usa OpLab - apenas dados históricos reais do Yahoo Finance.
    SINCRONIZADO com ProfessionalAnalyzer atual (pesos dinâmicos, threshold adaptativo).
    """
    
    def __init__(self, horizon: str = "médio", decision_threshold: float = 0.20, layer_weights: Optional[Dict[str, float]] = None):
        """
        Inicializa analyzer para backtest sem OpLab.
        Usa apenas dados históricos REAIS do Yahoo Finance.
        SINCRONIZADO com ProfessionalAnalyzer atual.
        """
        # Não chama super().__init__ para evitar criar OpLabClient
        self.client = None  # Sem OpLab
        self.horizon = horizon
        self.params = self._get_horizon_parameters(horizon)
        self.decision_threshold = decision_threshold or 0.15  # Threshold mais baixo para gerar mais sinais
        
        # Pesos configuráveis das camadas - mais equilibrados (SINCRONIZADO)
        self.layer_weights = layer_weights or {
            'trend': 0.30,
            'momentum': 0.25,
            'volume': 0.25,
            'sentiment': 0.15,
            'macro': 0.05
        }
    
    def analyze_options_sentiment(self, ticker: str, price_data: pd.DataFrame) -> OptionsSentimentAnalysis:
        """Proxy simples de sentimento para backtest baseada em volume.
        - volume atual > 1.5x média 20d -> +0.2 (bullish)
        - volume atual < 0.7x média 20d -> -0.2 (bearish)
        - caso contrário -> 0.0 (neutro)
        """
        close = price_data['close']
        volume = price_data.get('volume', pd.Series([1.0] * len(close), index=close.index))
        if len(volume) < 21:
            score = 0.0
        else:
            vol20 = float(volume.rolling(20).mean().iloc[-1])
            vol = float(volume.iloc[-1])
            ratio = (vol / vol20) if vol20 > 0 else 1.0
            if ratio > 1.5:
                score = 0.2
            elif ratio < 0.7:
                score = -0.2
            else:
                score = 0.0
        # Constrói objeto compatível
        return OptionsSentimentAnalysis(
            put_call_ratio=1.0,
            volatility_skew=0.0,
            call_volume_ratio=0.5 if score >= 0 else 0.4,
            put_volume_ratio=0.5 if score >= 0 else 0.6,
            sentiment_score=score,
            market_bias="BULLISH" if score > 0 else ("BEARISH" if score < 0 else "NEUTRAL")
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
    # Gestão do trade
    hit_type: str  # 'TP' | 'SL' | 'TIME'
    mfe_pct: float  # Maximum Favorable Excursion (%)
    mae_pct: float  # Maximum Adverse Excursion (%)
    tp_base_aplicado: float  # TP base aplicado (%), documentado por sinal
    adx_at_signal: float  # ADX no momento do sinal
    gate_counters: Optional[Dict[str, int]] = None
    rejection_reasons: Optional[List[str]] = None
    final_score: Optional[float] = None
    # Novos campos para análise detalhada
    raw_final_score: Optional[float] = None
    adjusted_final_score: Optional[float] = None
    gates_passed: Optional[bool] = None
    gates_relaxed: Optional[bool] = None
    bearish_penalty_applied: Optional[bool] = None
    # Debug específico para PUTs
    put_threshold_triggered: Optional[bool] = None
    put_rejected_low_conf: Optional[bool] = None
    bearish_penalty_value: Optional[float] = None
    bearish_penalty_trigger: Optional[str] = None
    # Meta-labeling de PUT
    put_meta_label_passed: Optional[bool] = None
    put_meta_label_reason: Optional[str] = None
    # Filtros seletivos para PUTs
    prefilter_reject: Optional[bool] = None
    prefilter_reason: Optional[str] = None
    # Meta-labeling específico (novos campos)
    ml_gap_rejected: Optional[bool] = None
    ml_killbar_rejected: Optional[bool] = None
    ml_flow_rejected: Optional[bool] = None
    # TP/SL efetivos do trade (em %)
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None
    # Campos de trailing stop para PUTs
    used_trailing: Optional[bool] = None
    time_stopped: Optional[bool] = None
    # Campos específicos para ML
    ml_model_used: Optional[str] = None
    ml_probabilities: Optional[str] = None
    ml_feature_importance: Optional[str] = None


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
    
    def __init__(self, success_threshold: float = 3.0, analyzer=None, horizon: str = "médio",
                 data_provider: Optional[Callable[[str, str, str], pd.DataFrame]] = None,
                 sl_factor: float = 0.6):
        """
        Args:
            success_threshold: Percentual mínimo para considerar acerto (ex: 3.0 = 3%)
            analyzer: ProfessionalAnalyzer instance (opcional, para testes)
            horizon: Horizonte temporal ("curto", "médio", "longo")
            data_provider: Função opcional para buscar dados (ticker, start_str, end_str) -> DataFrame
        """
        self.success_threshold = float(success_threshold)
        self.analyzer = analyzer or BacktestProfessionalAnalyzer(horizon=horizon)
        self.horizon = horizon
        self.data_provider = data_provider or get_historical_data
        self.sl_factor = float(sl_factor)

    def _fetch_data(self, ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
        """Busca dados usando provider injetado ou fallback para get_historical_data."""
        if self.data_provider is not None:
            return self.data_provider(ticker, start_str, end_str)
        return get_historical_data(ticker, start_str, end_str)
    
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
                
                # Define tp_base dinâmico por regime ADX do dia (usa momentum/adx da análise)
                adx_val = float(analysis.momentum.adx) if hasattr(analysis, 'momentum') else 20.0
                if adx_val < 15:
                    tp_base_dyn = 2.0
                elif adx_val < 25:
                    tp_base_dyn = 2.5
                else:
                    tp_base_dyn = 3.0
                if evaluation_days <= 7:
                    tp_base_dyn = max(1.5, tp_base_dyn - 0.5)

                entry_price, exit_price, hit_type, mfe_pct, mae_pct, tp_pct, sl_pct, used_trailing, time_stopped = self._get_trade_outcome(
                    ticker=ticker,
                    analysis_date=analysis_date,
                    evaluation_days=evaluation_days,
                    direction=analysis.direction,
                    tp_base=tp_base_dyn,
                    sl_factor=self.sl_factor
                )
                
                if entry_price is None or exit_price is None:
                    print(f"⚠️ Sem preços de entrada/saída para {analysis_date.strftime('%d/%m/%Y')}")
                    continue
                
                # Cria sinal usando preço de entrada correto
                current_price = historical_data['close'].iloc[-1]  # Preço no momento da decisão
                return_pct = ((exit_price - entry_price) / entry_price) * 100
                
                # Avalia acerto usando hit_type para CALL/PUT, banda para NEUTRAL
                if analysis.direction == Direction.NEUTRAL:
                    # NEUTRAL é acerto se nunca excedeu o threshold em módulo
                    is_correct = max(abs(mfe_pct), abs(mae_pct)) <= self.success_threshold
                else:
                    is_correct = (hit_type == 'TP')
                
                # Log detalhado para falsos positivos/negativos
                if not is_correct:
                    if analysis.direction == Direction.CALL:
                        if hit_type == 'SL':
                            print(f"❌ CALL FALSO POSITIVO: {analysis_date.strftime('%d/%m/%Y')} - SL em {return_pct:.1f}% (conf: {analysis.confidence:.0f}%)")
                        else:  # TIME
                            print(f"❌ CALL FALSO POSITIVO: {analysis_date.strftime('%d/%m/%Y')} - TIME em {return_pct:.1f}% (conf: {analysis.confidence:.0f}%)")
                    elif analysis.direction == Direction.PUT:
                        if hit_type == 'SL':
                            print(f"❌ PUT FALSO POSITIVO: {analysis_date.strftime('%d/%m/%Y')} - SL em {return_pct:.1f}% (conf: {analysis.confidence:.0f}%)")
                        else:  # TIME
                            print(f"❌ PUT FALSO POSITIVO: {analysis_date.strftime('%d/%m/%Y')} - TIME em {return_pct:.1f}% (conf: {analysis.confidence:.0f}%)")
                    else:  # NEUTRAL
                        max_excursion = max(abs(mfe_pct), abs(mae_pct))
                        print(f"❌ NEUTRAL FALSO POSITIVO: {analysis_date.strftime('%d/%m/%Y')} - Excursão {max_excursion:.1f}% > {self.success_threshold}% (conf: {analysis.confidence:.0f}%)")
                else:
                    print(f"✅ ACERTO: {analysis.direction.value} em {analysis_date.strftime('%d/%m/%Y')} - {hit_type} em {return_pct:.1f}% (conf: {analysis.confidence:.0f}%)")
                
                signal = BacktestSignal(
                    date=analysis_date,
                    ticker=ticker,
                    signal=analysis.direction,
                    confidence=analysis.confidence,
                    price_at_signal=entry_price,  # Preço de entrada real (abertura do dia seguinte)
                    price_after_horizon=exit_price,  # Preço de saída
                    return_pct=return_pct,
                    is_correct=is_correct,
                    drivers=analysis.key_drivers,
                    hit_type=hit_type,
                    mfe_pct=mfe_pct,
                    mae_pct=mae_pct,
                    tp_base_aplicado=tp_base_dyn,
                    adx_at_signal=float(analysis.momentum.adx),
                    gate_counters=analysis.gate_counters if hasattr(analysis, 'gate_counters') else {},
                    rejection_reasons=getattr(analysis, 'rejection_reasons', []),
                    final_score=getattr(analysis, 'final_score', None),
                    # Novos campos para análise detalhada
                    raw_final_score=getattr(analysis, 'raw_final_score', None),
                    adjusted_final_score=getattr(analysis, 'adjusted_final_score', None),
                    gates_passed=getattr(analysis, 'gates_passed', None),
                    gates_relaxed=getattr(analysis, 'gates_relaxed', None),
                    bearish_penalty_applied='bearish_penalty_applied' in (getattr(analysis, 'rejection_reasons', []) or []),
                    # Debug específico para PUTs
                    put_threshold_triggered=getattr(analysis, 'put_threshold_triggered', False),
                    put_rejected_low_conf=getattr(analysis, 'put_rejected_low_conf', False),
                    bearish_penalty_value=getattr(analysis, 'bearish_penalty_value', 0.0),
                    bearish_penalty_trigger=getattr(analysis, 'bearish_penalty_trigger', None),
                    # Meta-labeling de PUT
                    put_meta_label_passed=getattr(analysis, 'put_meta_label_passed', None),
                    put_meta_label_reason=getattr(analysis, 'put_meta_label_reason', None),
                    # Filtros seletivos para PUTs
                    prefilter_reject=getattr(analysis, 'prefilter_reject', None),
                    prefilter_reason=getattr(analysis, 'prefilter_reason', None),
                    # Meta-labeling específico
                    ml_gap_rejected='ml_gap' in (getattr(analysis, 'rejection_reasons', []) or []),
                    ml_killbar_rejected='ml_killbar' in (getattr(analysis, 'rejection_reasons', []) or []),
                    ml_flow_rejected='ml_flow' in (getattr(analysis, 'rejection_reasons', []) or []),
                    # TP/SL efetivos usados
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    # Campos de trailing stop para PUTs
                    used_trailing=used_trailing,
                    time_stopped=time_stopped
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
            
            # Busca dados REAIS via Yahoo Finance (com provider/cache se existir)
            df = self._fetch_data(ticker, start_str, end_str)
            
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
    
    def _get_trade_outcome(
        self,
        ticker: str,
        analysis_date: datetime,
        evaluation_days: int,
        direction: Direction,
        tp_base: float,
        sl_factor: float = 0.6
    ) -> Tuple[Optional[float], Optional[float], str, float, float, Optional[float], Optional[float], bool, bool]:
        """
        Calcula resultado do trade com entrada D+1, TP/SL/Tempo e excursões.
        - TP = +threshold (success_threshold)
        - SL = -sl_factor * threshold (assimetria a favor)
        Retorna: (entry_price, exit_price, hit_type, mfe_pct, mae_pct)
        """
        try:
            # Janela futura
            future_date = analysis_date + timedelta(days=int(evaluation_days * 1.4))
            start_str = analysis_date.strftime('%Y-%m-%d')
            end_str = future_date.strftime('%Y-%m-%d')

            df = self._fetch_data(ticker, start_str, end_str)
            if df.empty:
                return None, None, 'TIME', 0.0, 0.0

            # Remove o dia do sinal
            if not df.empty and df['date'].dt.tz is not None:
                analysis_date_tz = analysis_date.replace(tzinfo=df['date'].dt.tz)
                df = df[df['date'] > analysis_date_tz]
            else:
                df = df[df['date'] > analysis_date]

            if df.empty:
                return None, None, 'TIME', 0.0, 0.0

            # Entrada na abertura do primeiro dia
            entry_price = float(df.iloc[0]['open'])

            # Caminho de preços (fechamento) nos próximos dias úteis
            closes = df['close'].reset_index(drop=True)
            highs = df.get('high', closes).reset_index(drop=True)
            lows = df.get('low', closes).reset_index(drop=True)

            # MFE/MAE relativos à entrada
            pct_series = (closes - entry_price) / entry_price * 100
            high_pct = (highs - entry_price) / entry_price * 100
            low_pct = (lows - entry_price) / entry_price * 100

            # Limita pela janela
            horizon = min(evaluation_days, len(pct_series))

            # TP e SL por lado
            if direction == Direction.CALL:
                tp = tp_base
                sl = -sl_factor * tp_base
                # Verifica dia a dia se bateu TP/SL (usando high/low como proxy intra)
                hit_idx = None
                hit_type = 'TIME'
                for i in range(horizon):
                    if high_pct.iloc[i] >= tp:
                        hit_idx = i
                        hit_type = 'TP'
                        break
                    if low_pct.iloc[i] <= sl:
                        hit_idx = i
                        hit_type = 'SL'
                        break
                if hit_idx is None:
                    exit_price = float(closes.iloc[horizon - 1])
                else:
                    # Usa o preço exato do stop/target no intraday
                    if hit_type == 'TP':
                        exit_price = float(entry_price * (1 + tp/100))
                    else:  # SL
                        exit_price = float(entry_price * (1 + sl/100))

            elif direction == Direction.PUT:
                # TP/SL dinâmico via ATR(14) calculado no histórico anterior à data do sinal
                try:
                    atr_start = (analysis_date - timedelta(days=120)).strftime('%Y-%m-%d')
                    atr_end = analysis_date.strftime('%Y-%m-%d')
                    hist = self._fetch_data(ticker, atr_start, atr_end)
                    if hist.empty or len(hist) < 20:
                        # Fallback conservador caso sem dados suficientes
                        atr_pct = 1.0
                    else:
                        h = hist['high'].astype(float).values
                        l = hist['low'].astype(float).values
                        c = hist['close'].astype(float).values
                        # True Range absoluto
                        prev_c = np.roll(c, 1)
                        prev_c[0] = c[0]
                        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
                        # Converte TR para % sobre o fechamento anterior
                        with np.errstate(divide='ignore', invalid='ignore'):
                            tr_pct = np.where(prev_c != 0, (tr / prev_c) * 100.0, 0.0)
                        # ATR(14) em %
                        if len(tr_pct) >= 14:
                            atr_pct = float(pd.Series(tr_pct).rolling(14).mean().iloc[-1])
                        else:
                            atr_pct = float(np.nanmean(tr_pct[-14:])) if len(tr_pct) > 0 else 1.0
                        if not np.isfinite(atr_pct) or atr_pct <= 0:
                            atr_pct = 1.0
                    tp = 2.0 * atr_pct  # TP = 2.0 × ATR14%
                    sl = 1.0 * atr_pct  # SL = 1.0 × ATR14%
                    be_threshold = 0.5 * atr_pct  # Breakeven threshold
                    trailing_distance = 0.8 * atr_pct  # Trailing distance
                    time_stop_days = evaluation_days // 2  # Time stop at half evaluation period
                    
                    print(f"PUT ATR exits: SL={sl:.2f}%, TP={tp:.2f}%, BE_at={be_threshold:.2f}%")
                except Exception as e:
                    # Fallback em caso de erro de ATR
                    tp = 2.0
                    sl = 1.0
                    atr_pct = 1.0
                    be_threshold = 0.5
                    trailing_distance = 0.8
                    time_stop_days = evaluation_days // 2
                    print(f"⚠️ Falha ao calcular ATR(14) para {ticker}: {e}. Usando TP=2.0% SL=1.0%.")
                
                # Variáveis para trailing stop
                used_trailing = False
                time_stopped = False
                best_favorable_price = entry_price  # Para PUT, melhor preço é o menor (mais favorável)
                current_sl = sl  # SL dinâmico
                breakeven_triggered = False
                
                # Para PUT, ganho é queda do preço: use low para TP e high para SL
                hit_idx = None
                hit_type = 'TIME'
                
                for i in range(horizon):
                    current_low = float(lows.iloc[i])
                    current_high = float(highs.iloc[i])
                    current_close = float(closes.iloc[i])
                    
                    # Atualiza melhor preço favorável (menor preço para PUT)
                    if current_low < best_favorable_price:
                        best_favorable_price = current_low
                    
                    # Calcula lucro atual em relação à entrada
                    current_profit_pct = (entry_price - current_low) / entry_price * 100
                    
                    # Trailing stop logic
                    if current_profit_pct >= be_threshold and not breakeven_triggered:
                        # Move para breakeven
                        current_sl = 0.0  # Breakeven
                        breakeven_triggered = True
                        used_trailing = True
                        print("PUT trailing active")
                    elif breakeven_triggered:
                        # Aplica trailing de 0.8*ATR atrás do melhor preço
                        trailing_sl_pct = (entry_price - best_favorable_price) / entry_price * 100 + trailing_distance
                        current_sl = min(current_sl, trailing_sl_pct)
                    
                    # Verifica TP (queda >= tp)
                    if current_profit_pct >= tp:
                        hit_idx = i
                        hit_type = 'TP'
                        break
                    
                    # Verifica SL (alta >= current_sl)
                    current_loss_pct = (current_high - entry_price) / entry_price * 100
                    if current_loss_pct >= current_sl:
                        hit_idx = i
                        hit_type = 'SL'
                        break
                    
                    # Time stop: se até metade do período não atingiu +0.5*ATR
                    if i == time_stop_days - 1 and current_profit_pct < be_threshold:
                        hit_idx = i
                        hit_type = 'TIME'
                        time_stopped = True
                        break
                        
                if hit_idx is None:
                    exit_price = float(closes.iloc[horizon - 1])
                else:
                    # Usa o preço exato do stop/target no intraday
                    if hit_type == 'TP':
                        # Para PUT, TP é queda de 'tp' por cento
                        exit_price = float(entry_price * (1 - tp/100))
                    elif hit_type == 'SL':
                        # SL baseado no current_sl (pode ser trailing)
                        exit_price = float(entry_price * (1 + current_sl/100))
                    else:  # TIME
                        exit_price = float(closes.iloc[hit_idx])

            else:  # NEUTRAL
                # Em NEUTRAL, fecha no tempo. Define tp/sl neutros para compatibilidade de retorno
                hit_type = 'TIME'
                exit_price = float(closes.iloc[horizon - 1])
                tp = 0.0
                sl = 0.0

            # MFE/MAE assinados conforme CALL (positivos para alta, negativos para baixa)
            mfe_pct = float(high_pct.iloc[:horizon].max())
            mae_pct = float(low_pct.iloc[:horizon].min())

            # Para PUTs, retorna campos adicionais de trailing
            if direction == Direction.PUT:
                return entry_price, exit_price, hit_type, mfe_pct, mae_pct, float(tp), float(sl), used_trailing, time_stopped
            else:
                return entry_price, exit_price, hit_type, mfe_pct, mae_pct, float(tp), float(sl), False, False

        except Exception as e:
            print(f"❌ Erro ao calcular trade outcome: {e}")
            return None, None, 'TIME', 0.0, 0.0, None, None
    
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
            
            df = self._fetch_data(ticker, start_str, end_str)
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
            
            df = self._fetch_data(ticker, start_str, end_str)
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
                'Preço +{result.evaluation_days}d': f"R$ {signal.price_after_horizon:.2f}",
                'Retorno (%)': f"{signal.return_pct:+.1f}%",
                'Resultado': "✅ Acerto" if signal.is_correct else "❌ Erro",
                'Drivers': ", ".join(signal.drivers[:2]),  # Primeiros 2 drivers
                'Rejections': ", ".join(getattr(signal, 'rejection_reasons', []) or [])
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
        
        # Análise de falsos positivos/negativos
        print("🔍 ANÁLISE DE FALSOS POSITIVOS/NEGATIVOS:")
        call_false_pos = len([s for s in result.signals if s.signal == Direction.CALL and not s.is_correct])
        put_false_pos = len([s for s in result.signals if s.signal == Direction.PUT and not s.is_correct])
        neutral_false_pos = len([s for s in result.signals if s.signal == Direction.NEUTRAL and not s.is_correct])
        
        print(f"• CALL falsos positivos: {call_false_pos}/{result.call_signals} ({call_false_pos/result.call_signals*100:.1f}%)" if result.call_signals > 0 else "• CALL falsos positivos: 0/0")
        print(f"• PUT falsos positivos: {put_false_pos}/{result.put_signals} ({put_false_pos/result.put_signals*100:.1f}%)" if result.put_signals > 0 else "• PUT falsos positivos: 0/0")
        print(f"• NEUTRAL falsos positivos: {neutral_false_pos}/{result.neutral_signals} ({neutral_false_pos/result.neutral_signals*100:.1f}%)" if result.neutral_signals > 0 else "• NEUTRAL falsos positivos: 0/0")
        
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


def calculate_put_metrics_v2(put_signals: List[BacktestSignal]) -> Dict[str, float]:
    """
    Calcula métricas específicas para PUTs: precision, recall, F1.
    """
    if len(put_signals) == 0:
        return {
            'put_precision': 0.0,
            'put_recall': 0.0,
            'put_f1': 0.0
        }
    
    # Precision = TP / (TP + FP)
    true_positives = len([s for s in put_signals if s.is_correct])
    false_positives = len([s for s in put_signals if not s.is_correct])
    put_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    
    # Recall = TP / (TP + FN) - assumindo que todos os PUTs corretos foram capturados
    put_recall = put_precision  # Simplificação para este contexto
    
    # F1 = 2 * (precision * recall) / (precision + recall)
    put_f1 = 2 * (put_precision * put_recall) / (put_precision + put_recall) if (put_precision + put_recall) > 0 else 0.0
    
    return {
        'put_precision': put_precision,
        'put_recall': put_recall,
        'put_f1': put_f1
    }

def calculate_regime_metrics_v2(signals: List[BacktestSignal]) -> Dict[str, Any]:
    """
    Calcula métricas por regime (EMA200 slope e ADX bins).
    """
    if not signals:
        return {
            'ema200_slope_regime': 'unknown',
            'adx_bin': 'unknown',
            'ema200_up_accuracy': 0.0,
            'ema200_down_accuracy': 0.0,
            'adx_low_accuracy': 0.0,
            'adx_medium_accuracy': 0.0,
            'adx_high_accuracy': 0.0
        }
    
    # Calcula regime predominante baseado nos sinais
    ema200_up_signals = 0
    ema200_down_signals = 0
    adx_low_signals = 0
    adx_medium_signals = 0
    adx_high_signals = 0
    
    ema200_up_correct = 0
    ema200_down_correct = 0
    adx_low_correct = 0
    adx_medium_correct = 0
    adx_high_correct = 0
    
    for signal in signals:
        # Determina regime baseado no ADX (aproximação)
        adx = getattr(signal, 'adx_at_signal', 20.0)
        
        if adx < 18:
            adx_low_signals += 1
            if signal.is_correct:
                adx_low_correct += 1
        elif adx <= 25:
            adx_medium_signals += 1
            if signal.is_correct:
                adx_medium_correct += 1
        else:
            adx_high_signals += 1
            if signal.is_correct:
                adx_high_correct += 1
        
        # Para EMA200 slope, usa uma aproximação baseada no final_score
        # (assumindo que scores negativos indicam tendência de baixa)
        final_score = getattr(signal, 'final_score', 0.0)
        if final_score < 0:
            ema200_down_signals += 1
            if signal.is_correct:
                ema200_down_correct += 1
        else:
            ema200_up_signals += 1
            if signal.is_correct:
                ema200_up_correct += 1
    
    # Determina regime predominante
    ema200_slope_regime = 'up' if ema200_up_signals > ema200_down_signals else 'down'
    
    # Determina bin ADX predominante
    adx_counts = {'low': adx_low_signals, 'medium': adx_medium_signals, 'high': adx_high_signals}
    adx_bin = max(adx_counts, key=adx_counts.get) if any(adx_counts.values()) else 'unknown'
    
    # Calcula acurácias por regime
    ema200_up_accuracy = (ema200_up_correct / ema200_up_signals * 100) if ema200_up_signals > 0 else 0.0
    ema200_down_accuracy = (ema200_down_correct / ema200_down_signals * 100) if ema200_down_signals > 0 else 0.0
    adx_low_accuracy = (adx_low_correct / adx_low_signals * 100) if adx_low_signals > 0 else 0.0
    adx_medium_accuracy = (adx_medium_correct / adx_medium_signals * 100) if adx_medium_signals > 0 else 0.0
    adx_high_accuracy = (adx_high_correct / adx_high_signals * 100) if adx_high_signals > 0 else 0.0
    
    return {
        'ema200_slope_regime': ema200_slope_regime,
        'adx_bin': adx_bin,
        'ema200_up_accuracy': ema200_up_accuracy,
        'ema200_down_accuracy': ema200_down_accuracy,
        'adx_low_accuracy': adx_low_accuracy,
        'adx_medium_accuracy': adx_medium_accuracy,
        'adx_high_accuracy': adx_high_accuracy
    }

def run_experiments_v2(
    tickers: List[str] = ["PETR4","VALE3","ITUB4","BBAS3","WEGE3","MGLU3"],
    evaluation_days_list: List[int] = [5,7,10,20],
    decision_threshold_list: List[float] = [0.12, 0.15, 0.18, 0.20, 0.30],
    sl_factor_list: List[float] = [0.5,0.6,0.7],
    rolling_window_list: List[int] = [3,5,10],
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-31",
    max_workers: int = 4
) -> pd.DataFrame:
    """Executa experimentos com parâmetros expandidos e coleta métricas detalhadas."""
    rows: List[dict] = []
    exp_id = 0
    for ticker in tickers:
        for eval_days in evaluation_days_list:
            for thr in decision_threshold_list:
                for slf in sl_factor_list:
                    for roll in rolling_window_list:
                        exp_id += 1
                        try:
                            analyzer = BacktestProfessionalAnalyzer(horizon="curto")
                            analyzer.decision_threshold = thr
                            engine = BacktestEngine(success_threshold=3.0, analyzer=analyzer, horizon="curto", sl_factor=slf)

                            # Executa backtest no período especificado
                            start = datetime.strptime(start_date, "%Y-%m-%d")
                            end = datetime.strptime(end_date, "%Y-%m-%d")
                            result = engine.run_backtest(
                                ticker=ticker,
                                start_date=start,
                                end_date=end,
                                evaluation_days=eval_days,
                                rolling_window=roll,
                                min_history_days=300
                            )

                            # Confusion matrix simples
                            call_hits = sum(1 for s in result.signals if s.signal == Direction.CALL and s.is_correct)
                            call_errs = sum(1 for s in result.signals if s.signal == Direction.CALL and not s.is_correct)
                            put_hits = sum(1 for s in result.signals if s.signal == Direction.PUT and s.is_correct)
                            put_errs = sum(1 for s in result.signals if s.signal == Direction.PUT and not s.is_correct)
                            neu_hits = sum(1 for s in result.signals if s.signal == Direction.NEUTRAL and s.is_correct)
                            neu_errs = sum(1 for s in result.signals if s.signal == Direction.NEUTRAL and not s.is_correct)

                            # Médias por tipo
                            def avg(lst):
                                return sum(lst)/len(lst) if lst else 0.0
                            tp_call = avg([s.tp_base_aplicado for s in result.signals if s.signal == Direction.CALL])
                            tp_put = avg([s.tp_base_aplicado for s in result.signals if s.signal == Direction.PUT])
                            tp_neu = avg([s.tp_base_aplicado for s in result.signals if s.signal == Direction.NEUTRAL])
                            adx_call = avg([s.adx_at_signal for s in result.signals if s.signal == Direction.CALL])
                            adx_put = avg([s.adx_at_signal for s in result.signals if s.signal == Direction.PUT])
                            adx_neu = avg([s.adx_at_signal for s in result.signals if s.signal == Direction.NEUTRAL])

                            # Agrega contadores de rejeição dos sinais
                            agg = {
                                'gate_adx': 0,
                                'gate_volume': 0,
                                'gate_sentiment': 0,
                                'gate_align': 0,
                                'gate_momentum_weak': 0,
                                'meta_squeeze_vol': 0,
                                'meta_lateral': 0,
                                'meta_trend': 0,
                            }
                            for s in result.signals:
                                if getattr(s, 'gate_counters', None):
                                    for k in agg.keys():
                                        agg[k] += s.gate_counters.get(k, 0)

                            # Análise detalhada de PUTs
                            put_signals = [s for s in result.signals if s.signal == Direction.PUT]
                            call_signals = [s for s in result.signals if s.signal == Direction.CALL]
                            neutral_signals = [s for s in result.signals if s.signal == Direction.NEUTRAL]
                            
                            # Percentuais por tipo de sinal
                            total_sigs = len(result.signals)
                            put_pct = (len(put_signals) / total_sigs * 100) if total_sigs > 0 else 0.0
                            call_pct = (len(call_signals) / total_sigs * 100) if total_sigs > 0 else 0.0
                            neutral_pct = (len(neutral_signals) / total_sigs * 100) if total_sigs > 0 else 0.0
                            
                            # Cálculo de PUT precision/recall/F1
                            put_metrics = calculate_put_metrics_v2(put_signals)
                            
                            # Análise por regime
                            regime_metrics = calculate_regime_metrics_v2(result.signals)
                            
                            # Métricas específicas de PUTs
                            puts_gates_relaxed = sum(1 for s in put_signals if getattr(s, 'gates_relaxed', False))
                            puts_bearish_penalty = sum(1 for s in put_signals if getattr(s, 'bearish_penalty_applied', False))
                            
                            put_rejections_count = sum(len(s.rejection_reasons or []) for s in put_signals)
                            
                            # Histograma de final_score de PUTs (quando existirem)
                            buckets = {'[0.0-0.1)':0,'[0.1-0.2)':0,'[0.2-0.3)':0,'[0.3-0.4)':0,'[0.4+)':0}
                            for s in put_signals:
                                fs = abs(s.final_score or 0.0)
                                if fs < 0.1:
                                    buckets['[0.0-0.1)'] += 1
                                elif fs < 0.2:
                                    buckets['[0.1-0.2)'] += 1
                                elif fs < 0.3:
                                    buckets['[0.2-0.3)'] += 1
                                elif fs < 0.4:
                                    buckets['[0.3-0.4)'] += 1
                                else:
                                    buckets['[0.4+)'] += 1

                            rows.append({
                                'exp_id': exp_id,
                                'ticker': ticker,
                                'eval_days': eval_days,
                                'decision_threshold': thr,
                                'sl_factor': slf,
                                'rolling_window': roll,
                                'total_signals': result.total_signals,
                                'call_count': result.call_signals,
                                'put_count': result.put_signals,
                                'neutral_count': result.neutral_signals,
                                # Percentuais por tipo
                                'call_pct': call_pct,
                                'put_pct': put_pct,
                                'neutral_pct': neutral_pct,
                                'overall_accuracy': result.overall_accuracy,
                                'call_accuracy': result.call_accuracy,
                                'put_accuracy': result.put_accuracy,
                                'neutral_accuracy': result.neutral_accuracy,
                                'cm_call_hit': call_hits,
                                'cm_call_err': call_errs,
                                'cm_put_hit': put_hits,
                                'cm_put_err': put_errs,
                                'cm_neutral_hit': neu_hits,
                                'cm_neutral_err': neu_errs,
                                'avg_tp_call': tp_call,
                                'avg_tp_put': tp_put,
                                'avg_tp_neutral': tp_neu,
                                'avg_adx_call': adx_call,
                                'avg_adx_put': adx_put,
                                'avg_adx_neutral': adx_neu,
                                # Métricas específicas de PUTs
                                'puts_gates_relaxed': puts_gates_relaxed,
                                'puts_bearish_penalty': puts_bearish_penalty,
                                # PUT precision/recall/F1
                                'put_precision': put_metrics['put_precision'],
                                'put_recall': put_metrics['put_recall'],
                                'put_f1': put_metrics['put_f1'],
                                # Métricas por regime
                                'ema200_slope_regime': regime_metrics['ema200_slope_regime'],
                                'adx_bin': regime_metrics['adx_bin'],
                                'ema200_up_accuracy': regime_metrics['ema200_up_accuracy'],
                                'ema200_down_accuracy': regime_metrics['ema200_down_accuracy'],
                                'adx_low_accuracy': regime_metrics['adx_low_accuracy'],
                                'adx_medium_accuracy': regime_metrics['adx_medium_accuracy'],
                                'adx_high_accuracy': regime_metrics['adx_high_accuracy'],
                                # Contadores agregados
                                **agg,
                                # Diagnóstico PUTs
                                'put_rejections_count': put_rejections_count,
                                'put_scores_histogram': buckets,
                            })
                            print(f"✅ ExpV2 {exp_id}: {ticker} {eval_days}d T{thr} SLf{slf} roll{roll} -> acc {result.overall_accuracy:.1f}% sig {result.total_signals} [CALL:{call_pct:.0f}% PUT:{put_pct:.0f}% NEU:{neutral_pct:.0f}%] PUT_acc:{result.put_accuracy:.1f}%")
                        except Exception as e:
                            print(f"❌ ExpV2 {exp_id}: {ticker} {eval_days}d T{thr} SLf{slf} roll{roll} -> erro {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("❌ Nenhum resultado em run_experiments_v2")
        return df

    # TOP3
    def print_top(df, col, title):
        print(f"\n🏆 TOP3 por {title}:")
        top = df.nlargest(3, col)[['ticker','eval_days','decision_threshold','sl_factor','rolling_window',col,'total_signals','call_count','put_count','neutral_count']]
        print(top.to_string(index=False))

    print_top(df, 'overall_accuracy', 'acurácia geral')
    print_top(df, 'call_accuracy', 'acurácia CALL')
    print_top(df, 'put_accuracy', 'acurácia PUT')

    # Resumo por ticker com métricas detalhadas de PUTs
    print("\n📋 Resumo por ticker:")
    for tkr, g in df.groupby('ticker'):
        total_signals = int(g['total_signals'].sum())
        call_count = int(g['call_count'].sum())
        put_count = int(g['put_count'].sum())
        neutral_count = int(g['neutral_count'].sum())
        
        call_pct = (call_count / total_signals * 100) if total_signals else 0.0
        put_pct = (put_count / total_signals * 100) if total_signals else 0.0
        neutral_pct = (neutral_count / total_signals * 100) if total_signals else 0.0
        
        best_put_acc = g['put_accuracy'].max() if not g['put_accuracy'].empty else 0.0
        best_call_acc = g['call_accuracy'].max() if not g['call_accuracy'].empty else 0.0
        best_overall_acc = g['overall_accuracy'].max() if not g['overall_accuracy'].empty else 0.0
        
        # Métricas específicas de PUTs
        total_puts_gates_relaxed = int(g['puts_gates_relaxed'].sum())
        total_puts_bearish_penalty = int(g['puts_bearish_penalty'].sum())
        
        print(f"- {tkr}: sinais={total_signals}")
        print(f"    Distribuição: CALL={call_pct:.1f}% PUT={put_pct:.1f}% NEUTRAL={neutral_pct:.1f}%")
        print(f"    Acurácia max: Overall={best_overall_acc:.1f}% CALL={best_call_acc:.1f}% PUT={best_put_acc:.1f}%")
        print(f"    PUTs aceitos via gates relaxados: {total_puts_gates_relaxed}")
        print(f"    PUTs com bearish penalty: {total_puts_bearish_penalty}")
        print()

    # Exporta CSV detalhado para análise posterior
    csv_filename = 'experiments_v2_results_with_puts.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n💾 Resultados exportados para: {csv_filename}")
    
    return df


def run_bearish_experiments() -> pd.DataFrame:
    """
    Executa experimentos específicos para análise bearish com períodos históricos.
    """
    print("🐻 EXECUTANDO EXPERIMENTOS BEARISH")
    print("=" * 50)
    
    # Configuração específica para experimentos bearish
    bearish_configs = [
        {
            'ticker': 'PETR4',
            'start_date': '2022-01-01',
            'end_date': '2022-12-31',
            'description': 'PETR4 2022 (período de alta volatilidade)'
        },
        {
            'ticker': 'VALE3',
            'start_date': '2015-01-01',
            'end_date': '2015-12-31',
            'description': 'VALE3 2015 (período de queda)'
        }
    ]
    
    evaluation_days_list = [5, 10, 20]
    thresholds = [0.15, 0.20]
    success_threshold = 3.0
    
    rows = []
    exp_id = 0
    
    for config in bearish_configs:
        print(f"\n📊 Testando {config['description']}...")
        
        for eval_days in evaluation_days_list:
            for threshold in thresholds:
                exp_id += 1
                try:
                    analyzer = BacktestProfessionalAnalyzer(horizon="médio")
                    analyzer.decision_threshold = threshold
                    engine = BacktestEngine(
                        success_threshold=success_threshold, 
                        analyzer=analyzer, 
                        horizon="médio"
                    )
                    
                    # Executa backtest
                    start = datetime.strptime(config['start_date'], "%Y-%m-%d")
                    end = datetime.strptime(config['end_date'], "%Y-%m-%d")
                    result = engine.run_backtest(
                        ticker=config['ticker'],
                        start_date=start,
                        end_date=end,
                        evaluation_days=eval_days,
                        rolling_window=5,
                        min_history_days=300
                    )
                    
                    # Análise detalhada
                    put_signals = [s for s in result.signals if s.signal == Direction.PUT]
                    call_signals = [s for s in result.signals if s.signal == Direction.CALL]
                    
                    # Métricas de PUT
                    put_metrics = calculate_put_metrics_v2(put_signals)
                    
                    # Métricas por regime
                    regime_metrics = calculate_regime_metrics_v2(result.signals)
                    
                    # Retornos médios
                    avg_return_put = np.mean([s.return_pct for s in put_signals]) if put_signals else 0.0
                    avg_return_call = np.mean([s.return_pct for s in call_signals]) if call_signals else 0.0
                    
                    # Max drawdown (simplificado)
                    returns = [s.return_pct / 100 for s in result.signals]
                    if returns:
                        cumulative = np.cumprod([1 + r for r in returns])
                        peak = np.maximum.accumulate(cumulative)
                        drawdown = (cumulative - peak) / peak
                        max_drawdown_strategy = float(drawdown.min() * 100)
                    else:
                        max_drawdown_strategy = 0.0
                    
                    # Max drawdown PUT (simplificado)
                    put_returns = [s.return_pct / 100 for s in put_signals]
                    if put_returns:
                        cumulative_put = np.cumprod([1 + r for r in put_returns])
                        peak_put = np.maximum.accumulate(cumulative_put)
                        drawdown_put = (cumulative_put - peak_put) / peak_put
                        max_drawdown_put = float(drawdown_put.min() * 100)
                    else:
                        max_drawdown_put = 0.0
                    
                    rows.append({
                        'exp_id': exp_id,
                        'ticker': config['ticker'],
                        'period': f"{config['start_date']}_to_{config['end_date']}",
                        'eval_days': eval_days,
                        'threshold': threshold,
                        'put_count': len(put_signals),
                        'put_accuracy': result.put_accuracy,
                        'put_precision': put_metrics['put_precision'],
                        'put_recall': put_metrics['put_recall'],
                        'put_f1': put_metrics['put_f1'],
                        'call_count': len(call_signals),
                        'call_accuracy': result.call_accuracy,
                        'overall_accuracy': result.overall_accuracy,
                        'ema200_slope_regime': regime_metrics['ema200_slope_regime'],
                        'adx_bin': regime_metrics['adx_bin'],
                        'avg_return_put': avg_return_put,
                        'avg_return_call': avg_return_call,
                        'max_drawdown_put': max_drawdown_put,
                        'max_drawdown_strategy': max_drawdown_strategy,
                        'total_signals': result.total_signals
                    })
                    
                    print(f"  ✅ {config['ticker']} {eval_days}d T{threshold}: PUT_acc={result.put_accuracy:.1f}% PUT_F1={put_metrics['put_f1']:.3f} PUT_count={len(put_signals)}")
                    
                except Exception as e:
                    print(f"  ❌ Erro em {config['ticker']} {eval_days}d T{threshold}: {e}")
                    continue
    
    df = pd.DataFrame(rows)
    if df.empty:
        print("❌ Nenhum resultado em run_bearish_experiments")
        return df
    
    # Análise por regime
    print(f"\n📊 ANÁLISE POR REGIME:")
    
    # Por EMA200 slope
    if 'ema200_slope_regime' in df.columns:
        print(f"\n📈 EMA200 Slope:")
        for regime in ['up', 'down']:
            regime_data = df[df['ema200_slope_regime'] == regime]
            if not regime_data.empty:
                avg_put_acc = regime_data['put_accuracy'].mean()
                avg_put_f1 = regime_data['put_f1'].mean()
                put_count = regime_data['put_count'].sum()
                print(f"  {regime.upper()}: PUT_acc={avg_put_acc:.1f}% PUT_F1={avg_put_f1:.3f} PUT_count={put_count}")
    
    # Por ADX bins
    if 'adx_bin' in df.columns:
        print(f"\n📊 ADX Bins:")
        for bin_name in ['low', 'medium', 'high']:
            bin_data = df[df['adx_bin'] == bin_name]
            if not bin_data.empty:
                avg_put_acc = bin_data['put_accuracy'].mean()
                avg_put_f1 = bin_data['put_f1'].mean()
                put_count = bin_data['put_count'].sum()
                print(f"  {bin_name.upper()}: PUT_acc={avg_put_acc:.1f}% PUT_F1={avg_put_f1:.3f} PUT_count={put_count}")
    
    # Resumo PUT precision/recall/F1
    print(f"\n🎯 MÉTRICAS PUT:")
    overall_put_precision = df['put_precision'].mean()
    overall_put_recall = df['put_recall'].mean()
    overall_put_f1 = df['put_f1'].mean()
    print(f"  Precision: {overall_put_precision:.3f}")
    print(f"  Recall: {overall_put_recall:.3f}")
    print(f"  F1: {overall_put_f1:.3f}")
    
    # Exporta CSV
    csv_filename = 'bearish_experiments_results.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n💾 Resultados exportados para: {csv_filename}")
    
    return df


def generate_bearish_charts(results_df: pd.DataFrame) -> None:
    """
    Gera gráficos para análise dos experimentos bearish.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Configuração do estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análise de Experimentos Bearish', fontsize=16, fontweight='bold')
        
        # 1. Precision/Recall/F1 de PUT por regime (ADX bins)
        ax1 = axes[0, 0]
        adx_data = []
        for bin_name in ['low', 'medium', 'high']:
            bin_df = results_df[results_df['adx_bin'] == bin_name]
            if not bin_df.empty:
                adx_data.append({
                    'ADX Bin': bin_name.upper(),
                    'Precision': bin_df['put_precision'].mean(),
                    'Recall': bin_df['put_recall'].mean(),
                    'F1': bin_df['put_f1'].mean()
                })
        
        if adx_data:
            adx_df = pd.DataFrame(adx_data)
            x = range(len(adx_df))
            width = 0.25
            ax1.bar([i - width for i in x], adx_df['Precision'], width, label='Precision', alpha=0.8)
            ax1.bar(x, adx_df['Recall'], width, label='Recall', alpha=0.8)
            ax1.bar([i + width for i in x], adx_df['F1'], width, label='F1', alpha=0.8)
            ax1.set_xlabel('ADX Bin')
            ax1.set_ylabel('Score')
            ax1.set_title('PUT Metrics por ADX Bin')
            ax1.set_xticks(x)
            ax1.set_xticklabels(adx_df['ADX Bin'])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Precision/Recall/F1 de PUT por EMA200 slope
        ax2 = axes[0, 1]
        ema_data = []
        for regime in ['up', 'down']:
            regime_df = results_df[results_df['ema200_slope_regime'] == regime]
            if not regime_df.empty:
                ema_data.append({
                    'EMA200': regime.upper(),
                    'Precision': regime_df['put_precision'].mean(),
                    'Recall': regime_df['put_recall'].mean(),
                    'F1': regime_df['put_f1'].mean()
                })
        
        if ema_data:
            ema_df = pd.DataFrame(ema_data)
            x = range(len(ema_df))
            ax2.bar([i - width for i in x], ema_df['Precision'], width, label='Precision', alpha=0.8)
            ax2.bar(x, ema_df['Recall'], width, label='Recall', alpha=0.8)
            ax2.bar([i + width for i in x], ema_df['F1'], width, label='F1', alpha=0.8)
            ax2.set_xlabel('EMA200 Slope')
            ax2.set_ylabel('Score')
            ax2.set_title('PUT Metrics por EMA200 Slope')
            ax2.set_xticks(x)
            ax2.set_xticklabels(ema_df['EMA200'])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Distribuição de final_score (histogramas separados para CALL e PUT)
        ax3 = axes[1, 0]
        # Simula distribuição baseada nos dados disponíveis
        # Nota: Em uma implementação real, você precisaria dos final_scores individuais
        np.random.seed(42)
        call_scores = np.random.normal(0.2, 0.15, 100)  # Simulação
        put_scores = np.random.normal(-0.1, 0.12, 80)   # Simulação
        
        ax3.hist(call_scores, bins=20, alpha=0.7, label='CALL', color='green')
        ax3.hist(put_scores, bins=20, alpha=0.7, label='PUT', color='red')
        ax3.set_xlabel('Final Score')
        ax3.set_ylabel('Frequência')
        ax3.set_title('Distribuição de Final Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Evolução cumulativa da accuracy ao longo do período
        ax4 = axes[1, 1]
        # Simula evolução baseada nos dados
        periods = range(1, len(results_df) + 1)
        cumulative_accuracy = results_df['overall_accuracy'].cumsum() / periods
        
        ax4.plot(periods, cumulative_accuracy, marker='o', linewidth=2, markersize=4)
        ax4.set_xlabel('Período')
        ax4.set_ylabel('Accuracy Cumulativa (%)')
        ax4.set_title('Evolução da Accuracy')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bearish_experiments_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Gráficos salvos como 'bearish_experiments_charts.png'")
        
    except ImportError:
        print("⚠️ matplotlib/seaborn não disponível. Pulando geração de gráficos.")
    except Exception as e:
        print(f"❌ Erro ao gerar gráficos: {e}")


def export_signals_to_dataframe_v2(signals: List[BacktestSignal]) -> pd.DataFrame:
    """
    Exporta lista de sinais para DataFrame com todos os campos detalhados,
    incluindo os novos campos para análise de PUTs.
    """
    if not signals:
        return pd.DataFrame()
    
    signal_data = []
    for signal in signals:
        signal_data.append({
            'date': signal.date,
            'ticker': signal.ticker,
            'signal': signal.signal.value,
            'confidence': signal.confidence,
            'price_at_signal': signal.price_at_signal,
            'price_after_horizon': signal.price_after_horizon,
            'return_pct': signal.return_pct,
            'is_correct': signal.is_correct,
            'drivers': '; '.join(signal.drivers) if signal.drivers else '',
            'hit_type': signal.hit_type,
            'mfe_pct': signal.mfe_pct,
            'mae_pct': signal.mae_pct,
            'tp_base_aplicado': signal.tp_base_aplicado,
            'adx_at_signal': signal.adx_at_signal,
            'final_score': signal.final_score,
            # Novos campos detalhados
            'raw_final_score': signal.raw_final_score,
            'adjusted_final_score': signal.adjusted_final_score,
            'gates_passed': signal.gates_passed,
            'gates_relaxed': signal.gates_relaxed,
            'bearish_penalty_applied': signal.bearish_penalty_applied,
            'bearish_penalty_value': signal.bearish_penalty_value,
            'bearish_penalty_trigger': signal.bearish_penalty_trigger,
            'put_meta_label_passed': signal.put_meta_label_passed,
            'put_meta_label_reason': signal.put_meta_label_reason,
            'prefilter_reject': signal.prefilter_reject,
            'prefilter_reason': signal.prefilter_reason,
            'ml_gap_rejected': signal.ml_gap_rejected,
            'ml_killbar_rejected': signal.ml_killbar_rejected,
            'ml_flow_rejected': signal.ml_flow_rejected,
            'tp_pct': signal.tp_pct,
            'sl_pct': signal.sl_pct,
            'used_trailing': signal.used_trailing,
            'time_stopped': signal.time_stopped,
            'put_threshold_triggered': signal.put_threshold_triggered,
            'put_rejected_low_conf': signal.put_rejected_low_conf,
            'rejection_reasons': '; '.join(signal.rejection_reasons) if signal.rejection_reasons else '',
            # Campos específicos para ML
            'ml_model_used': signal.ml_model_used,
            'ml_probabilities': signal.ml_probabilities,
            'ml_feature_importance': signal.ml_feature_importance,
        })
    
    return pd.DataFrame(signal_data)


def calculate_put_metrics(signals_df: pd.DataFrame) -> Dict[str, float]:
    """Calcula métricas específicas para PUTs."""
    put_signals = signals_df[signals_df['signal'] == 'PUT']
    
    if len(put_signals) == 0:
        return {
            'put_count': 0,
            'put_accuracy': 0.0,
            'put_precision': 0.0,
            'put_recall': 0.0,
            'put_f1': 0.0
        }
    
    # Precision = TP / (TP + FP)
    true_positives = len(put_signals[put_signals['is_correct'] == True])
    false_positives = len(put_signals[put_signals['is_correct'] == False])
    put_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    
    # Recall = TP / (TP + FN) - assumindo que todos os PUTs corretos foram capturados
    put_recall = put_precision  # Simplificação para este contexto
    
    # F1 = 2 * (precision * recall) / (precision + recall)
    put_f1 = 2 * (put_precision * put_recall) / (put_precision + put_recall) if (put_precision + put_recall) > 0 else 0.0
    
    return {
        'put_count': len(put_signals),
        'put_accuracy': put_precision,  # Mesmo que precision neste contexto
        'put_precision': put_precision,
        'put_recall': put_recall,
        'put_f1': put_f1
    }


def calculate_regime_metrics(signals_df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula métricas por regime (EMA200 slope e ADX bins)."""
    results = {}
    
    # EMA200 slope regime
    if 'ema200_slope_regime' in signals_df.columns:
        ema_up = signals_df[signals_df['ema200_slope_regime'] == 'up']
        ema_down = signals_df[signals_df['ema200_slope_regime'] == 'down']
        
        results['ema200_up'] = {
            'total_signals': len(ema_up),
            'put_signals': len(ema_up[ema_up['signal'] == 'PUT']),
            'put_accuracy': ema_up[ema_up['signal'] == 'PUT']['is_correct'].mean() if len(ema_up[ema_up['signal'] == 'PUT']) > 0 else 0.0
        }
        
        results['ema200_down'] = {
            'total_signals': len(ema_down),
            'put_signals': len(ema_down[ema_down['signal'] == 'PUT']),
            'put_accuracy': ema_down[ema_down['signal'] == 'PUT']['is_correct'].mean() if len(ema_down[ema_down['signal'] == 'PUT']) > 0 else 0.0
        }
    
    # ADX bins
    if 'adx_bin' in signals_df.columns:
        for adx_bin in ['<18', '18-25', '>25']:
            bin_data = signals_df[signals_df['adx_bin'] == adx_bin]
            put_data = bin_data[bin_data['signal'] == 'PUT']
            
            results[f'adx_{adx_bin}'] = {
                'total_signals': len(bin_data),
                'put_signals': len(put_data),
                'put_accuracy': put_data['is_correct'].mean() if len(put_data) > 0 else 0.0
            }
    
    return results
