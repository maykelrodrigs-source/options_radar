#!/usr/bin/env python3
"""
Script para executar backtests com modelos de Machine Learning.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any

# Configurações de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Adiciona o diretório raiz do projeto ao PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.ml.ml_analyzer import MLAnalyzer, ModelType
from src.core.backtesting.backtest import BacktestEngine, BacktestSignal
from src.core.professional.professional_analysis import Direction, ProfessionalAnalyzer


class MLBacktestEngine(BacktestEngine):
    """Engine de backtest adaptado para modelos ML."""
    
    def __init__(self, ml_analyzer: MLAnalyzer, success_threshold: float = 3.0):
        super().__init__(success_threshold=success_threshold, analyzer=None, horizon="curto")
        self.ml_analyzer = ml_analyzer
        
    def run_ml_backtest(self, ticker: str, start_date: datetime, end_date: datetime,
                       evaluation_days: int = 10, rolling_window: int = 5) -> Dict[str, Any]:
        """
        Executa backtest usando modelo ML.
        """
        print(f"🤖 Executando ML backtest para {ticker}...")
        
        # Busca dados históricos
        df = self._fetch_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if df.empty:
            raise ValueError(f"❌ Sem dados para {ticker}")
        
        signals = []
        current_date = start_date + timedelta(days=rolling_window)
        
        while current_date <= end_date:
            try:
                # Faz predição ML
                prediction = self.ml_analyzer.predict(ticker, current_date)
                
                # Cria sinal de backtest
                signal = BacktestSignal(
                    date=current_date,
                    ticker=ticker,
                    signal=prediction.direction,
                    confidence=prediction.confidence,
                    price_at_signal=0.0,  # Será preenchido depois
                    price_after_horizon=0.0,  # Será preenchido depois
                    return_pct=0.0,  # Será preenchido depois
                    is_correct=None,  # Será preenchido depois
                    drivers=[],  # ML não usa drivers tradicionais
                    hit_type='TIME',  # Será preenchido depois
                    mfe_pct=0.0,  # Será preenchido depois
                    mae_pct=0.0,  # Será preenchido depois
                    tp_base_aplicado=3.0,  # TP padrão
                    adx_at_signal=0.0,  # Será preenchido depois
                    final_score=prediction.probabilities.get('CALL', 0) - prediction.probabilities.get('PUT', 0),
                    raw_final_score=prediction.probabilities.get('CALL', 0) - prediction.probabilities.get('PUT', 0),
                    adjusted_final_score=prediction.probabilities.get('CALL', 0) - prediction.probabilities.get('PUT', 0),
                    gates_passed=True,  # ML não usa gates tradicionais
                    gates_relaxed=False,
                    bearish_penalty_applied=False,
                    bearish_penalty_value=0.0,
                    bearish_penalty_trigger="",
                    put_threshold_triggered=prediction.direction == Direction.PUT,
                    put_rejected_low_conf=False,
                    put_meta_label_passed=True,
                    put_meta_label_reason="ml_model",
                    prefilter_reject=False,
                    prefilter_reason="ml_model",
                    tp_pct=3.0,  # TP padrão
                    sl_pct=1.8,  # SL padrão
                    used_trailing=False,
                    time_stopped=False,
                    rejection_reasons=[],
                    # Campos ML específicos
                    ml_model_used=prediction.model_used.value,
                    ml_probabilities=str(prediction.probabilities),
                    ml_feature_importance=str(prediction.features_importance)
                )
                
                # Calcula resultado do trade
                entry_price, exit_price, hit_type, mfe_pct, mae_pct, tp_pct, sl_pct, used_trailing, time_stopped = self._get_trade_outcome(
                    ticker, current_date, evaluation_days, prediction.direction, 3.0, 0.6
                )
                
                # Atualiza sinal com resultados
                signal.entry_price = entry_price
                signal.exit_price = exit_price
                signal.hit_type = hit_type
                signal.mfe_pct = mfe_pct
                signal.mae_pct = mae_pct
                signal.tp_pct = tp_pct
                signal.sl_pct = sl_pct
                signal.used_trailing = used_trailing
                signal.time_stopped = time_stopped
                
                # Calcula se foi correto
                if entry_price and exit_price:
                    if prediction.direction == Direction.CALL:
                        signal.is_correct = exit_price > entry_price
                    elif prediction.direction == Direction.PUT:
                        signal.is_correct = exit_price < entry_price
                    else:
                        signal.is_correct = None
                else:
                    signal.is_correct = None
                
                signals.append(signal)
                
            except Exception as e:
                print(f"⚠️ Erro na predição para {current_date}: {e}")
            
            current_date += timedelta(days=rolling_window)
        
        # Calcula métricas
        return self._calculate_metrics(signals)
    
    def _calculate_metrics(self, signals: List[BacktestSignal]) -> Dict[str, Any]:
        """Calcula métricas do backtest."""
        if not signals:
            return {
                'total_signals': 0,
                'overall_accuracy': 0.0,
                'call_accuracy': 0.0,
                'put_accuracy': 0.0,
                'call_signals': 0,
                'put_signals': 0,
                'neutral_signals': 0
            }
        
        total_signals = len(signals)
        call_signals = [s for s in signals if s.direction == Direction.CALL]
        put_signals = [s for s in signals if s.direction == Direction.PUT]
        neutral_signals = [s for s in signals if s.direction == Direction.NEUTRAL]
        
        # Calcula acurácias
        correct_signals = [s for s in signals if s.is_correct is True]
        overall_accuracy = len(correct_signals) / total_signals * 100 if total_signals > 0 else 0.0
        
        call_correct = [s for s in call_signals if s.is_correct is True]
        call_accuracy = len(call_correct) / len(call_signals) * 100 if call_signals else 0.0
        
        put_correct = [s for s in put_signals if s.is_correct is True]
        put_accuracy = len(put_correct) / len(put_signals) * 100 if put_signals else 0.0
        
        return {
            'total_signals': total_signals,
            'overall_accuracy': overall_accuracy,
            'call_accuracy': call_accuracy,
            'put_accuracy': put_accuracy,
            'call_signals': len(call_signals),
            'put_signals': len(put_signals),
            'neutral_signals': len(neutral_signals),
            'signals': signals
        }


def run_ml_experiments():
    """Executa experimentos com modelos ML."""
    print("🤖 EXECUTANDO EXPERIMENTOS ML")
    print("=" * 60)
    
    # Carrega modelos
    models = {}
    
    try:
        rf_analyzer = MLAnalyzer(ModelType.RANDOM_FOREST)
        rf_analyzer.load_models("models/random_forest_model.pkl")
        models['Random Forest'] = rf_analyzer
        print("✅ Random Forest carregado")
    except Exception as e:
        print(f"❌ Erro ao carregar Random Forest: {e}")
    
    try:
        xgb_analyzer = MLAnalyzer(ModelType.XGBOOST)
        xgb_analyzer.load_models("models/xgboost_model.pkl")
        models['XGBoost'] = xgb_analyzer
        print("✅ XGBoost carregado")
    except Exception as e:
        print(f"❌ Erro ao carregar XGBoost: {e}")
    
    try:
        ensemble_analyzer = MLAnalyzer(ModelType.ENSEMBLE)
        ensemble_analyzer.load_models("models/ensemble_model.pkl")
        models['Ensemble'] = ensemble_analyzer
        print("✅ Ensemble carregado")
    except Exception as e:
        print(f"❌ Erro ao carregar Ensemble: {e}")
    
    if not models:
        print("❌ Nenhum modelo carregado com sucesso")
        return
    
    # Configurações de teste
    tickers = ["PETR4", "VALE3", "ITUB4"]
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    evaluation_days = 10
    
    print(f"\n📊 Configurações:")
    print(f"   • Tickers: {', '.join(tickers)}")
    print(f"   • Período: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
    print(f"   • Evaluation days: {evaluation_days}")
    
    # Executa experimentos
    results = []
    
    for ticker in tickers:
        print(f"\n🔍 Testando {ticker}...")
        
        for model_name, analyzer in models.items():
            try:
                print(f"   🤖 {model_name}...")
                
                engine = MLBacktestEngine(analyzer)
                result = engine.run_ml_backtest(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    evaluation_days=evaluation_days,
                    rolling_window=5
                )
                
                # Adiciona metadados
                result['ticker'] = ticker
                result['model'] = model_name
                result['evaluation_days'] = evaluation_days
                result['start_date'] = start_date.strftime('%Y-%m-%d')
                result['end_date'] = end_date.strftime('%Y-%m-%d')
                
                results.append(result)
                
                print(f"      ✅ {model_name}: {result['overall_accuracy']:.1f}% overall, {result['call_accuracy']:.1f}% CALL, {result['put_accuracy']:.1f}% PUT")
                
            except Exception as e:
                print(f"      ❌ Erro em {model_name}: {e}")
    
    # Salva resultados
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv("results/ml_experiments_results.csv", index=False)
        print(f"\n💾 Resultados salvos em: results/ml_experiments_results.csv")
        
        # Análise dos resultados
        print(f"\n📊 ANÁLISE DOS RESULTADOS ML:")
        print("=" * 60)
        
        for model_name in models.keys():
            model_results = df_results[df_results['model'] == model_name]
            if not model_results.empty:
                avg_overall = model_results['overall_accuracy'].mean()
                avg_call = model_results['call_accuracy'].mean()
                avg_put = model_results['put_accuracy'].mean()
                total_signals = model_results['total_signals'].sum()
                
                print(f"\n🎯 {model_name}:")
                print(f"   • Overall accuracy: {avg_overall:.1f}%")
                print(f"   • CALL accuracy: {avg_call:.1f}%")
                print(f"   • PUT accuracy: {avg_put:.1f}%")
                print(f"   • Total signals: {total_signals}")
        
        # Melhor modelo
        best_model = df_results.loc[df_results['overall_accuracy'].idxmax()]
        print(f"\n🏆 MELHOR MODELO:")
        print(f"   • Modelo: {best_model['model']}")
        print(f"   • Ticker: {best_model['ticker']}")
        print(f"   • Overall accuracy: {best_model['overall_accuracy']:.1f}%")
        print(f"   • CALL accuracy: {best_model['call_accuracy']:.1f}%")
        print(f"   • PUT accuracy: {best_model['put_accuracy']:.1f}%")


def compare_ml_vs_original():
    """Compara performance ML vs modelo original."""
    print("\n⚖️ COMPARANDO ML vs MODELO ORIGINAL")
    print("=" * 60)
    
    # Carrega melhor modelo ML
    try:
        ensemble_analyzer = MLAnalyzer(ModelType.ENSEMBLE)
        ensemble_analyzer.load_models("models/ensemble_model.pkl")
        print("✅ Ensemble carregado")
    except Exception as e:
        print(f"❌ Erro ao carregar Ensemble: {e}")
        return
    
    # Configurações de teste
    ticker = "PETR4"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    evaluation_days = 10
    
    print(f"📊 Testando {ticker} de {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
    
    # Testa modelo original
    print("\n🔍 Modelo Original...")
    try:
        original_analyzer = ProfessionalAnalyzer(horizon="curto")
        original_analyzer.decision_threshold = 0.15
        
        engine = BacktestEngine(success_threshold=3.0, analyzer=original_analyzer, horizon="curto")
        original_result = engine.run_backtest(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            evaluation_days=evaluation_days,
            rolling_window=5,
            min_history_days=300
        )
        
        print(f"✅ Original: {original_result.overall_accuracy:.1f}% overall, {original_result.call_accuracy:.1f}% CALL, {original_result.put_accuracy:.1f}% PUT")
        
    except Exception as e:
        print(f"❌ Erro no modelo original: {e}")
        original_result = None
    
    # Testa modelo ML
    print("\n🤖 Modelo ML (Ensemble)...")
    try:
        ml_engine = MLBacktestEngine(ensemble_analyzer)
        ml_result = ml_engine.run_ml_backtest(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            evaluation_days=evaluation_days,
            rolling_window=5
        )
        
        print(f"✅ ML: {ml_result['overall_accuracy']:.1f}% overall, {ml_result['call_accuracy']:.1f}% CALL, {ml_result['put_accuracy']:.1f}% PUT")
        
        # Comparação
        if original_result:
            print(f"\n📊 COMPARAÇÃO:")
            print(f"   • Overall: {original_result.overall_accuracy:.1f}% vs {ml_result['overall_accuracy']:.1f}%")
            print(f"   • CALL: {original_result.call_accuracy:.1f}% vs {ml_result['call_accuracy']:.1f}%")
            print(f"   • PUT: {original_result.put_accuracy:.1f}% vs {ml_result['put_accuracy']:.1f}%")
            print(f"   • Total signals: {original_result.total_signals} vs {ml_result['total_signals']}")
            
            # Melhoria
            overall_improvement = ml_result['overall_accuracy'] - original_result.overall_accuracy
            call_improvement = ml_result['call_accuracy'] - original_result.call_accuracy
            put_improvement = ml_result['put_accuracy'] - original_result.put_accuracy
            
            print(f"\n📈 MELHORIA:")
            print(f"   • Overall: {overall_improvement:+.1f} pontos percentuais")
            print(f"   • CALL: {call_improvement:+.1f} pontos percentuais")
            print(f"   • PUT: {put_improvement:+.1f} pontos percentuais")
            
            if overall_improvement > 0:
                print("🎉 ML superou o modelo original!")
            else:
                print("⚠️ ML não superou o modelo original")
        
    except Exception as e:
        print(f"❌ Erro no modelo ML: {e}")


def main():
    """Função principal."""
    print("🤖 BACKTESTS COM MACHINE LEARNING")
    print("=" * 60)
    
    # Cria diretório para resultados
    os.makedirs("results", exist_ok=True)
    
    try:
        # 1. Executa experimentos ML
        run_ml_experiments()
        
        # 2. Compara com modelo original
        compare_ml_vs_original()
        
        print("\n🎉 PROCESSO CONCLUÍDO!")
        print("=" * 60)
        print("📁 Resultados salvos em: results/ml_experiments_results.csv")
        print("🔍 Analise os resultados para validar a performance")
        
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
