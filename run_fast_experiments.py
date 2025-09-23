#!/usr/bin/env python3
"""
Versão OTIMIZADA dos experimentos - execução rápida focada em validar PUTs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.backtesting.backtest import BacktestEngine, BacktestProfessionalAnalyzer
from src.core.professional.professional_analysis import Direction
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache
import time

# Cache global para dados históricos
_data_cache = {}

@lru_cache(maxsize=100)
def get_cached_data(ticker: str, start_str: str, end_str: str):
    """Cache agressivo para dados históricos."""
    cache_key = f"{ticker}_{start_str}_{end_str}"
    if cache_key in _data_cache:
        return _data_cache[cache_key]
    
    from src.core.data.data import get_historical_data
    try:
        data = get_historical_data(ticker, start_str, end_str)
        _data_cache[cache_key] = data
        return data
    except Exception as e:
        print(f"❌ Erro ao buscar dados para {ticker}: {e}")
        return pd.DataFrame()

def run_single_experiment(args):
    """Executa um único experimento de forma otimizada."""
    ticker, eval_days, threshold, sl_factor, rolling_window = args
    
    try:
        # Configuração mínima
        analyzer = BacktestProfessionalAnalyzer(horizon="curto")
        analyzer.decision_threshold = threshold
        engine = BacktestEngine(success_threshold=3.0, analyzer=analyzer, horizon="curto", sl_factor=sl_factor)
        
        # Período reduzido: apenas 3 meses
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 31)  # Apenas 3 meses
        
        # Executa backtest
        result = engine.run_backtest(
            ticker=ticker,
            start_date=start,
            end_date=end,
            evaluation_days=eval_days,
            rolling_window=rolling_window,
            min_history_days=200  # Reduzido
        )
        
        # Análise rápida dos sinais
        put_signals = [s for s in result.signals if s.signal == Direction.PUT]
        call_signals = [s for s in result.signals if s.signal == Direction.CALL]
        neutral_signals = [s for s in result.signals if s.signal == Direction.NEUTRAL]
        
        # Métricas específicas de PUTs
        puts_gates_relaxed = sum(1 for s in put_signals if getattr(s, 'gates_relaxed', False))
        puts_bearish_penalty = sum(1 for s in put_signals if getattr(s, 'bearish_penalty_applied', False))
        
        return {
            'ticker': ticker,
            'eval_days': eval_days,
            'threshold': threshold,
            'sl_factor': sl_factor,
            'rolling_window': rolling_window,
            'total_signals': result.total_signals,
            'call_count': len(call_signals),
            'put_count': len(put_signals),
            'neutral_count': len(neutral_signals),
            'overall_accuracy': result.overall_accuracy,
            'call_accuracy': result.call_accuracy,
            'put_accuracy': result.put_accuracy,
            'puts_gates_relaxed': puts_gates_relaxed,
            'puts_bearish_penalty': puts_bearish_penalty,
            'success': True
        }
        
    except Exception as e:
        return {
            'ticker': ticker,
            'eval_days': eval_days,
            'threshold': threshold,
            'sl_factor': sl_factor,
            'rolling_window': rolling_window,
            'error': str(e),
            'success': False
        }

def run_fast_experiments():
    """Executa experimentos otimizados com paralelização."""
    print("🚀 INICIANDO EXPERIMENTOS OTIMIZADOS (RÁPIDOS)")
    print("=" * 60)
    
    # Configuração MÍNIMA para velocidade
    tickers = ["PETR4"]  # Apenas 1 ticker
    evaluation_days_list = [10]  # Apenas 1 eval_day
    threshold_list = [0.15, 0.25]  # Apenas 2 thresholds
    sl_factor_list = [0.6]  # Apenas 1 sl_factor
    rolling_window_list = [5]  # Apenas 1 rolling_window
    
    print(f"📋 CONFIGURAÇÃO OTIMIZADA:")
    print(f"  Tickers: {tickers}")
    print(f"  Evaluation days: {evaluation_days_list}")
    print(f"  Thresholds: {threshold_list}")
    print(f"  SL factors: {sl_factor_list}")
    print(f"  Rolling windows: {rolling_window_list}")
    print(f"  Período: Jan-Mar 2024 (3 meses)")
    print()
    
    # Gera todas as combinações
    experiments = []
    for ticker in tickers:
        for eval_days in evaluation_days_list:
            for threshold in threshold_list:
                for sl_factor in sl_factor_list:
                    for rolling_window in rolling_window_list:
                        experiments.append((ticker, eval_days, threshold, sl_factor, rolling_window))
    
    print(f"🔢 Total de experimentos: {len(experiments)}")
    print(f"⚡ Usando {mp.cpu_count()} cores para paralelização")
    print()
    
    start_time = time.time()
    results = []
    
    # Execução paralela
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # Submete todos os jobs
        future_to_exp = {executor.submit(run_single_experiment, exp): exp for exp in experiments}
        
        # Coleta resultados conforme completam
        for i, future in enumerate(as_completed(future_to_exp), 1):
            exp = future_to_exp[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    print(f"✅ {i}/{len(experiments)}: {result['ticker']} T{result['threshold']} -> "
                          f"PUTs:{result['put_count']} ({result['put_accuracy']:.1f}%) "
                          f"Gates:{result['puts_gates_relaxed']} Penalty:{result['puts_bearish_penalty']}")
                else:
                    print(f"❌ {i}/{len(experiments)}: {exp[0]} T{exp[2]} -> ERRO: {result['error']}")
                    
            except Exception as e:
                print(f"❌ {i}/{len(experiments)}: {exp[0]} T{exp[2]} -> EXCEÇÃO: {e}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n⏱️ TEMPO TOTAL: {execution_time:.1f} segundos")
    print(f"⚡ VELOCIDADE: {len(experiments)/execution_time:.1f} experimentos/segundo")
    
    # Análise dos resultados
    df = pd.DataFrame(results)
    successful_results = df[df['success'] == True]
    
    if successful_results.empty:
        print("❌ Nenhum experimento foi bem-sucedido!")
        return None
    
    print(f"\n📊 ANÁLISE DOS RESULTADOS:")
    print(f"  Experimentos bem-sucedidos: {len(successful_results)}/{len(experiments)}")
    
    # Métricas de PUTs
    total_puts = successful_results['put_count'].sum()
    total_signals = successful_results['total_signals'].sum()
    put_percentage = (total_puts / total_signals * 100) if total_signals > 0 else 0.0
    
    print(f"\n🎯 SINAIS PUT:")
    print(f"  Total de PUTs gerados: {total_puts}")
    print(f"  Percentual de PUTs: {put_percentage:.2f}%")
    
    if total_puts > 0:
        total_gates_relaxed = successful_results['puts_gates_relaxed'].sum()
        total_bearish_penalty = successful_results['puts_bearish_penalty'].sum()
        
        print(f"  PUTs com gates relaxados: {total_gates_relaxed}")
        print(f"  PUTs com bearish penalty: {total_bearish_penalty}")
        
        # Melhor configuração para PUTs
        best_put_config = successful_results.loc[successful_results['put_accuracy'].idxmax()]
        print(f"\n🏆 MELHOR CONFIGURAÇÃO PARA PUTs:")
        print(f"  Ticker: {best_put_config['ticker']}")
        print(f"  Threshold: {best_put_config['threshold']}")
        print(f"  PUT Accuracy: {best_put_config['put_accuracy']:.1f}%")
        print(f"  PUTs gerados: {best_put_config['put_count']}")
        print(f"  Gates relaxados: {best_put_config['puts_gates_relaxed']}")
        print(f"  Bearish penalty: {best_put_config['puts_bearish_penalty']}")
    else:
        print("⚠️ NENHUM PUT FOI GERADO!")
        print("   Isso indica que as mudanças ainda não estão funcionando adequadamente.")
    
    # Salva resultados
    csv_filename = 'fast_experiments_results.csv'
    successful_results.to_csv(csv_filename, index=False)
    print(f"\n💾 Resultados salvos em: {csv_filename}")
    
    return successful_results

if __name__ == "__main__":
    # Configuração para multiprocessing
    mp.set_start_method('spawn', force=True)
    run_fast_experiments()



