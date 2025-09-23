#!/usr/bin/env python3
"""
Script para executar backtests em cenários bearish históricos.
Foca em períodos de queda acentuada para validar ajustes de PUT.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os
from typing import List, Dict, Any

# Configurações de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Adiciona o diretório raiz do projeto ao PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
if project_root not in os.sys.path:
    os.sys.path.insert(0, project_root)

from src.core.backtesting.backtest import run_experiments_v2, BacktestEngine, BacktestSignal
from src.core.professional.professional_analysis import ProfessionalAnalyzer

# --- Configurações dos Experimentos Bearish ---

BEARISH_PERIODS = [
    {
        "name": "PETR4_2022_BEAR_MARKET",
        "ticker": "PETR4", 
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
        "description": "Período de queda acentuada da PETR4 em 2022"
    },
    {
        "name": "VALE3_2015_COMMODITIES_CRISIS", 
        "ticker": "VALE3",
        "start_date": "2015-01-01", 
        "end_date": "2015-12-31",
        "description": "Crise de commodities da VALE3 em 2015"
    }
]

EXPERIMENT_CONFIG = {
    "evaluation_days": [5, 10, 20],
    "decision_thresholds": [0.15, 0.20],
    "sl_factors": [0.6],
    "rolling_windows": [5],
    "max_workers": 8
}

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

def run_bearish_experiment(period_config: dict, experiment_config: dict) -> dict:
    """
    Executa experimento para um período bearish específico.
    """
    logger.info(f"🚀 Iniciando experimento bearish: {period_config['name']}")
    logger.info(f"📅 Período: {period_config['start_date']} a {period_config['end_date']}")
    logger.info(f"📊 Descrição: {period_config['description']}")
    
    start_time = time.time()
    
    try:
        # Executa backtest otimizado
        results = run_experiments_v2(
            tickers=[period_config['ticker']],
            start_date=period_config['start_date'],
            end_date=period_config['end_date'],
            evaluation_days_list=experiment_config['evaluation_days'],
            decision_threshold_list=experiment_config['decision_thresholds'],
            sl_factor_list=experiment_config['sl_factors'],
            rolling_window_list=experiment_config['rolling_windows'],
            max_workers=experiment_config['max_workers']
        )
        
        execution_time = time.time() - start_time
        
        if not results.empty:
            # Adiciona informações do período
            results['period_name'] = period_config['name']
            results['ticker'] = period_config['ticker']
            results['start_date'] = period_config['start_date']
            results['end_date'] = period_config['end_date']
            
            logger.info(f"✅ Experimento {period_config['name']} concluído em {execution_time:.1f}s")
            logger.info(f"📊 Total de configurações testadas: {len(results)}")
            
            return {
                "name": period_config['name'],
                "status": "success", 
                "results": results,
                "time": execution_time
            }
        else:
            logger.error(f"❌ Experimento {period_config['name']} não gerou resultados")
            return {
                "name": period_config['name'],
                "status": "no_results",
                "time": execution_time
            }
            
    except Exception as e:
        logger.error(f"❌ Erro no experimento {period_config['name']}: {e}")
        return {
            "name": period_config['name'],
            "status": "error",
            "error": str(e),
            "time": time.time() - start_time
        }

def analyze_bearish_results(all_results_df: pd.DataFrame) -> None:
    """
    Analisa e exibe resultados consolidados dos experimentos bearish.
    """
    logger.info("=" * 80)
    logger.info("📊 ANÁLISE CONSOLIDADA DOS EXPERIMENTOS BEARISH")
    logger.info("=" * 80)
    
    if all_results_df.empty:
        logger.error("❌ Nenhum resultado para analisar")
        return
    
    # Análise por período
    for period in all_results_df['period_name'].unique():
        period_data = all_results_df[all_results_df['period_name'] == period]
        ticker = period_data['ticker'].iloc[0]
        
        logger.info(f"\n🎯 PERÍODO: {period}")
        logger.info(f"📈 Ticker: {ticker}")
        logger.info(f"📅 Período: {period_data['start_date'].iloc[0]} a {period_data['end_date'].iloc[0]}")
        
        # Métricas gerais
        total_configs = len(period_data)
        avg_put_count = period_data['put_count'].mean()
        avg_put_accuracy = period_data['put_accuracy'].mean()
        avg_overall_accuracy = period_data['overall_accuracy'].mean()
        
        logger.info(f"📊 Configurações testadas: {total_configs}")
        logger.info(f"📉 PUTs gerados (média): {avg_put_count:.1f}")
        logger.info(f"🎯 PUT accuracy (média): {avg_put_accuracy:.1%}")
        logger.info(f"📈 Overall accuracy (média): {avg_overall_accuracy:.1%}")
        
        # Melhor configuração para PUTs
        best_put_config = period_data.loc[period_data['put_accuracy'].idxmax()]
        logger.info(f"🏆 Melhor PUT accuracy: {best_put_config['put_accuracy']:.1%}")
        logger.info(f"   Configuração: eval_days={best_put_config['eval_days']}, threshold={best_put_config['decision_threshold']}")
    
    # Análise por regime (se disponível)
    if 'ema200_slope_regime' in all_results_df.columns:
        logger.info(f"\n📊 ANÁLISE POR REGIME EMA200:")
        ema_regimes = all_results_df.groupby('ema200_slope_regime').agg({
            'put_count': 'mean',
            'put_accuracy': 'mean',
            'overall_accuracy': 'mean'
        }).round(3)
        
        for regime, metrics in ema_regimes.iterrows():
            logger.info(f"   {regime.upper()}: PUTs={metrics['put_count']:.1f}, PUT_acc={metrics['put_accuracy']:.1%}, Overall={metrics['overall_accuracy']:.1%}")
    
    if 'adx_bin' in all_results_df.columns:
        logger.info(f"\n📊 ANÁLISE POR ADX BINS:")
        adx_bins = all_results_df.groupby('adx_bin').agg({
            'put_count': 'mean',
            'put_accuracy': 'mean',
            'overall_accuracy': 'mean'
        }).round(3)
        
        for bin_name, metrics in adx_bins.iterrows():
            logger.info(f"   ADX {bin_name}: PUTs={metrics['put_count']:.1f}, PUT_acc={metrics['put_accuracy']:.1%}, Overall={metrics['overall_accuracy']:.1%}")

def main():
    """Função principal para executar experimentos bearish."""
    logger.info("🚀 INICIANDO EXPERIMENTOS BEARISH")
    logger.info("=" * 80)
    logger.info(f"📋 Períodos bearish a testar: {len(BEARISH_PERIODS)}")
    logger.info(f"⚡ Workers paralelos: {EXPERIMENT_CONFIG['max_workers']}")
    
    main_start_time = time.time()
    all_results_df = pd.DataFrame()
    successful_periods = []
    
    # Executa experimentos em paralelo
    with ProcessPoolExecutor(max_workers=EXPERIMENT_CONFIG['max_workers']) as executor:
        futures = {
            executor.submit(run_bearish_experiment, period_config, EXPERIMENT_CONFIG): period_config['name'] 
            for period_config in BEARISH_PERIODS
        }
        
        for future in as_completed(futures):
            period_name = futures[future]
            try:
                result = future.result()
                if result['status'] == "success":
                    all_results_df = pd.concat([all_results_df, result['results']], ignore_index=True)
                    successful_periods.append(period_name)
                else:
                    logger.error(f"❌ Erro em {period_name}: {result.get('error', 'Sem resultados')}")
            except Exception as exc:
                logger.error(f"❌ Erro em {period_name}: {exc}")
    
    # Salva resultados consolidados
    if not all_results_df.empty:
        output_filename = "bearish_experiments_results.csv"
        all_results_df.to_csv(output_filename, index=False)
        logger.info(f"💾 Resultados consolidados exportados para: {output_filename}")
        
        # Análise dos resultados
        analyze_bearish_results(all_results_df)
        
        # Resumo final
        logger.info(f"\n🎯 RESUMO FINAL:")
        logger.info(f"✅ Períodos analisados com sucesso: {', '.join(successful_periods)}")
        logger.info(f"📊 Total de configurações testadas: {len(all_results_df)}")
        logger.info(f"📉 PUTs gerados (total): {all_results_df['put_count'].sum()}")
        logger.info(f"🎯 PUT accuracy (média): {all_results_df['put_accuracy'].mean():.1%}")
        
    else:
        logger.error("❌ Nenhum período foi analisado com sucesso")
    
    total_time = time.time() - main_start_time
    logger.info(f"\n⏱️ TEMPO TOTAL: {total_time:.1f}s")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
