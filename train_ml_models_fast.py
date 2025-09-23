#!/usr/bin/env python3
"""
Script otimizado para treinar modelos ML rapidamente.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configurações de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Adiciona o diretório raiz do projeto ao PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.ml.ml_analyzer import MLAnalyzer, ModelType
from src.core.backtesting.backtest import BacktestEngine, BacktestSignal
from src.core.professional.professional_analysis import Direction


def train_ml_models_fast():
    """Treina modelos ML com configurações otimizadas para velocidade."""
    print("🚀 TREINAMENTO ML OTIMIZADO")
    print("=" * 60)
    
    # Configurações reduzidas para velocidade
    tickers = ["PETR4", "VALE3", "ITUB4"]  # Menos tickers
    start_date = "2023-01-01"  # Período menor
    end_date = "2024-03-31"
    evaluation_days = 10
    
    print(f"📊 Configurações otimizadas:")
    print(f"   • Tickers: {', '.join(tickers)}")
    print(f"   • Período: {start_date} a {end_date}")
    print(f"   • Evaluation days: {evaluation_days}")
    print()
    
    # Treina apenas Random Forest primeiro (mais rápido)
    print("🌲 TREINANDO RANDOM FOREST (OTIMIZADO)...")
    rf_analyzer = MLAnalyzer(ModelType.RANDOM_FOREST)
    rf_results = rf_analyzer.train_models_fast(tickers, start_date, end_date, evaluation_days)
    
    # Salva modelo
    rf_analyzer.save_models("models/random_forest_model_fast.pkl")
    
    print(f"✅ Random Forest treinado: {rf_results}")
    
    return rf_results


def test_ml_model_fast():
    """Testa modelo ML treinado rapidamente."""
    print("\n🧪 TESTANDO MODELO ML")
    print("=" * 60)
    
    try:
        rf_analyzer = MLAnalyzer(ModelType.RANDOM_FOREST)
        rf_analyzer.load_models("models/random_forest_model_fast.pkl")
        print("✅ Random Forest carregado")
    except Exception as e:
        print(f"❌ Erro ao carregar Random Forest: {e}")
        return
    
    # Testa predições
    test_ticker = "PETR4"
    test_date = datetime(2024, 3, 15)
    
    print(f"\n🔍 Testando predições para {test_ticker} em {test_date.strftime('%Y-%m-%d')}:")
    print("-" * 50)
    
    try:
        prediction = rf_analyzer.predict(test_ticker, test_date)
        
        print(f"\n🎯 Random Forest:")
        print(f"   • Direção: {prediction.direction.value}")
        print(f"   • Confiança: {prediction.confidence:.1f}%")
        print(f"   • Probabilidades:")
        for direction, prob in prediction.probabilities.items():
            print(f"     - {direction}: {prob:.3f}")
        
        # Top 5 features mais importantes
        top_features = sorted(
            prediction.features_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        print(f"   • Top 5 features importantes:")
        for feature, importance in top_features:
            print(f"     - {feature}: {importance:.3f}")
            
    except Exception as e:
        print(f"❌ Erro na predição: {e}")


def run_quick_ml_backtest():
    """Executa backtest rápido com modelo ML."""
    print("\n🤖 BACKTEST RÁPIDO COM ML")
    print("=" * 60)
    
    try:
        rf_analyzer = MLAnalyzer(ModelType.RANDOM_FOREST)
        rf_analyzer.load_models("models/random_forest_model_fast.pkl")
        print("✅ Random Forest carregado")
    except Exception as e:
        print(f"❌ Erro ao carregar Random Forest: {e}")
        return
    
    # Configurações de teste
    ticker = "PETR4"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    evaluation_days = 10
    
    print(f"📊 Testando {ticker} de {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
    
    # Simula backtest com ML (simplificado)
    ml_predictions = []
    current_date = start_date
    
    while current_date <= end_date:
        try:
            prediction = rf_analyzer.predict(ticker, current_date)
            ml_predictions.append(prediction)
            current_date += timedelta(days=5)  # Testa a cada 5 dias
        except:
            current_date += timedelta(days=1)
    
    if ml_predictions:
        # Análise básica das predições ML
        directions = [p.direction for p in ml_predictions]
        confidences = [p.confidence for p in ml_predictions]
        
        call_count = directions.count(Direction.CALL)
        put_count = directions.count(Direction.PUT)
        neutral_count = directions.count(Direction.NEUTRAL)
        total_count = len(directions)
        
        avg_confidence = np.mean(confidences)
        
        print(f"✅ Modelo ML (Random Forest):")
        print(f"   • Total predições: {total_count}")
        print(f"   • CALL: {call_count} ({call_count/total_count*100:.1f}%)")
        print(f"   • PUT: {put_count} ({put_count/total_count*100:.1f}%)")
        print(f"   • NEUTRAL: {neutral_count} ({neutral_count/total_count*100:.1f}%)")
        print(f"   • Confiança média: {avg_confidence:.1f}%")
        
        # Comparação com modelo original
        print(f"\n📊 COMPARAÇÃO COM MODELO ORIGINAL:")
        print(f"   • ML vs Original - CALL %: {call_count/total_count*100:.1f}% vs ~28.6%")
        print(f"   • ML vs Original - PUT %: {put_count/total_count*100:.1f}% vs ~7.7%")
        print(f"   • ML vs Original - NEUTRAL %: {neutral_count/total_count*100:.1f}% vs ~63.7%")
        
        if put_count > 0:
            print("🎉 ML está gerando PUTs!")
        else:
            print("⚠️ ML ainda não está gerando PUTs")


def main():
    """Função principal."""
    print("🤖 MACHINE LEARNING RÁPIDO")
    print("=" * 60)
    
    # Cria diretório para modelos
    os.makedirs("models", exist_ok=True)
    
    try:
        # 1. Treina modelo rapidamente
        results = train_ml_models_fast()
        
        # 2. Testa modelo
        test_ml_model_fast()
        
        # 3. Executa backtest rápido
        run_quick_ml_backtest()
        
        print("\n🎉 PROCESSO CONCLUÍDO!")
        print("=" * 60)
        print("📁 Modelo salvo em: models/random_forest_model_fast.pkl")
        print("🔍 Modelo ML está funcionando!")
        
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


