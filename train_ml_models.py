#!/usr/bin/env python3
"""
Script para treinar e testar modelos de Machine Learning.
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


def train_ml_models():
    """Treina modelos ML com dados históricos."""
    print("🚀 INICIANDO TREINAMENTO DE MODELOS ML")
    print("=" * 60)
    
    # Configurações de treinamento
    tickers = ["PETR4", "VALE3", "ITUB4", "BBAS3", "WEGE3", "MGLU3"]
    start_date = "2020-01-01"
    end_date = "2024-03-31"
    evaluation_days = 10
    
    print(f"📊 Configurações:")
    print(f"   • Tickers: {', '.join(tickers)}")
    print(f"   • Período: {start_date} a {end_date}")
    print(f"   • Evaluation days: {evaluation_days}")
    print()
    
    # Treina diferentes tipos de modelos
    results = {}
    
    # 1. Random Forest
    print("🌲 TREINANDO RANDOM FOREST...")
    rf_analyzer = MLAnalyzer(ModelType.RANDOM_FOREST)
    rf_results = rf_analyzer.train_models(tickers, start_date, end_date, evaluation_days)
    results['Random Forest'] = rf_results
    
    # Salva modelo
    rf_analyzer.save_models("models/random_forest_model.pkl")
    
    # 2. XGBoost
    print("\n🚀 TREINANDO XGBOOST...")
    xgb_analyzer = MLAnalyzer(ModelType.XGBOOST)
    xgb_results = xgb_analyzer.train_models(tickers, start_date, end_date, evaluation_days)
    results['XGBoost'] = xgb_results
    
    # Salva modelo
    xgb_analyzer.save_models("models/xgboost_model.pkl")
    
    # 3. Ensemble
    print("\n🎯 TREINANDO ENSEMBLE...")
    ensemble_analyzer = MLAnalyzer(ModelType.ENSEMBLE)
    ensemble_results = ensemble_analyzer.train_models(tickers, start_date, end_date, evaluation_days)
    results['Ensemble'] = ensemble_results
    
    # Salva modelo
    ensemble_analyzer.save_models("models/ensemble_model.pkl")
    
    # Exibe resultados
    print("\n📊 RESULTADOS DO TREINAMENTO:")
    print("=" * 60)
    
    for model_name, model_results in results.items():
        print(f"\n🎯 {model_name}:")
        for metric, value in model_results.items():
            print(f"   • {metric}: {value:.3f}")
    
    return results


def test_ml_models():
    """Testa modelos ML treinados."""
    print("\n🧪 TESTANDO MODELOS ML")
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
    
    # Testa predições
    test_ticker = "PETR4"
    test_date = datetime(2024, 3, 15)
    
    print(f"\n🔍 Testando predições para {test_ticker} em {test_date.strftime('%Y-%m-%d')}:")
    print("-" * 50)
    
    for model_name, analyzer in models.items():
        try:
            prediction = analyzer.predict(test_ticker, test_date)
            
            print(f"\n🎯 {model_name}:")
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
            print(f"❌ Erro na predição {model_name}: {e}")


def compare_with_original():
    """Compara performance dos modelos ML com o modelo original."""
    print("\n⚖️ COMPARANDO COM MODELO ORIGINAL")
    print("=" * 60)
    
    # Carrega modelo ensemble (melhor performance esperada)
    try:
        ensemble_analyzer = MLAnalyzer(ModelType.ENSEMBLE)
        ensemble_analyzer.load_models("models/ensemble_model.pkl")
        print("✅ Ensemble carregado para comparação")
    except Exception as e:
        print(f"❌ Erro ao carregar Ensemble: {e}")
        return
    
    # Configurações de teste
    ticker = "PETR4"
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    
    print(f"📊 Testando {ticker} de {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
    
    # Testa modelo original
    print("\n🔍 Testando modelo original...")
    try:
        from src.core.professional.professional_analysis import ProfessionalAnalyzer
        
        original_analyzer = ProfessionalAnalyzer(horizon="curto")
        original_analyzer.decision_threshold = 0.15
        
        engine = BacktestEngine(success_threshold=3.0, analyzer=original_analyzer, horizon="curto")
        original_result = engine.run_backtest(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            evaluation_days=10,
            rolling_window=5,
            min_history_days=300
        )
        
        print(f"✅ Modelo Original:")
        print(f"   • Overall accuracy: {original_result.overall_accuracy:.1f}%")
        print(f"   • CALL accuracy: {original_result.call_accuracy:.1f}%")
        print(f"   • PUT accuracy: {original_result.put_accuracy:.1f}%")
        print(f"   • Total signals: {original_result.total_signals}")
        
    except Exception as e:
        print(f"❌ Erro no modelo original: {e}")
        original_result = None
    
    # Testa modelo ML
    print("\n🤖 Testando modelo ML...")
    try:
        # Simula backtest com ML (simplificado)
        ml_predictions = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                prediction = ensemble_analyzer.predict(ticker, current_date)
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
            
            print(f"✅ Modelo ML (Ensemble):")
            print(f"   • Total predições: {total_count}")
            print(f"   • CALL: {call_count} ({call_count/total_count*100:.1f}%)")
            print(f"   • PUT: {put_count} ({put_count/total_count*100:.1f}%)")
            print(f"   • NEUTRAL: {neutral_count} ({neutral_count/total_count*100:.1f}%)")
            print(f"   • Confiança média: {avg_confidence:.1f}%")
            
            # Comparação
            if original_result:
                print(f"\n📊 COMPARAÇÃO:")
                print(f"   • Original vs ML - Total signals: {original_result.total_signals} vs {total_count}")
                print(f"   • Original vs ML - CALL %: {original_result.call_signals/original_result.total_signals*100:.1f}% vs {call_count/total_count*100:.1f}%")
                print(f"   • Original vs ML - PUT %: {original_result.put_signals/original_result.total_signals*100:.1f}% vs {put_count/total_count*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Erro no modelo ML: {e}")


def main():
    """Função principal."""
    print("🤖 MACHINE LEARNING PARA TRADING")
    print("=" * 60)
    
    # Cria diretório para modelos
    os.makedirs("models", exist_ok=True)
    
    try:
        # 1. Treina modelos
        results = train_ml_models()
        
        # 2. Testa modelos
        test_ml_models()
        
        # 3. Compara com modelo original
        compare_with_original()
        
        print("\n🎉 PROCESSO CONCLUÍDO!")
        print("=" * 60)
        print("📁 Modelos salvos em: models/")
        print("🔍 Use os modelos para predições futuras")
        
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


