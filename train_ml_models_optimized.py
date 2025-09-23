#!/usr/bin/env python3
"""
Script para treinar modelos ML com hiperparâmetros otimizados
Loop 3: Otimização de hiperparâmetros
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.ml.ml_analyzer import MLAnalyzer, ModelType
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

def train_optimized_models():
    """Treina modelos com hiperparâmetros otimizados"""
    
    print('🔄 LOOP 3: OTIMIZANDO HIPERPARÂMETROS')
    print('=' * 60)
    
    # Configurações
    tickers = ['PETR4', 'VALE3', 'ITUB4']
    start_date = '2023-01-01'
    end_date = '2024-12-31'
    evaluation_days = 10
    
    print('📊 Configurações Loop 3:')
    print('   • Random Forest: 300 árvores, profundidade 20')
    print('   • XGBoost: 500 estimadores, learning rate 0.05')
    print('   • Treinamento direto (sem GridSearchCV)')
    
    # Cria analyzer
    ensemble_analyzer = MLAnalyzer(ModelType.ENSEMBLE)
    
    # Treina modelos
    print('\n🤖 Treinando modelos com hiperparâmetros otimizados...')
    results = ensemble_analyzer.train_models(tickers, start_date, end_date, evaluation_days)
    
    # Modifica hiperparâmetros após treinamento inicial
    print('\n🔧 Aplicando hiperparâmetros otimizados...')
    
    # Random Forest otimizado
    if ModelType.RANDOM_FOREST in ensemble_analyzer.models:
        print('🌲 Otimizando Random Forest...')
        rf_optimized = RandomForestClassifier(
            n_estimators=300,      # Mais árvores
            max_depth=20,          # Profundidade maior
            min_samples_split=3,   # Mais sensível
            min_samples_leaf=1,    # Mais sensível
            max_features='sqrt',   # Feature selection
            random_state=42,
            n_jobs=-1
        )
        
        # Treina com dados existentes
        X_train = ensemble_analyzer.feature_engineer.X_train
        y_train = ensemble_analyzer.feature_engineer.y_train
        X_test = ensemble_analyzer.feature_engineer.X_test
        y_test = ensemble_analyzer.feature_engineer.y_test
        
        rf_optimized.fit(X_train, y_train)
        
        train_score = rf_optimized.score(X_train, y_train)
        test_score = rf_optimized.score(X_test, y_test)
        
        # Atualiza modelo
        ensemble_analyzer.models[ModelType.RANDOM_FOREST].model = rf_optimized
        ensemble_analyzer.models[ModelType.RANDOM_FOREST].training_accuracy = train_score
        ensemble_analyzer.models[ModelType.RANDOM_FOREST].validation_accuracy = test_score
        
        print(f'✅ Random Forest otimizado - Train: {train_score:.3f}, Test: {test_score:.3f}')
    
    # XGBoost otimizado
    if ModelType.XGBOOST in ensemble_analyzer.models:
        print('🚀 Otimizando XGBoost...')
        xgb_optimized = XGBClassifier(
            n_estimators=500,      # Mais estimadores
            max_depth=8,           # Profundidade maior
            learning_rate=0.05,    # Learning rate menor
            subsample=0.8,         # Subsample
            colsample_bytree=0.8,  # Feature sampling
            random_state=42,
            n_jobs=-1
        )
        
        xgb_optimized.fit(X_train, y_train)
        
        train_score = xgb_optimized.score(X_train, y_train)
        test_score = xgb_optimized.score(X_test, y_test)
        
        # Atualiza modelo
        ensemble_analyzer.models[ModelType.XGBOOST].model = xgb_optimized
        ensemble_analyzer.models[ModelType.XGBOOST].training_accuracy = train_score
        ensemble_analyzer.models[ModelType.XGBOOST].validation_accuracy = test_score
        
        print(f'✅ XGBoost otimizado - Train: {train_score:.3f}, Test: {test_score:.3f}')
    
    # Retreina ensemble
    print('\n🎯 Retreinando Ensemble...')
    ensemble_results = ensemble_analyzer._train_ensemble()
    
    print(f'\n📊 RESULTADOS LOOP 3 (HIPERPARÂMETROS OTIMIZADOS):')
    print(f'   • RF Train: {ensemble_analyzer.models[ModelType.RANDOM_FOREST].training_accuracy:.3f}')
    print(f'   • RF Test: {ensemble_analyzer.models[ModelType.RANDOM_FOREST].validation_accuracy:.3f}')
    print(f'   • XGB Train: {ensemble_analyzer.models[ModelType.XGBOOST].training_accuracy:.3f}')
    print(f'   • XGB Test: {ensemble_analyzer.models[ModelType.XGBOOST].validation_accuracy:.3f}')
    
    # Salva modelo otimizado v2
    model_path = 'models/ensemble_model_optimized_v2.pkl'
    ensemble_analyzer.save_models(model_path)
    print(f'\n💾 Modelo otimizado v2 salvo: {model_path}')
    
    print('\n✅ LOOP 3 CONCLUÍDO!')
    print('   • Hiperparâmetros otimizados aplicados')
    print('   • Próximo: testar performance')
    
    return ensemble_analyzer

if __name__ == '__main__':
    train_optimized_models()
