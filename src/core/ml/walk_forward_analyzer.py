"""
Walk-Forward Validation para modelos ML de trading.
Sistema de retreinamento automático com janela deslizante.
"""

import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .ml_analyzer import MLAnalyzer, ModelType, MLPrediction
from .ml_professional_analyzer import MLProfessionalAnalyzer


@dataclass
class WalkForwardConfig:
    """Configuração para Walk-Forward Validation."""
    training_window_months: int = 6  # Janela de treinamento em meses
    retrain_frequency_days: int = 30  # Frequência de retreinamento em dias
    min_accuracy_threshold: float = 0.35  # Threshold mínimo de accuracy
    max_retrain_attempts: int = 3  # Máximo de tentativas de retreinamento
    evaluation_days: int = 10  # Dias para avaliação
    tickers: List[str] = None  # Lista de tickers para treinamento
    
    def __post_init__(self):
        if self.tickers is None:
            self.tickers = ['PETR4', 'VALE3', 'ITUB4']


@dataclass
class WalkForwardResult:
    """Resultado de uma iteração do Walk-Forward."""
    date: datetime
    model_path: str
    training_period: Tuple[datetime, datetime]
    validation_accuracy: float
    training_accuracy: float
    retrain_reason: str
    success: bool
    error_message: Optional[str] = None


class WalkForwardAnalyzer:
    """
    Analisador com Walk-Forward Validation.
    Mantém modelo ML sempre atualizado com retreinamento automático.
    """
    
    def __init__(self, 
                 model_type: ModelType = ModelType.ENSEMBLE,
                 config: WalkForwardConfig = None,
                 base_model_path: str = "models/walk_forward_model.pkl"):
        """
        Inicializa o Walk-Forward Analyzer.
        
        Args:
            model_type: Tipo do modelo ML
            config: Configuração do Walk-Forward
            base_model_path: Caminho base para salvar modelos
        """
        self.model_type = model_type
        self.config = config or WalkForwardConfig()
        self.base_model_path = base_model_path
        self.current_model_path = base_model_path
        self.last_retrain_date = None
        self.retrain_history: List[WalkForwardResult] = []
        
        # Cria diretório se não existir
        os.makedirs(os.path.dirname(base_model_path), exist_ok=True)
        
        # Inicializa analyzer
        self.analyzer = None
        self._initialize_analyzer()
    
    def _initialize_analyzer(self):
        """Inicializa o analyzer ML."""
        try:
            if os.path.exists(self.current_model_path):
                # Carrega modelo existente
                self.analyzer = MLProfessionalAnalyzer(
                    horizon='médio',
                    ml_model_type=self.model_type,
                    ml_model_path=self.current_model_path
                )
                print(f"✅ Modelo Walk-Forward carregado: {self.current_model_path}")
            else:
                # Cria novo modelo
                self.analyzer = MLProfessionalAnalyzer(
                    horizon='médio',
                    ml_model_type=self.model_type
                )
                print(f"🆕 Novo modelo Walk-Forward criado")
        except Exception as e:
            print(f"❌ Erro ao inicializar analyzer: {e}")
            # Fallback para modelo padrão
            self.analyzer = MLProfessionalAnalyzer(
                horizon='médio',
                ml_model_type=self.model_type
            )
    
    def should_retrain(self, current_date: datetime) -> bool:
        """
        Verifica se o modelo deve ser retreinado.
        
        Args:
            current_date: Data atual
            
        Returns:
            True se deve retreinar
        """
        # Primeira execução
        if self.last_retrain_date is None:
            return True
        
        # Verifica frequência de retreinamento
        days_since_retrain = (current_date - self.last_retrain_date).days
        if days_since_retrain >= self.config.retrain_frequency_days:
            return True
        
        # Verifica performance recente
        if len(self.retrain_history) > 0:
            recent_results = [r for r in self.retrain_history 
                            if (current_date - r.date).days <= 30]
            if recent_results:
                avg_accuracy = np.mean([r.validation_accuracy for r in recent_results])
                if avg_accuracy < self.config.min_accuracy_threshold:
                    return True
        
        return False
    
    def retrain_model(self, current_date: datetime) -> WalkForwardResult:
        """
        Retreina o modelo ML com janela deslizante.
        
        Args:
            current_date: Data atual
            
        Returns:
            Resultado do retreinamento
        """
        print(f"🔄 Retreinando modelo Walk-Forward em {current_date.strftime('%Y-%m-%d')}")
        
        try:
            # Calcula período de treinamento
            end_date = current_date - timedelta(days=1)  # Exclui dia atual
            start_date = end_date - timedelta(days=self.config.training_window_months * 30)
            
            print(f"📊 Período de treinamento: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
            
            # Treina novo modelo
            ml_analyzer = MLAnalyzer(self.model_type)
            results = ml_analyzer.train_models(
                tickers=self.config.tickers,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                evaluation_days=self.config.evaluation_days
            )
            
            # Salva modelo
            timestamp = current_date.strftime('%Y%m%d_%H%M%S')
            model_path = f"{self.base_model_path.replace('.pkl', '')}_{timestamp}.pkl"
            ml_analyzer.save_models(model_path)
            
            # Atualiza analyzer
            self.analyzer.ml_analyzer = ml_analyzer
            self.current_model_path = model_path
            self.last_retrain_date = current_date
            
            # Calcula accuracy média
            avg_accuracy = np.mean([
                results.get('rf_test_accuracy', 0),
                results.get('xgb_test_accuracy', 0)
            ])
            
            result = WalkForwardResult(
                date=current_date,
                model_path=model_path,
                training_period=(start_date, end_date),
                validation_accuracy=avg_accuracy,
                training_accuracy=np.mean([
                    results.get('rf_train_accuracy', 0),
                    results.get('xgb_train_accuracy', 0)
                ]),
                retrain_reason="scheduled_retrain",
                success=True
            )
            
            self.retrain_history.append(result)
            
            print(f"✅ Retreinamento concluído - Accuracy: {avg_accuracy:.1%}")
            return result
            
        except Exception as e:
            error_msg = f"Erro no retreinamento: {str(e)}"
            print(f"❌ {error_msg}")
            
            result = WalkForwardResult(
                date=current_date,
                model_path=self.current_model_path,
                training_period=(current_date, current_date),
                validation_accuracy=0.0,
                training_accuracy=0.0,
                retrain_reason="error",
                success=False,
                error_message=error_msg
            )
            
            self.retrain_history.append(result)
            return result
    
    def analyze_ticker(self, ticker: str, analysis_date: datetime) -> Any:
        """
        Analisa um ticker com Walk-Forward Validation.
        
        Args:
            ticker: Ticker a ser analisado
            analysis_date: Data da análise
            
        Returns:
            Análise do ticker
        """
        # Verifica se deve retreinar
        if self.should_retrain(analysis_date):
            print(f"🔄 Retreinamento necessário para {analysis_date.strftime('%Y-%m-%d')}")
            self.retrain_model(analysis_date)
        
        # Usa modelo atual para análise
        return self.analyzer.analyze_ticker(ticker, analysis_date)
    
    def analyze(self, ticker: str, analysis_date: datetime) -> Any:
        """
        Alias para analyze_ticker para compatibilidade.
        """
        return self.analyze_ticker(ticker, analysis_date)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance do Walk-Forward."""
        if not self.retrain_history:
            return {"error": "Nenhum histórico de retreinamento"}
        
        successful_retrains = [r for r in self.retrain_history if r.success]
        
        if not successful_retrains:
            return {"error": "Nenhum retreinamento bem-sucedido"}
        
        return {
            "total_retrains": len(self.retrain_history),
            "successful_retrains": len(successful_retrains),
            "success_rate": len(successful_retrains) / len(self.retrain_history),
            "avg_validation_accuracy": np.mean([r.validation_accuracy for r in successful_retrains]),
            "avg_training_accuracy": np.mean([r.training_accuracy for r in successful_retrains]),
            "last_retrain_date": self.last_retrain_date.isoformat() if self.last_retrain_date else None,
            "current_model_path": self.current_model_path,
            "retrain_frequency_days": self.config.retrain_frequency_days,
            "training_window_months": self.config.training_window_months
        }
    
    def save_history(self, filepath: str = "models/walk_forward_history.pkl"):
        """Salva histórico de retreinamento."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.retrain_history, f)
        print(f"💾 Histórico salvo em: {filepath}")
    
    def load_history(self, filepath: str = "models/walk_forward_history.pkl"):
        """Carrega histórico de retreinamento."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.retrain_history = pickle.load(f)
            print(f"📂 Histórico carregado de: {filepath}")
        else:
            print(f"⚠️ Arquivo de histórico não encontrado: {filepath}")


def create_walk_forward_analyzer(
    model_type: str = "ensemble",
    training_window_months: int = 6,
    retrain_frequency_days: int = 30,
    min_accuracy_threshold: float = 0.35
) -> WalkForwardAnalyzer:
    """
    Função de conveniência para criar Walk-Forward Analyzer.
    
    Args:
        model_type: Tipo do modelo ("random_forest", "xgboost", "ensemble")
        training_window_months: Janela de treinamento em meses
        retrain_frequency_days: Frequência de retreinamento em dias
        min_accuracy_threshold: Threshold mínimo de accuracy
        
    Returns:
        WalkForwardAnalyzer configurado
    """
    model_type_enum = ModelType.ENSEMBLE
    if model_type == "random_forest":
        model_type_enum = ModelType.RANDOM_FOREST
    elif model_type == "xgboost":
        model_type_enum = ModelType.XGBOOST
    elif model_type == "ensemble":
        model_type_enum = ModelType.ENSEMBLE
    
    config = WalkForwardConfig(
        training_window_months=training_window_months,
        retrain_frequency_days=retrain_frequency_days,
        min_accuracy_threshold=min_accuracy_threshold
    )
    
    return WalkForwardAnalyzer(
        model_type=model_type_enum,
        config=config
    )