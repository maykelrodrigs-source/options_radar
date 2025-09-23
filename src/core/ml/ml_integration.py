"""
Integração final do ML com UI e Backtest.
Fornece interface simples para usar modelos ML no sistema.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from .ml_professional_analyzer import MLProfessionalAnalyzer, create_ml_analyzer
from .walk_forward_analyzer import WalkForwardAnalyzer, create_walk_forward_analyzer
from .ml_analyzer import ModelType


class MLIntegration:
    """
    Classe de integração que fornece interface simples para usar ML.
    Escolhe automaticamente entre modelo estático e Walk-Forward.
    """
    
    def __init__(self, 
                 use_walk_forward: bool = True,
                 model_type: str = "ensemble",
                 walk_forward_config: Dict[str, Any] = None):
        """
        Inicializa a integração ML.
        
        Args:
            use_walk_forward: Se deve usar Walk-Forward Validation
            model_type: Tipo do modelo ("random_forest", "xgboost", "ensemble")
            walk_forward_config: Configuração do Walk-Forward
        """
        self.use_walk_forward = use_walk_forward
        self.model_type = model_type
        
        if use_walk_forward:
            # Configuração padrão do Walk-Forward
            default_config = {
                'training_window_months': 12,  # Janela maior para evitar dataset vazio
                'retrain_frequency_days': 30,
                'min_accuracy_threshold': 0.35
            }
            if walk_forward_config:
                default_config.update(walk_forward_config)
            
            self.analyzer = create_walk_forward_analyzer(
                model_type=model_type,
                **default_config
            )
            print(f"🤖 ML Integration: Walk-Forward ativado ({model_type})")
        else:
            # Modelo estático
            self.analyzer = create_ml_analyzer(
                horizon='médio',
                model_type=model_type
            )
            print(f"🤖 ML Integration: Modelo estático ({model_type})")
    
    def analyze_ticker(self, ticker: str, analysis_date: datetime) -> Any:
        """
        Analisa um ticker usando ML.
        
        Args:
            ticker: Ticker a ser analisado
            analysis_date: Data da análise
            
        Returns:
            Análise do ticker
        """
        return self.analyzer.analyze_ticker(ticker, analysis_date)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo ML."""
        if self.use_walk_forward:
            return self.analyzer.get_performance_stats()
        else:
            return self.analyzer.get_model_info()
    
    def get_accuracy_summary(self) -> Dict[str, float]:
        """Retorna resumo de accuracy do modelo."""
        info = self.get_model_info()
        
        if self.use_walk_forward:
            return {
                'validation_accuracy': info.get('avg_validation_accuracy', 0.0),
                'training_accuracy': info.get('avg_training_accuracy', 0.0),
                'success_rate': info.get('success_rate', 0.0)
            }
        else:
            models = info.get('models', {})
            if models:
                # Média das accuracies dos modelos
                accuracies = [model_info['validation_accuracy'] for model_info in models.values()]
                return {
                    'validation_accuracy': sum(accuracies) / len(accuracies),
                    'training_accuracy': sum([model_info['training_accuracy'] for model_info in models.values()]) / len(models),
                    'success_rate': 1.0  # Modelo estático sempre "funciona"
                }
            else:
                return {
                    'validation_accuracy': 0.0,
                    'training_accuracy': 0.0,
                    'success_rate': 0.0
                }


# Funções de conveniência para uso direto
def create_ml_integration(use_walk_forward: bool = True, 
                         model_type: str = "ensemble") -> MLIntegration:
    """
    Cria integração ML para uso na UI e backtest.
    
    Args:
        use_walk_forward: Se deve usar Walk-Forward Validation
        model_type: Tipo do modelo
        
    Returns:
        MLIntegration configurada
    """
    return MLIntegration(
        use_walk_forward=use_walk_forward,
        model_type=model_type
    )


def get_best_ml_analyzer() -> MLIntegration:
    """
    Retorna o melhor analyzer ML disponível.
    Tenta Walk-Forward primeiro, fallback para estático.
    """
    try:
        # Tenta Walk-Forward primeiro
        return create_ml_integration(use_walk_forward=True, model_type="ensemble")
    except Exception as e:
        print(f"⚠️ Walk-Forward falhou, usando modelo estático: {e}")
        # Fallback para modelo estático
        return create_ml_integration(use_walk_forward=False, model_type="ensemble")


# Para compatibilidade com código existente
def create_ml_professional_analyzer(model_type: str = "ensemble", 
                                   use_walk_forward: bool = True) -> Any:
    """
    Função de compatibilidade que retorna analyzer ML.
    Substitui create_ml_analyzer() em código existente.
    """
    integration = create_ml_integration(
        use_walk_forward=use_walk_forward,
        model_type=model_type
    )
    
    if use_walk_forward:
        return integration.analyzer
    else:
        return integration.analyzer
