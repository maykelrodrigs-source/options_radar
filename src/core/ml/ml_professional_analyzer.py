"""
Wrapper que integra modelos ML com o ProfessionalAnalyzer.
Permite usar ML na UI e backtest de forma transparente.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from src.core.professional.professional_analysis import ProfessionalAnalyzer, Direction, ProfessionalAnalysis
from src.core.ml.ml_analyzer import MLAnalyzer, ModelType


class MLProfessionalAnalyzer(ProfessionalAnalyzer):
    """
    ProfessionalAnalyzer que usa modelos ML em vez de regras manuais.
    Mantém a mesma interface para compatibilidade com UI e backtest.
    """
    
    def __init__(self, horizon: str = "médio", decision_threshold: float = 0.20, 
                 ml_model_type: ModelType = ModelType.ENSEMBLE, 
                 ml_model_path: Optional[str] = None):
        """
        Inicializa o analyzer com modelo ML.
        
        Args:
            horizon: Horizonte de análise ("curto", "médio", "longo")
            decision_threshold: Threshold de decisão (não usado no ML)
            ml_model_type: Tipo do modelo ML (RANDOM_FOREST, XGBOOST, ENSEMBLE)
            ml_model_path: Caminho para o modelo ML (opcional)
        """
        # Não chama super().__init__ para evitar criar OpLabClient
        self.client = None  # Sem OpLab
        self.horizon = horizon
        self.params = self._get_horizon_parameters(horizon)
        self.decision_threshold = decision_threshold
        
        # Inicializa modelo ML
        self.ml_analyzer = MLAnalyzer(ml_model_type)
        
        # Carrega modelo ML
        if ml_model_path:
            self.ml_analyzer.load_models(ml_model_path)
        else:
            # Caminho padrão baseado no tipo
            if ml_model_type == ModelType.ENSEMBLE:
                default_path = "models/ensemble_model.pkl"
            elif ml_model_type == ModelType.RANDOM_FOREST:
                default_path = "models/random_forest_model_fast.pkl"
            elif ml_model_type == ModelType.XGBOOST:
                default_path = "models/xgboost_model.pkl"
            else:
                default_path = "models/ensemble_model.pkl"
            
            try:
                self.ml_analyzer.load_models(default_path)
                print(f"✅ Modelo ML carregado: {default_path}")
            except Exception as e:
                print(f"❌ Erro ao carregar modelo ML: {e}")
                raise ValueError(f"Não foi possível carregar modelo ML: {e}")
        
        # Verifica se modelo está treinado
        if not self.ml_analyzer.is_trained:
            raise ValueError("Modelo ML não foi treinado")
        
        print(f"🤖 MLProfessionalAnalyzer inicializado com {ml_model_type.value}")
    
    def analyze_ticker(self, ticker: str, analysis_date: datetime) -> ProfessionalAnalysis:
        """
        Analisa um ticker usando modelo ML.
        Mantém a mesma interface do ProfessionalAnalyzer original.
        """
        try:
            # Usa modelo ML para predição
            ml_prediction = self.ml_analyzer.predict(ticker, analysis_date)
            
            # Cria análises dummy (ML não usa essas camadas)
            from src.core.professional.professional_analysis import (
                TrendAnalysis, MomentumAnalysis, VolumeFlowAnalysis, 
                OptionsSentimentAnalysis, MacroContextAnalysis, TrendStrength
            )
            
            # Análises dummy
            trend = TrendAnalysis(
                sma_10=0.0, sma_50=0.0, sma_100=0.0, sma_200=0.0,
                golden_cross=False, death_cross=False,
                volatility_regime="NORMAL", atr_ratio=1.0,
                trend_score=0.0, trend_strength=TrendStrength.LATERAL
            )
            
            momentum = MomentumAnalysis(
                rsi_7=50.0, rsi_14=50.0, rsi_21=50.0,
                macd_histogram=0.0, macd_slope=0.0, macd_signal=0.0,
                adx=20.0, roc_10=0.0, roc_20=0.0,
                bb_position=0.0, bb_squeeze=False,
                stoch_k=50.0, stoch_d=50.0, stoch_signal="NEUTRAL",
                ema_9=0.0, ema_21=0.0, mfi_14=50.0,
                momentum_score=0.0, momentum_strength="NEUTRAL"
            )
            
            volume_flow = VolumeFlowAnalysis(
                volume_ratio_5d=1.0, volume_ratio_20d=1.0,
                obv_trend=0.0, accumulation_distribution=0.0,
                volume_score=0.0, flow_direction="NEUTRO"
            )
            
            options_sentiment = OptionsSentimentAnalysis(
                put_call_ratio=1.0, volatility_skew=0.0,
                call_volume_ratio=0.5, put_volume_ratio=0.5,
                sentiment_score=0.0, market_bias="NEUTRAL"
            )
            
            macro_context = MacroContextAnalysis(
                sector_score=0.0, macro_score=0.0, commodity_score=0.0,
                overall_context_score=0.0, context_bias="NEUTRAL"
            )
            
            # Converte para ProfessionalAnalysis
            analysis = ProfessionalAnalysis(
                ticker=ticker,
                analysis_date=analysis_date,
                current_price=0.0,  # Será preenchido pelo backtest
                trend=trend,
                momentum=momentum,
                volume_flow=volume_flow,
                options_sentiment=options_sentiment,
                macro_context=macro_context,
                direction=ml_prediction.direction,
                confidence=ml_prediction.confidence,
                final_score=ml_prediction.probabilities.get('CALL', 0) - ml_prediction.probabilities.get('PUT', 0),
                key_drivers=["ML_MODEL"],
                strategy_recommendation=f"ML {ml_prediction.model_used.value}",
                raw_final_score=ml_prediction.probabilities.get('CALL', 0) - ml_prediction.probabilities.get('PUT', 0),
                adjusted_final_score=ml_prediction.probabilities.get('CALL', 0) - ml_prediction.probabilities.get('PUT', 0),
                gates_passed=True,  # ML não usa gates tradicionais
                gates_relaxed=False,
                put_threshold_triggered=ml_prediction.direction == Direction.PUT,
                put_rejected_low_conf=False,
                put_meta_label_passed=True,
                put_meta_label_reason="ml_model",
                prefilter_reject=False,
                prefilter_reason="ml_model",
                rejection_reasons=[]
            )
            
            # Adiciona informações ML
            analysis.ml_prediction = ml_prediction
            
            return analysis
            
        except Exception as e:
            print(f"❌ Erro na análise ML de {ticker}: {e}")
            # Retorna análise neutra em caso de erro
            from src.core.professional.professional_analysis import (
                TrendAnalysis, MomentumAnalysis, VolumeFlowAnalysis, 
                OptionsSentimentAnalysis, MacroContextAnalysis, TrendStrength
            )
            
            # Análises dummy para erro
            trend = TrendAnalysis(
                sma_10=0.0, sma_50=0.0, sma_100=0.0, sma_200=0.0,
                golden_cross=False, death_cross=False,
                volatility_regime="NORMAL", atr_ratio=1.0,
                trend_score=0.0, trend_strength=TrendStrength.LATERAL
            )
            
            momentum = MomentumAnalysis(
                rsi_7=50.0, rsi_14=50.0, rsi_21=50.0,
                macd_histogram=0.0, macd_slope=0.0, macd_signal=0.0,
                adx=20.0, roc_10=0.0, roc_20=0.0,
                bb_position=0.0, bb_squeeze=False,
                stoch_k=50.0, stoch_d=50.0, stoch_signal="NEUTRAL",
                ema_9=0.0, ema_21=0.0, mfi_14=50.0,
                momentum_score=0.0, momentum_strength="NEUTRAL"
            )
            
            volume_flow = VolumeFlowAnalysis(
                volume_ratio_5d=1.0, volume_ratio_20d=1.0,
                obv_trend=0.0, accumulation_distribution=0.0,
                volume_score=0.0, flow_direction="NEUTRO"
            )
            
            options_sentiment = OptionsSentimentAnalysis(
                put_call_ratio=1.0, volatility_skew=0.0,
                call_volume_ratio=0.5, put_volume_ratio=0.5,
                sentiment_score=0.0, market_bias="NEUTRAL"
            )
            
            macro_context = MacroContextAnalysis(
                sector_score=0.0, macro_score=0.0, commodity_score=0.0,
                overall_context_score=0.0, context_bias="NEUTRAL"
            )
            
            return ProfessionalAnalysis(
                ticker=ticker,
                analysis_date=analysis_date,
                current_price=0.0,
                trend=trend,
                momentum=momentum,
                volume_flow=volume_flow,
                options_sentiment=options_sentiment,
                macro_context=macro_context,
                direction=Direction.NEUTRAL,
                confidence=0.0,
                final_score=0.0,
                key_drivers=["ML_ERROR"],
                strategy_recommendation="ML_ERROR",
                raw_final_score=0.0,
                adjusted_final_score=0.0,
                gates_passed=False,
                gates_relaxed=False,
                put_threshold_triggered=False,
                put_rejected_low_conf=True,
                put_meta_label_passed=False,
                put_meta_label_reason="ml_error",
                prefilter_reject=True,
                prefilter_reason="ml_error",
                rejection_reasons=[f"ml_error: {str(e)}"]
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo ML."""
        if not self.ml_analyzer.is_trained:
            return {"error": "Modelo não treinado"}
        
        info = {
            "model_type": self.ml_analyzer.model_type.value,
            "is_trained": self.ml_analyzer.is_trained,
            "feature_count": len(self.ml_analyzer.feature_engineer.feature_names),
            "models": {}
        }
        
        for model_type, model in self.ml_analyzer.models.items():
            info["models"][model_type.value] = {
                "training_accuracy": model.training_accuracy,
                "validation_accuracy": model.validation_accuracy,
                "training_date": model.training_date.isoformat(),
                "feature_count": len(model.feature_names)
            }
        
        return info
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Retorna importância das features do modelo ML."""
        return self.ml_analyzer.get_feature_importance()


def create_ml_analyzer(horizon: str = "médio", model_type: str = "ensemble") -> MLProfessionalAnalyzer:
    """
    Função de conveniência para criar analyzer ML.
    
    Args:
        horizon: "curto", "médio", "longo"
        model_type: "random_forest", "xgboost", "ensemble"
    """
    model_type_enum = ModelType.ENSEMBLE
    if model_type == "random_forest":
        model_type_enum = ModelType.RANDOM_FOREST
    elif model_type == "xgboost":
        model_type_enum = ModelType.XGBOOST
    elif model_type == "ensemble":
        model_type_enum = ModelType.ENSEMBLE
    
    return MLProfessionalAnalyzer(
        horizon=horizon,
        ml_model_type=model_type_enum
    )
