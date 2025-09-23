"""
Módulo de Machine Learning para análise de sinais de trading.
Implementa Random Forest, XGBoost e ensemble de modelos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import joblib
import os
from dataclasses import dataclass
from enum import Enum

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Local imports
from src.core.professional.professional_analysis import ProfessionalAnalyzer, Direction
from src.core.data.data import get_historical_data


class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"


@dataclass
class MLPrediction:
    """Resultado de predição do modelo ML."""
    direction: Direction
    confidence: float
    probabilities: Dict[str, float]
    model_used: ModelType
    features_importance: Dict[str, float]
    prediction_date: datetime


@dataclass
class MLModel:
    """Modelo de ML treinado."""
    model: Any
    model_type: ModelType
    scaler: Optional[StandardScaler]
    feature_names: List[str]
    training_accuracy: float
    validation_accuracy: float
    training_date: datetime
    metadata: Dict[str, Any]


class MLFeatureEngineer:
    """Engenheiro de features para modelos ML."""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features avançadas para ML a partir dos dados históricos.
        """
        features_df = df.copy()
        
        # 1. Features de Preço
        features_df['price_change_1d'] = df['close'].pct_change(1)
        features_df['price_change_3d'] = df['close'].pct_change(3)
        features_df['price_change_5d'] = df['close'].pct_change(5)
        features_df['price_change_10d'] = df['close'].pct_change(10)
        
        # 2. Features de Volatilidade
        features_df['volatility_5d'] = df['close'].rolling(5).std()
        features_df['volatility_10d'] = df['close'].rolling(10).std()
        features_df['volatility_20d'] = df['close'].rolling(20).std()
        
        # 3. Features de Volume
        features_df['volume_ratio_5d'] = df['volume'] / df['volume'].rolling(5).mean()
        features_df['volume_ratio_10d'] = df['volume'] / df['volume'].rolling(10).mean()
        features_df['volume_ratio_20d'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # 4. Features de Médias Móveis
        features_df['sma_5'] = df['close'].rolling(5).mean()
        features_df['sma_10'] = df['close'].rolling(10).mean()
        features_df['sma_20'] = df['close'].rolling(20).mean()
        features_df['sma_50'] = df['close'].rolling(50).mean()
        features_df['sma_200'] = df['close'].rolling(200).mean()
        
        # 5. Features de EMA
        features_df['ema_9'] = df['close'].ewm(span=9).mean()
        features_df['ema_21'] = df['close'].ewm(span=21).mean()
        features_df['ema_50'] = df['close'].ewm(span=50).mean()
        features_df['ema_200'] = df['close'].ewm(span=200).mean()
        
        # 6. Features de RSI
        features_df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        features_df['rsi_21'] = self._calculate_rsi(df['close'], 21)
        
        # 7. Features de MACD
        macd_data = self._calculate_macd(df['close'])
        features_df['macd'] = macd_data['macd']
        features_df['macd_signal'] = macd_data['signal']
        features_df['macd_histogram'] = macd_data['histogram']
        
        # 8. Features de Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df['close'], 20, 2)
        features_df['bb_upper'] = bb_data['upper']
        features_df['bb_middle'] = bb_data['middle']
        features_df['bb_lower'] = bb_data['lower']
        features_df['bb_position'] = (df['close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
        
        # 9. Features de ADX
        features_df['adx'] = self._calculate_adx(df, 14)
        
        # 10. Features de Momentum
        features_df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
        features_df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
        features_df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
        
        # 11. Features de Tendência
        features_df['trend_5d'] = (df['close'] > df['close'].shift(5)).astype(int)
        features_df['trend_10d'] = (df['close'] > df['close'].shift(10)).astype(int)
        features_df['trend_20d'] = (df['close'] > df['close'].shift(20)).astype(int)
        
        # 12. Features de Gap
        features_df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # 13. Features de Candle Patterns
        features_df['doji'] = self._is_doji(df)
        features_df['hammer'] = self._is_hammer(df)
        features_df['shooting_star'] = self._is_shooting_star(df)
        
        # 14. Features de Suporte/Resistência
        features_df['support_level'] = self._calculate_support_level(df)
        features_df['resistance_level'] = self._calculate_resistance_level(df)
        
        # 15. Features de Regime de Mercado
        features_df['market_regime'] = self._classify_market_regime(df)
        
        # Remove NaN values
        features_df = features_df.dropna()
        
        # Remove colunas não numéricas (como Timestamp)
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_columns]
        
        # Store feature names (apenas colunas numéricas)
        self.feature_names = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calcula MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calcula Bollinger Bands."""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula ADX."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Smoothed values
        atr = tr.rolling(period).mean()
        di_plus = 100 * (dm_plus.rolling(period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(period).mean() / atr)
        
        # ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _is_doji(self, df: pd.DataFrame) -> pd.Series:
        """Identifica padrão Doji."""
        body_size = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        return (body_size / total_range < 0.1).astype(int)
    
    def _is_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Identifica padrão Hammer."""
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        return ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
    
    def _is_shooting_star(self, df: pd.DataFrame) -> pd.Series:
        """Identifica padrão Shooting Star."""
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        return ((upper_shadow > 2 * body_size) & (lower_shadow < body_size)).astype(int)
    
    def _calculate_support_level(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calcula nível de suporte."""
        return df['low'].rolling(period).min()
    
    def _calculate_resistance_level(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calcula nível de resistência."""
        return df['high'].rolling(period).max()
    
    def _classify_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Classifica regime de mercado."""
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        regime = pd.Series(index=df.index, dtype=int)
        regime[df['close'] > sma_20] = 1  # Bullish
        regime[df['close'] < sma_20] = -1  # Bearish
        regime[(df['close'] > sma_20) & (sma_20 > sma_50)] = 2  # Strong Bull
        regime[(df['close'] < sma_20) & (sma_20 < sma_50)] = -2  # Strong Bear
        
        return regime


class MLAnalyzer:
    """Analisador principal de Machine Learning."""
    
    def __init__(self, model_type: ModelType = ModelType.ENSEMBLE):
        self.model_type = model_type
        self.models: Dict[ModelType, MLModel] = {}
        self.feature_engineer = MLFeatureEngineer()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def train_models(self, tickers: List[str], start_date: str, end_date: str, 
                    evaluation_days: int = 10) -> Dict[str, float]:
        """
        Treina os modelos ML com dados históricos.
        """
        print(f"🚀 Iniciando treinamento ML para {len(tickers)} tickers...")
        
        all_features = []
        all_targets = []
        
        # Coleta dados de todos os tickers
        for ticker in tickers:
            print(f"📊 Processando {ticker}...")
            
            # Busca dados históricos
            df = get_historical_data(ticker, start_date, end_date)
            if df.empty:
                print(f"❌ Sem dados para {ticker}")
                continue
            
            # Cria features
            features_df = self.feature_engineer.create_features(df)
            
            # Cria targets (direção do preço após evaluation_days)
            targets = self._create_targets(features_df, evaluation_days)
            
            # Combina features e targets
            valid_indices = ~targets.isna()
            features_subset = features_df[self.feature_engineer.feature_names][valid_indices]
            targets_subset = targets[valid_indices]
            
            all_features.append(features_subset)
            all_targets.append(targets_subset)
        
        if not all_features:
            raise ValueError("❌ Nenhum dado válido encontrado para treinamento")
        
        # Combina todos os dados
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        
        print(f"📊 Dataset final: {len(X)} amostras, {len(X.columns)} features")
        
        # Remove NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"📊 Dataset limpo: {len(X)} amostras")
        
        # Codifica targets
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Treina modelos
        results = {}
        
        if self.model_type in [ModelType.RANDOM_FOREST, ModelType.ENSEMBLE]:
            results.update(self._train_random_forest(X_train, X_test, y_train, y_test))
        
        if self.model_type in [ModelType.XGBOOST, ModelType.ENSEMBLE]:
            results.update(self._train_xgboost(X_train, X_test, y_train, y_test))
        
        self.is_trained = True
        print("✅ Treinamento concluído!")
        
        return results
    
    def train_models_fast(self, tickers: List[str], start_date: str, end_date: str, 
                         evaluation_days: int = 10) -> Dict[str, float]:
        """
        Treina modelos ML rapidamente com configurações otimizadas.
        """
        print(f"🚀 Treinamento ML rápido para {len(tickers)} tickers...")
        
        all_features = []
        all_targets = []
        
        # Coleta dados de todos os tickers
        for ticker in tickers:
            print(f"📊 Processando {ticker}...")
            
            # Busca dados históricos
            df = get_historical_data(ticker, start_date, end_date)
            if df.empty:
                print(f"❌ Sem dados para {ticker}")
                continue
            
            # Cria features
            features_df = self.feature_engineer.create_features(df)
            
            # Cria targets (direção do preço após evaluation_days)
            targets = self._create_targets(features_df, evaluation_days)
            
            # Combina features e targets
            valid_indices = ~targets.isna()
            features_subset = features_df[self.feature_engineer.feature_names][valid_indices]
            targets_subset = targets[valid_indices]
            
            all_features.append(features_subset)
            all_targets.append(targets_subset)
        
        if not all_features:
            raise ValueError("❌ Nenhum dado válido encontrado para treinamento")
        
        # Combina todos os dados
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        
        print(f"📊 Dataset final: {len(X)} amostras, {len(X.columns)} features")
        
        # Remove NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"📊 Dataset limpo: {len(X)} amostras")
        
        # Codifica targets
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Treina apenas Random Forest com configurações otimizadas
        results = {}
        
        if self.model_type in [ModelType.RANDOM_FOREST, ModelType.ENSEMBLE]:
            results.update(self._train_random_forest_fast(X_train, X_test, y_train, y_test))
        
        self.is_trained = True
        print("✅ Treinamento rápido concluído!")
        
        return results
    
    def _create_targets(self, df: pd.DataFrame, evaluation_days: int) -> pd.Series:
        """Cria targets para classificação."""
        future_prices = df['close'].shift(-evaluation_days)
        current_prices = df['close']
        
        # Classifica direção
        price_change = (future_prices - current_prices) / current_prices
        
        targets = pd.Series(index=df.index, dtype=str)
        targets[price_change > 0.03] = 'CALL'  # >3% = CALL
        targets[price_change < -0.03] = 'PUT'  # <-3% = PUT
        targets[(price_change >= -0.03) & (price_change <= 0.03)] = 'NEUTRAL'  # -3% a 3% = NEUTRAL
        
        return targets
    
    def _train_random_forest(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                           y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Treina modelo Random Forest."""
        print("🌲 Treinando Random Forest...")
        
        # Grid search para otimização
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        
        # Treina com melhor modelo
        best_rf.fit(X_train, y_train)
        
        # Avalia
        train_score = best_rf.score(X_train, y_train)
        test_score = best_rf.score(X_test, y_test)
        
        # Salva modelo
        self.models[ModelType.RANDOM_FOREST] = MLModel(
            model=best_rf,
            model_type=ModelType.RANDOM_FOREST,
            scaler=None,
            feature_names=self.feature_engineer.feature_names,
            training_accuracy=train_score,
            validation_accuracy=test_score,
            training_date=datetime.now(),
            metadata={
                'best_params': grid_search.best_params_,
                'cv_score': grid_search.best_score_
            }
        )
        
        print(f"✅ Random Forest - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        return {
            'rf_train_accuracy': train_score,
            'rf_test_accuracy': test_score,
            'rf_cv_score': grid_search.best_score_
        }
    
    def _train_random_forest_fast(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                  y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Treina modelo Random Forest rapidamente com configurações otimizadas."""
        print("🌲 Treinando Random Forest (otimizado)...")
        
        # Configurações otimizadas para melhor performance
        rf = RandomForestClassifier(
            n_estimators=300,      # Mais árvores
            max_depth=20,          # Profundidade maior
            min_samples_split=3,   # Mais sensível
            min_samples_leaf=1,    # Mais sensível
            max_features='sqrt',   # Feature selection
            random_state=42,
            n_jobs=-1
        )
        
        # Treina diretamente sem grid search
        rf.fit(X_train, y_train)
        
        # Avalia
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        
        # Salva modelo
        self.models[ModelType.RANDOM_FOREST] = MLModel(
            model=rf,
            model_type=ModelType.RANDOM_FOREST,
            scaler=None,
            feature_names=self.feature_engineer.feature_names,
            training_accuracy=train_score,
            validation_accuracy=test_score,
            training_date=datetime.now(),
            metadata={
                'fast_training': True,
                'n_estimators': 100,
                'max_depth': 10
            }
        )
        
        print(f"✅ Random Forest (rápido) - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        return {
            'rf_train_accuracy': train_score,
            'rf_test_accuracy': test_score
        }
    
    def _train_xgboost_fast(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                            y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Treina modelo XGBoost rapidamente com configurações otimizadas."""
        print("🚀 Treinando XGBoost (otimizado)...")
        
        # Configurações otimizadas para melhor performance
        xgb = XGBClassifier(
            n_estimators=500,      # Mais estimadores
            max_depth=8,           # Profundidade maior
            learning_rate=0.05,    # Learning rate menor
            subsample=0.8,         # Subsample
            colsample_bytree=0.8,  # Feature sampling
            random_state=42,
            n_jobs=-1
        )
        
        # Treina diretamente sem grid search
        xgb.fit(X_train, y_train)
        
        # Avalia
        train_score = xgb.score(X_train, y_train)
        test_score = xgb.score(X_test, y_test)
        
        # Salva modelo
        self.models[ModelType.XGBOOST] = MLModel(
            model=xgb,
            model_type=ModelType.XGBOOST,
            scaler=None,
            feature_names=self.feature_engineer.feature_names,
            training_accuracy=train_score,
            validation_accuracy=test_score,
            training_date=datetime.now(),
            metadata={
                'fast_training': True,
                'n_estimators': 500,
                'max_depth': 8,
                'learning_rate': 0.05
            }
        )
        
        print(f"✅ XGBoost (otimizado) - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        return {
            'xgb_train_accuracy': train_score,
            'xgb_test_accuracy': test_score
        }
    
    def _train_xgboost(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Treina modelo XGBoost."""
        print("🚀 Treinando XGBoost...")
        
        # Grid search para otimização
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_xgb = grid_search.best_estimator_
        
        # Treina com melhor modelo
        best_xgb.fit(X_train, y_train)
        
        # Avalia
        train_score = best_xgb.score(X_train, y_train)
        test_score = best_xgb.score(X_test, y_test)
        
        # Salva modelo
        self.models[ModelType.XGBOOST] = MLModel(
            model=best_xgb,
            model_type=ModelType.XGBOOST,
            scaler=None,
            feature_names=self.feature_engineer.feature_names,
            training_accuracy=train_score,
            validation_accuracy=test_score,
            training_date=datetime.now(),
            metadata={
                'best_params': grid_search.best_params_,
                'cv_score': grid_search.best_score_
            }
        )
        
        print(f"✅ XGBoost - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        return {
            'xgb_train_accuracy': train_score,
            'xgb_test_accuracy': test_score,
            'xgb_cv_score': grid_search.best_score_
        }
    
    def predict(self, ticker: str, analysis_date: datetime) -> MLPrediction:
        """
        Faz predição usando modelos ML treinados.
        """
        if not self.is_trained:
            raise ValueError("❌ Modelos não foram treinados ainda")
        
        # Busca dados históricos
        start_date = (analysis_date - timedelta(days=300)).strftime('%Y-%m-%d')
        end_date = analysis_date.strftime('%Y-%m-%d')
        
        df = get_historical_data(ticker, start_date, end_date)
        if df.empty:
            raise ValueError(f"❌ Sem dados para {ticker}")
        
        # Cria features
        features_df = self.feature_engineer.create_features(df)
        
        # Pega última linha (dados mais recentes)
        latest_features = features_df[self.feature_engineer.feature_names].iloc[-1:].fillna(0)
        
        # Faz predições com todos os modelos
        predictions = {}
        confidences = {}
        probabilities = {}
        feature_importances = {}
        
        for model_type, model in self.models.items():
            pred = model.model.predict(latest_features)[0]
            proba = model.model.predict_proba(latest_features)[0]
            
            predictions[model_type] = pred
            confidences[model_type] = max(proba)
            probabilities[model_type] = dict(zip(self.label_encoder.classes_, proba))
            feature_importances[model_type] = dict(zip(
                model.feature_names, 
                model.model.feature_importances_
            ))
        
        # Ensemble prediction
        if self.model_type == ModelType.ENSEMBLE and len(predictions) > 1:
            # Média ponderada das probabilidades
            proba_arrays = []
            for proba_dict in probabilities.values():
                proba_array = np.array([proba_dict.get(cls, 0) for cls in self.label_encoder.classes_])
                proba_arrays.append(proba_array)
            
            ensemble_proba = np.mean(proba_arrays, axis=0)
            ensemble_pred = self.label_encoder.classes_[np.argmax(ensemble_proba)]
            ensemble_conf = max(ensemble_proba)
            
            final_pred = ensemble_pred
            final_conf = ensemble_conf
            final_proba = dict(zip(self.label_encoder.classes_, ensemble_proba))
            final_importance = {}
            
            # Média das importâncias das features
            for feature in self.feature_engineer.feature_names:
                final_importance[feature] = np.mean([
                    imp.get(feature, 0) for imp in feature_importances.values()
                ])
        else:
            # Usa o único modelo disponível
            model_type = list(predictions.keys())[0]
            final_pred = predictions[model_type]
            final_conf = confidences[model_type]
            final_proba = probabilities[model_type]
            final_importance = feature_importances[model_type]
        
        # Converte para Direction
        direction_map = {
            'CALL': Direction.CALL,
            'PUT': Direction.PUT,
            'NEUTRAL': Direction.NEUTRAL
        }
        
        # Converte predição para string se necessário
        if isinstance(final_pred, (np.integer, np.int64)):
            final_pred = self.label_encoder.classes_[final_pred]
        
        return MLPrediction(
            direction=direction_map[final_pred],
            confidence=final_conf * 100,
            probabilities=final_proba,
            model_used=self.model_type,
            features_importance=final_importance,
            prediction_date=analysis_date
        )
    
    def save_models(self, filepath: str):
        """Salva modelos treinados."""
        if not self.is_trained:
            raise ValueError("❌ Nenhum modelo treinado para salvar")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'models': self.models,
            'feature_engineer': self.feature_engineer,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(save_data, filepath)
        print(f"💾 Modelos salvos em: {filepath}")
    
    def load_models(self, filepath: str):
        """Carrega modelos treinados."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ Arquivo não encontrado: {filepath}")
        
        save_data = joblib.load(filepath)
        
        self.models = save_data['models']
        self.feature_engineer = save_data['feature_engineer']
        self.label_encoder = save_data['label_encoder']
        self.model_type = save_data['model_type']
        self.is_trained = save_data['is_trained']
        
        print(f"📂 Modelos carregados de: {filepath}")
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Retorna importância das features para cada modelo."""
        if not self.is_trained:
            raise ValueError("❌ Modelos não foram treinados ainda")
        
        importance = {}
        for model_type, model in self.models.items():
            importance[model_type.value] = dict(zip(
                model.feature_names,
                model.model.feature_importances_
            ))
        
        return importance
