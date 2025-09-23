"""
Scanner de Estratégias Iron Condor
Implementa análise e busca de oportunidades de Iron Condor baseada em probabilidade.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from src.core.data.oplab_client import OpLabClient

logger = logging.getLogger(__name__)


class IronCondorScanner:
    """
    Scanner para identificar oportunidades de Iron Condor baseadas em probabilidade.
    
    Iron Condor é uma estratégia neutra que:
    - Vende CALL OTM + Compra CALL mais alta (trava de alta)
    - Vende PUT OTM + Compra PUT mais baixa (trava de baixa)
    - Lucra quando o ativo permanece dentro do range
    """
    
    def __init__(self, client: OpLabClient, 
                 min_probability: float = 0.60, 
                 min_premium: float = 0.10,
                 min_premium_risk_ratio: float = 0.2,
                 min_iv_rank: float = 10.0,
                 apply_quality_filters: bool = True):
        """
        Inicializa o scanner com filtros avançados de qualidade.
        
        Args:
            client: Cliente OpLab para buscar dados
            min_probability: Probabilidade mínima de sucesso (default: 70%)
            min_premium: Prêmio líquido mínimo em R$ (default: 0.15)
            min_premium_risk_ratio: Relação prêmio/risco mínima (default: 0.5)
            min_iv_rank: IV Rank mínimo (default: 40)
            apply_quality_filters: Se deve aplicar filtros de qualidade (default: True)
        """
        self.client = client
        self.min_probability = min_probability
        self.min_premium = min_premium
        self.min_premium_risk_ratio = min_premium_risk_ratio
        self.min_iv_rank = min_iv_rank
        self.apply_quality_filters = apply_quality_filters
    
    def scan_iron_condor_opportunities(self, tickers: List[str], max_days: int = 30) -> List[Dict[str, Any]]:
        """
        Busca oportunidades de Iron Condor para múltiplos ativos.
        
        Args:
            tickers: Lista de tickers (ex: ['PETR4', 'VALE3', 'ITUB4'])
            max_days: Vencimento máximo em dias
            
        Returns:
            Lista de oportunidades ordenada por retorno esperado
        """
        all_opportunities = []
        
        for ticker in tickers:
            try:
                logger.info(f"🔍 Iniciando scan de Iron Condor para {ticker}")
                
                # 1. Busca dados do ativo e opções
                current_price = self.client.get_underlying_price(ticker)
                option_chain = self.client.get_option_chain(ticker)
                
                if option_chain.empty:
                    logger.warning(f"Nenhuma opção encontrada para {ticker}")
                    continue
                
                # 2. Filtra opções com vencimento adequado
                option_chain = self._filter_options_by_expiration(option_chain, max_days)
                
                if option_chain.empty:
                    logger.warning(f"Nenhuma opção com vencimento <= {max_days} dias para {ticker}")
                    continue
                
                # 3. Agrupa por vencimento
                opportunities = []
                for expiration_date, exp_options in option_chain.groupby('expiration'):
                    days_to_exp = (pd.to_datetime(expiration_date) - datetime.now()).days
                    if days_to_exp <= 0:
                        continue
                    
                    logger.info(f"Analisando vencimento {expiration_date} ({days_to_exp} dias)")
                    
                    # 4. Calcula volatilidade implícita média
                    avg_iv = self._calculate_average_implied_volatility(exp_options)
                    
                    # 5. Encontra estruturas Iron Condor
                    condor_structures = self._find_iron_condor_structures(
                        exp_options, current_price, avg_iv, days_to_exp
                    )
                    
                    opportunities.extend(condor_structures)
                
                all_opportunities.extend(opportunities)
                
                logger.info(f"✅ Encontradas {len(opportunities)} oportunidades de Iron Condor para {ticker}")
                
                # Se já encontrou pelo menos uma oportunidade, pode parar
                if len(all_opportunities) >= 1 and self.apply_quality_filters:
                    break
                    
            except Exception as e:
                logger.error(f"❌ Erro no scan de Iron Condor para {ticker}: {e}")
                continue
        
        # 6. Ordena por retorno esperado
        all_opportunities.sort(key=lambda x: x['retorno_esperado_pct'], reverse=True)
        
        logger.info(f"✅ Total de {len(all_opportunities)} oportunidades encontradas em {len(tickers)} tickers")
        return all_opportunities
    
    def get_top_liquid_tickers(self, max_days: int = 30) -> List[str]:
        """
        Retorna os 10 tickers mais líquidos em opções.
        
        Args:
            max_days: Vencimento máximo para considerar liquidez
            
        Returns:
            Lista dos 10 tickers mais líquidos
        """
        # Lista dos principais tickers com opções na B3
        potential_tickers = [
            'PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'BBAS3', 
            'ABEV3', 'WEGE3', 'MGLU3', 'RENT3', 'SUZB3',
            'LREN3', 'JBSS3', 'RADL3', 'CCRO3', 'CSAN3',
            'VIVT3', 'PETR3', 'BRFS3', 'GGBR4', 'USIM5'
        ]
        
        ticker_liquidity = []
        
        for ticker in potential_tickers:
            try:
                option_chain = self.client.get_option_chain(ticker)
                if not option_chain.empty:
                    # Filtra por vencimento
                    option_chain = self._filter_options_by_expiration(option_chain, max_days)
                    
                    if not option_chain.empty:
                        # Calcula liquidez (volume total + quantidade de strikes)
                        total_volume = option_chain['volume'].sum() if 'volume' in option_chain.columns else 0
                        unique_strikes = len(option_chain['strike'].unique())
                        liquidity_score = total_volume + (unique_strikes * 1000)  # Peso para strikes
                        
                        ticker_liquidity.append((ticker, liquidity_score))
                        
            except Exception as e:
                logger.debug(f"Erro ao verificar liquidez de {ticker}: {e}")
                continue
        
        # Ordena por liquidez e retorna os top 10
        ticker_liquidity.sort(key=lambda x: x[1], reverse=True)
        top_tickers = [ticker for ticker, _ in ticker_liquidity[:10]]
        
        logger.info(f"📊 Top 10 tickers mais líquidos: {top_tickers}")
        return top_tickers
    
    def _filter_options_by_expiration(self, df: pd.DataFrame, max_days: int) -> pd.DataFrame:
        """Filtra opções por vencimento máximo."""
        df = df.copy()
        df['expiration'] = pd.to_datetime(df['expiration'])
        
        cutoff_date = datetime.now() + timedelta(days=max_days)
        filtered_df = df[df['expiration'] <= cutoff_date]
        
        return filtered_df
    
    def _calculate_average_implied_volatility(self, options_df: pd.DataFrame) -> float:
        """
        Calcula volatilidade implícita média das opções.
        Como o OpLab pode não fornecer IV diretamente, usamos uma estimativa.
        """
        # Se houver dados de IV, usamos a média
        if 'implied_volatility' in options_df.columns:
            iv_values = options_df['implied_volatility'].dropna()
            if not iv_values.empty:
                return iv_values.mean()
        
        # Estimativa baseada em volatilidade histórica típica
        # Para ações brasileiras, IV típica varia entre 20-40%
        return 0.30  # 30% como estimativa conservadora
    
    def _find_iron_condor_structures(self, 
                                   options_df: pd.DataFrame, 
                                   current_price: float, 
                                   iv: float, 
                                   days_to_exp: int) -> List[Dict[str, Any]]:
        """
        Encontra estruturas Iron Condor viáveis.
        """
        structures = []
        
        # Separa CALLs e PUTs e filtra apenas com preços válidos
        calls = options_df[
            (options_df['option_type'] == 'CALL') & 
            (options_df['bid'].notna()) & 
            (options_df['ask'].notna())
        ].copy()
        puts = options_df[
            (options_df['option_type'] == 'PUT') & 
            (options_df['bid'].notna()) & 
            (options_df['ask'].notna())
        ].copy()
        
        if calls.empty or puts.empty:
            logger.warning(f"CALLs com preços: {len(calls)}, PUTs com preços: {len(puts)}")
            return structures
        
        # Ordena por strike
        calls = calls.sort_values('strike')
        puts = puts.sort_values('strike', ascending=False)
        
        # Busca estruturas com probabilidade adequada (OTIMIZADO)
        # Limita a busca para evitar loops infinitos
        max_combinations = 1000  # Máximo de combinações a testar
        combinations_tested = 0
        
        # Filtra opções próximas ao preço atual para reduzir combinações
        price_tolerance = 0.15  # 15% acima/abaixo do preço atual
        calls_filtered = calls[
            (calls['strike'] >= current_price * (1 - price_tolerance)) &
            (calls['strike'] <= current_price * (1 + price_tolerance * 2))
        ].head(20)  # Máximo 20 CALLs
        
        puts_filtered = puts[
            (puts['strike'] >= current_price * (1 - price_tolerance * 2)) &
            (puts['strike'] <= current_price * (1 + price_tolerance))
        ].head(20)  # Máximo 20 PUTs
        
        logger.info(f"Testando {len(calls_filtered)} CALLs e {len(puts_filtered)} PUTs")
        
        for _, call_short in calls_filtered.iterrows():
            if combinations_tested >= max_combinations:
                logger.warning(f"Limite de {max_combinations} combinações atingido")
                break
                
            # CALL vendida (strike mais baixo)
            call_strike_short = call_short['strike']
            
            # CALL comprada (strike mais alto, para proteção)
            call_strikes_long = calls_filtered[calls_filtered['strike'] > call_strike_short].head(5)
            
            if call_strikes_long.empty:
                continue
            
            for _, call_long in call_strikes_long.iterrows():
                call_strike_long = call_long['strike']
                
                # PUT vendida (strike mais alto)
                put_strikes_short = puts_filtered[puts_filtered['strike'] < current_price].head(5)
                
                for _, put_short in put_strikes_short.iterrows():
                    put_strike_short = put_short['strike']
                    
                    # PUT comprada (strike mais baixo, para proteção)
                    put_strikes_long = puts_filtered[puts_filtered['strike'] < put_strike_short].head(5)
                    
                    if put_strikes_long.empty:
                        continue
                    
                    for _, put_long in put_strikes_long.iterrows():
                        combinations_tested += 1
                        
                        if combinations_tested > max_combinations:
                            break
                            
                        put_strike_long = put_long['strike']
                        
                        # Verifica se a estrutura é válida
                        if self._is_valid_iron_condor(call_strike_short, call_strike_long, 
                                                    put_strike_short, put_strike_long, current_price):
                            
                            # Calcula métricas da estrutura
                            structure = self._calculate_iron_condor_metrics(
                                call_short, call_long, put_short, put_long,
                                current_price, iv, days_to_exp
                            )
                            
                            if structure:
                                if not self.apply_quality_filters or self._is_quality_opportunity(structure):
                                    structures.append(structure)
                                    # Limita a 10 melhores estruturas
                                    if len(structures) >= 10:
                                        break
                                else:
                                    # Log para debug - por que foi rejeitada
                                    rr_ratio = structure['premio_liquido']/structure['risco_maximo'] if structure['risco_maximo'] > 0 else 0
                                    logger.debug(f"Oportunidade rejeitada: Prob={structure['probabilidade_sucesso']:.2f}, "
                                               f"Prêmio={structure['premio_liquido']:.2f}, "
                                               f"R/R={rr_ratio:.2f}")
                            else:
                                logger.debug("Estrutura Iron Condor inválida")
                        
                        if combinations_tested > max_combinations:
                            break
                    
                    if combinations_tested > max_combinations:
                        break
                
                if combinations_tested > max_combinations:
                    break
            
            if combinations_tested > max_combinations:
                break
        
        return structures
    
    def _is_quality_opportunity(self, opportunity: Dict[str, Any]) -> bool:
        """
        Verifica se a oportunidade atende aos critérios de qualidade.
        
        Filtros aplicados:
        1. Probabilidade mínima
        2. Prêmio líquido mínimo
        3. Relação prêmio/risco mínima
        4. IV Rank mínimo
        5. DTE mínimo (7 dias)
        6. Largura das asas otimizada
        """
        # 1. Probabilidade mínima
        if opportunity['probabilidade_sucesso'] < self.min_probability:
            return False
        
        # 2. Prêmio líquido mínimo
        if opportunity['premio_liquido'] < self.min_premium:
            return False
        
        # 3. Relação prêmio/risco mínima (mais flexível)
        if opportunity['risco_maximo'] > 0:
            premium_risk_ratio = opportunity['premio_liquido'] / opportunity['risco_maximo']
            if premium_risk_ratio < self.min_premium_risk_ratio:
                return False
        else:
            # Se risco é zero, não pode ser uma oportunidade válida
            return False
        
        # 4. IV Rank mínimo
        if 'iv_rank' in opportunity and opportunity['iv_rank'] < self.min_iv_rank:
            return False
        
        # 5. DTE mínimo (2 dias) - mais permissivo
        if 'dte' in opportunity and opportunity['dte'] <= 2:
            return False
        
        # 6. Largura das asas otimizada (mais flexível para encontrar oportunidades)
        strikes = opportunity['strikes']
        current_price = (strikes['call_short'] + strikes['put_short']) / 2  # Preço médio do range
        
        call_wing_width = abs(strikes['call_long'] - strikes['call_short']) / current_price
        put_wing_width = abs(strikes['put_short'] - strikes['put_long']) / current_price
        
        min_wing_width = 0.005  # 0.5% - muito mais flexível
        max_wing_width = 0.20   # 20% - permite asas maiores
        
        if not (min_wing_width <= call_wing_width <= max_wing_width and 
                min_wing_width <= put_wing_width <= max_wing_width):
            return False
        
        return True
    
    def _is_valid_iron_condor(self, call_short: float, call_long: float, 
                            put_short: float, put_long: float, current_price: float) -> bool:
        """
        Verifica se a estrutura Iron Condor é válida.
        
        Validações relaxadas:
        - CALL vendida < CALL comprada
        - PUT vendida > PUT comprada  
        - PUT vendida < CALL vendida (range válido)
        - Strikes são próximos ao preço atual (relaxado)
        """
        # Validações básicas
        if not (call_short < call_long and put_short > put_long and put_short < call_short):
            return False
        
        # Verifica se os strikes estão em uma faixa razoável (±30% do preço atual)
        price_tolerance = 0.30
        min_price = current_price * (1 - price_tolerance)
        max_price = current_price * (1 + price_tolerance)
        
        return (min_price <= call_short <= max_price and
                min_price <= call_long <= max_price and
                min_price <= put_short <= max_price and
                min_price <= put_long <= max_price)
    
    def _calculate_iron_condor_metrics(self, 
                                     call_short: pd.Series, call_long: pd.Series,
                                     put_short: pd.Series, put_long: pd.Series,
                                     current_price: float, iv: float, days_to_exp: int) -> Optional[Dict[str, Any]]:
        """
        Calcula métricas completas da estrutura Iron Condor.
        """
        try:
            # Preços das opções (usa mid-price quando disponível)
            call_short_price = self._get_option_price(call_short)
            call_long_price = self._get_option_price(call_long)
            put_short_price = self._get_option_price(put_short)
            put_long_price = self._get_option_price(put_long)
            
            if any(p is None for p in [call_short_price, call_long_price, put_short_price, put_long_price]):
                return None
            
            # Prêmio líquido recebido (vendas - compras)
            premio_liquido = (call_short_price + put_short_price) - (call_long_price + put_long_price)
            
            if premio_liquido <= 0:
                return None  # Estrutura não lucrativa
            
            # Range de sucesso
            range_sucesso = [put_short['strike'], call_short['strike']]
            
            # Probabilidade de sucesso (preço ficar no range)
            prob_sucesso = self._calculate_range_probability(
                current_price, range_sucesso, iv, days_to_exp
            )
            
            # Risco máximo (distância entre strikes menos prêmio)
            call_spread = call_long['strike'] - call_short['strike']
            put_spread = put_short['strike'] - put_long['strike']
            risco_maximo = max(call_spread, put_spread) - premio_liquido
            
            # Retorno máximo (% sobre risco)
            retorno_maximo_pct = (premio_liquido / risco_maximo) * 100 if risco_maximo > 0 else 0
            
            # Cálculos avançados
            prob_perda = 1 - prob_sucesso
            
            # Valor Esperado (EV) com gestão embutida
            ev = self._calculate_expected_value(premio_liquido, risco_maximo, prob_sucesso)
            
            # Relação prêmio/risco
            relacao_premio_risco = premio_liquido / risco_maximo if risco_maximo > 0 else 0
            
            # Delta dos strikes vendidos
            delta_calls = self._calculate_delta(call_short, current_price, iv, days_to_exp)
            delta_puts = self._calculate_delta(put_short, current_price, iv, days_to_exp)
            
            # IV Rank (simplificado - usando IV média)
            iv_rank = self._calculate_iv_rank(iv)
            
            # DTE (Dias até vencimento)
            dte = days_to_exp
            
            # Verifica se está qualificada
            qualificada = self._is_qualified(
                prob_sucesso, premio_liquido, relacao_premio_risco, 
                iv_rank, dte, ev
            )
            
            return {
                "underlying": call_short.get('symbol', 'N/A').split()[0] if 'symbol' in call_short else 'N/A',
                "vencimento": call_short['expiration'].strftime('%Y-%m-%d') if 'expiration' in call_short else 'N/A',
                "probabilidade_sucesso": round(prob_sucesso, 3),
                "range_sucesso": [round(x, 2) for x in range_sucesso],
                "premio_liquido": round(premio_liquido, 2),
                "risco_maximo": round(risco_maximo, 2),
                "relacao_premio_risco": round(relacao_premio_risco, 3),
                "EV": round(ev, 2),
                "iv_rank": round(iv_rank, 1),
                "delta_calls": round(delta_calls, 3),
                "delta_puts": round(delta_puts, 3),
                "dte": dte,
                "qualificada": qualificada,
                "retorno_maximo_pct": round(retorno_maximo_pct, 1),
                "retorno_esperado_pct": round((ev / risco_maximo) * 100 if risco_maximo > 0 else 0, 1),
                "dias_vencimento": days_to_exp,
                "strikes": {
                    "call_short": call_short['strike'],
                    "call_long": call_long['strike'],
                    "put_short": put_short['strike'],
                    "put_long": put_long['strike']
                },
                "precos_opcoes": {
                    "call_short": call_short_price,
                    "call_long": call_long_price,
                    "put_short": put_short_price,
                    "put_long": put_long_price
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas Iron Condor: {e}")
            return None
    
    def _get_option_price(self, option: pd.Series) -> Optional[float]:
        """Obtém preço da opção (prefere mid-price)."""
        # Tenta bid-ask mid primeiro
        if pd.notna(option.get('bid')) and pd.notna(option.get('ask')):
            return (option['bid'] + option['ask']) / 2
        
        # Fallback para last price
        if pd.notna(option.get('last')):
            return option['last']
        
        return None
    
    def _calculate_range_probability(self, current_price: float, range_bounds: List[float], 
                                   iv: float, days_to_exp: int) -> float:
        """
        Calcula probabilidade do preço ficar dentro do range usando distribuição normal.
        
        Args:
            current_price: Preço atual do ativo
            range_bounds: [limite_inferior, limite_superior]
            iv: Volatilidade implícita anual
            days_to_exp: Dias até vencimento
            
        Returns:
            Probabilidade (0-1) do preço ficar no range
        """
        lower_bound, upper_bound = range_bounds
        
        try:
            # Parâmetros da distribuição log-normal
            # Para retornos logarítmicos: ln(S_T/S_0) ~ N((r - σ²/2)T, σ²T)
            # Assumimos taxa livre de risco = 0 (simplificação)
            time_to_exp = days_to_exp / 365.0
            
            # Evita divisão por zero
            if time_to_exp <= 0 or iv <= 0:
                logger.warning(f"Parâmetros inválidos: time_to_exp={time_to_exp}, iv={iv}")
                return 0.5  # Probabilidade neutra
            
            sigma = iv * np.sqrt(time_to_exp)
            
            # Evita divisão por zero no cálculo dos logs
            if sigma <= 0:
                logger.warning(f"Sigma inválido: {sigma}")
                return 0.5  # Probabilidade neutra
            
            # Log dos limites normalizados
            log_lower = np.log(lower_bound / current_price) / sigma
            log_upper = np.log(upper_bound / current_price) / sigma
            
            # Probabilidade usando distribuição normal padrão
            prob_lower = norm.cdf(log_lower)
            prob_upper = norm.cdf(log_upper)
            
            # Probabilidade de ficar no range
            probability = prob_upper - prob_lower
            
            return max(0, min(1, probability))  # Clampa entre 0 e 1
            
        except Exception as e:
            logger.error(f"Erro no cálculo de probabilidade: {e}")
            return 0.5  # Probabilidade neutra em caso de erro
    
    def _calculate_expected_value(self, premio_liquido: float, risco_maximo: float, prob_sucesso: float) -> float:
        """
        Calcula o Valor Esperado (EV) com gestão embutida.
        
        Gestão aplicada:
        - Take Profit: 60% do prêmio
        - Stop Loss: 1.5x o crédito recebido
        - Encerramento automático: DTE ≤ 7 dias
        """
        prob_perda = 1 - prob_sucesso
        
        # Take Profit: 60% do prêmio
        take_profit = premio_liquido * 0.6
        
        # Stop Loss: 1.5x o crédito recebido
        stop_loss = premio_liquido * 1.5
        
        # EV = (prob_sucesso * take_profit) - (prob_perda * stop_loss)
        ev = (prob_sucesso * take_profit) - (prob_perda * stop_loss)
        
        return ev
    
    def _calculate_delta(self, option: pd.Series, current_price: float, iv: float, days_to_exp: int) -> float:
        """
        Calcula o Delta da opção (simplificado).
        """
        try:
            # Delta aproximado baseado em Black-Scholes simplificado
            # Para CALLs: delta positivo (0 a 1)
            # Para PUTs: delta negativo (-1 a 0)
            
            strike = option['strike']
            
            # Determina se é CALL ou PUT
            is_call = 'C' in option.get('symbol', '') or 'CA' in option.get('symbol', '')
            
            # Cálculo simplificado do delta
            if is_call:
                # Delta para CALL (positivo)
                if current_price > strike:
                    return min(0.9, 0.5 + (current_price - strike) / (strike * 0.1))
                else:
                    return max(0.1, 0.5 - (strike - current_price) / (strike * 0.1))
            else:
                # Delta para PUT (negativo)
                if current_price < strike:
                    return max(-0.9, -0.5 - (strike - current_price) / (strike * 0.1))
                else:
                    return min(-0.1, -0.5 + (current_price - strike) / (strike * 0.1))
                    
        except Exception as e:
            logger.debug(f"Erro ao calcular delta: {e}")
            return 0.0
    
    def _calculate_iv_rank(self, iv: float) -> float:
        """
        Calcula o IV Rank (simplificado).
        Assume IV histórica entre 20% e 80%.
        """
        try:
            # IV Rank = (IV atual - IV mínima) / (IV máxima - IV mínima) * 100
            iv_min = 0.20  # 20%
            iv_max = 0.80  # 80%
            
            iv_rank = ((iv - iv_min) / (iv_max - iv_min)) * 100
            return max(0, min(100, iv_rank))  # Clampa entre 0 e 100
            
        except Exception as e:
            logger.debug(f"Erro ao calcular IV Rank: {e}")
            return 50.0  # Valor neutro
    
    def _is_qualified(self, prob_sucesso: float, premio_liquido: float, 
                     relacao_premio_risco: float, iv_rank: float, 
                     dte: int, ev: float) -> bool:
        """
        Verifica se a operação está qualificada baseada em todos os filtros.
        """
        # Todos os critérios devem ser atendidos
        criteria = [
            prob_sucesso >= self.min_probability,
            premio_liquido >= self.min_premium,
            relacao_premio_risco >= self.min_premium_risk_ratio,
            iv_rank >= self.min_iv_rank,
            dte > 7,  # DTE mínimo
            ev > 0    # EV positivo
        ]
        
        return all(criteria)


def scan_iron_condor_opportunities(client: OpLabClient,
                                 min_probability: float = 0.60,
                                 max_days: int = 30,
                                 min_premium: float = 0.10,
                                 min_premium_risk_ratio: float = 0.2,
                                 min_iv_rank: float = 10.0,
                                 apply_quality_filters: bool = True) -> List[Dict[str, Any]]:
    """
    Função conveniente para scan de oportunidades Iron Condor nos 10 tickers mais líquidos.
    
    Args:
        client: Cliente OpLab
        min_probability: Probabilidade mínima de sucesso (default: 70%)
        max_days: Vencimento máximo em dias
        min_premium: Prêmio líquido mínimo em R$ (default: 0.15)
        min_premium_risk_ratio: Relação prêmio/risco mínima (default: 0.5)
        min_iv_rank: IV Rank mínimo (default: 40)
        apply_quality_filters: Se deve aplicar filtros de qualidade
        
    Returns:
        Lista de oportunidades ordenada por EV (Valor Esperado)
    """
    scanner = IronCondorScanner(
        client, min_probability, min_premium, 
        min_premium_risk_ratio, min_iv_rank, apply_quality_filters
    )
    
    # Busca os 10 tickers mais líquidos
    top_tickers = scanner.get_top_liquid_tickers(max_days)
    
    if not top_tickers:
        logger.warning("❌ Nenhum ticker líquido encontrado")
        return []
    
    # Busca oportunidades nos tickers mais líquidos
    opportunities = scanner.scan_iron_condor_opportunities(top_tickers, max_days)
    
    # Ordena por EV (maior primeiro), depois por probabilidade
    opportunities.sort(key=lambda x: (x['EV'], x['probabilidade_sucesso']), reverse=True)
    
    return opportunities


__all__ = ['IronCondorScanner', 'scan_iron_condor_opportunities']


