#!/usr/bin/env python3
"""
Script para debug do filtro de opções
"""

import os
import sys
import logging
import pandas as pd

# Adiciona o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_options_filter():
    """Testa o filtro de opções."""
    
    try:
        from src.core.data.oplab_client import OpLabClient
        
        logger.info("🚀 Testando filtro de opções...")
        
        # Inicializa cliente
        client = OpLabClient()
        logger.info("✅ Cliente OpLab inicializado")
        
        # Testa com PETR4
        ticker = 'PETR4'
        logger.info(f"🔍 Testando {ticker}...")
        
        # Busca cadeia de opções
        option_chain = client.get_option_chain(ticker)
        logger.info(f"Total de opções: {len(option_chain)}")
        
        if not option_chain.empty:
            # Mostra informações sobre a estrutura dos dados
            logger.info(f"Colunas disponíveis: {list(option_chain.columns)}")
            logger.info(f"Tipos de dados:\n{option_chain.dtypes}")
            
            # Verifica tipos de opção únicos
            unique_types = option_chain['option_type'].unique()
            logger.info(f"Tipos de opção únicos: {unique_types}")
            
            # Mostra alguns exemplos
            logger.info(f"\nPrimeiras 5 linhas:")
            print(option_chain.head())
            
            # Testa filtros
            logger.info(f"\n🔍 Testando filtros...")
            
            # Filtro original
            calls_original = option_chain[option_chain['option_type'] == 'CALL']
            puts_original = option_chain[option_chain['option_type'] == 'PUT']
            logger.info(f"Filtro original - CALLs: {len(calls_original)}, PUTs: {len(puts_original)}")
            
            # Filtro melhorado
            calls_improved = option_chain[
                (
                    (option_chain['option_type'] == 'CALL') |
                    (option_chain['option_type'] == 'call') |
                    (option_chain['option_type'].str.contains('CALL', case=False, na=False)) |
                    (option_chain['option_type'].str.contains('CA', case=False, na=False))
                ) & 
                (option_chain['bid'].notna()) & 
                (option_chain['ask'].notna())
            ]
            
            puts_improved = option_chain[
                (
                    (option_chain['option_type'] == 'PUT') |
                    (option_chain['option_type'] == 'put') |
                    (option_chain['option_type'].str.contains('PUT', case=False, na=False)) |
                    (option_chain['option_type'].str.contains('PU', case=False, na=False))
                ) & 
                (option_chain['bid'].notna()) & 
                (option_chain['ask'].notna())
            ]
            
            logger.info(f"Filtro melhorado - CALLs: {len(calls_improved)}, PUTs: {len(puts_improved)}")
            
            # Mostra alguns exemplos de CALLs e PUTs
            if not calls_improved.empty:
                logger.info(f"\nExemplos de CALLs:")
                print(calls_improved[['option_type', 'strike', 'bid', 'ask', 'expiration']].head())
            
            if not puts_improved.empty:
                logger.info(f"\nExemplos de PUTs:")
                print(puts_improved[['option_type', 'strike', 'bid', 'ask', 'expiration']].head())
            
            # Testa com vencimento próximo
            from datetime import datetime, timedelta
            max_date = datetime.now() + timedelta(days=30)
            recent_options = option_chain[option_chain['expiration'] <= max_date]
            logger.info(f"\nOpções até 30 dias: {len(recent_options)}")
            
            if not recent_options.empty:
                calls_recent = recent_options[
                    (
                        (recent_options['option_type'] == 'CALL') |
                        (recent_options['option_type'] == 'call') |
                        (recent_options['option_type'].str.contains('CALL', case=False, na=False)) |
                        (recent_options['option_type'].str.contains('CA', case=False, na=False))
                    ) & 
                    (recent_options['bid'].notna()) & 
                    (recent_options['ask'].notna())
                ]
                
                puts_recent = recent_options[
                    (
                        (recent_options['option_type'] == 'PUT') |
                        (recent_options['option_type'] == 'put') |
                        (recent_options['option_type'].str.contains('PUT', case=False, na=False)) |
                        (recent_options['option_type'].str.contains('PU', case=False, na=False))
                    ) & 
                    (recent_options['bid'].notna()) & 
                    (recent_options['ask'].notna())
                ]
                
                logger.info(f"CALLs recentes com preços: {len(calls_recent)}")
                logger.info(f"PUTs recentes com preços: {len(puts_recent)}")
                
                if not calls_recent.empty and not puts_recent.empty:
                    logger.info("✅ Encontradas opções válidas para Iron Condor!")
                else:
                    logger.warning("❌ Não há opções válidas suficientes")
        
        logger.info("\n✅ Teste concluído!")
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_options_filter()
