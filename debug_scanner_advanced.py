#!/usr/bin/env python3
"""
Script de debug para testar o scanner Iron Condor avançado
"""

import os
import sys
import logging
from datetime import datetime

# Adiciona o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_scanner():
    """Testa o scanner com diferentes configurações."""
    
    try:
        from src.core.data.oplab_client import OpLabClient
        from src.features.radar.iron_condor_scanner import IronCondorScanner, scan_iron_condor_opportunities
        
        logger.info("🚀 Iniciando teste do scanner Iron Condor avançado...")
        
        # Inicializa cliente
        client = OpLabClient()
        logger.info("✅ Cliente OpLab inicializado")
        
        # Testa diferentes configurações
        test_configs = [
            {
                "name": "Configuração Muito Permissiva",
                "min_probability": 0.50,
                "min_premium": 0.05,
                "min_premium_risk_ratio": 0.2,
                "min_iv_rank": 20.0,
                "max_days": 90
            },
            {
                "name": "Configuração Moderada",
                "min_probability": 0.60,
                "min_premium": 0.10,
                "min_premium_risk_ratio": 0.3,
                "min_iv_rank": 30.0,
                "max_days": 60
            },
            {
                "name": "Configuração Conservadora",
                "min_probability": 0.70,
                "min_premium": 0.15,
                "min_premium_risk_ratio": 0.5,
                "min_iv_rank": 40.0,
                "max_days": 45
            }
        ]
        
        for config in test_configs:
            logger.info(f"\n🔍 Testando: {config['name']}")
            logger.info(f"   Prob. mín: {config['min_probability']:.0%}")
            logger.info(f"   Prêmio mín: R$ {config['min_premium']:.2f}")
            logger.info(f"   Relação mín: {config['min_premium_risk_ratio']:.1f}")
            logger.info(f"   IV Rank mín: {config['min_iv_rank']:.0f}%")
            logger.info(f"   Venc. máx: {config['max_days']} dias")
            
            try:
                opportunities = scan_iron_condor_opportunities(
                    client=client,
                    min_probability=config['min_probability'],
                    max_days=config['max_days'],
                    min_premium=config['min_premium'],
                    min_premium_risk_ratio=config['min_premium_risk_ratio'],
                    min_iv_rank=config['min_iv_rank'],
                    apply_quality_filters=True
                )
                
                logger.info(f"   ✅ Encontradas: {len(opportunities)} oportunidades")
                
                if opportunities:
                    # Mostra detalhes da primeira oportunidade
                    opp = opportunities[0]
                    logger.info(f"   📊 Melhor oportunidade:")
                    logger.info(f"      Ativo: {opp['underlying']}")
                    logger.info(f"      Vencimento: {opp['vencimento']}")
                    logger.info(f"      Probabilidade: {opp['probabilidade_sucesso']:.1%}")
                    logger.info(f"      Prêmio: R$ {opp['premio_liquido']:.2f}")
                    logger.info(f"      EV: R$ {opp['EV']:.2f}")
                    logger.info(f"      Qualificada: {opp['qualificada']}")
                    break
                else:
                    logger.warning(f"   ❌ Nenhuma oportunidade encontrada")
                    
            except Exception as e:
                logger.error(f"   ❌ Erro na configuração {config['name']}: {e}")
        
        # Teste adicional: verificar se há dados de opções
        logger.info("\n🔍 Verificando disponibilidade de dados...")
        
        test_tickers = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'BBAS3']
        
        for ticker in test_tickers:
            try:
                logger.info(f"   Testando {ticker}...")
                
                # Testa preço do ativo
                price = client.get_underlying_price(ticker)
                logger.info(f"      Preço: R$ {price:.2f}")
                
                # Testa cadeia de opções
                option_chain = client.get_option_chain(ticker)
                logger.info(f"      Opções encontradas: {len(option_chain)}")
                
                if not option_chain.empty:
                    # Filtra por vencimento
                    from datetime import datetime, timedelta
                    max_date = datetime.now() + timedelta(days=90)
                    recent_options = option_chain[option_chain['expiration'] <= max_date]
                    logger.info(f"      Opções até 90 dias: {len(recent_options)}")
                    
                    if not recent_options.empty:
                        # Verifica opções com preços válidos
                        valid_options = recent_options[
                            recent_options['bid'].notna() & 
                            recent_options['ask'].notna()
                        ]
                        logger.info(f"      Opções com preços válidos: {len(valid_options)}")
                        
                        if not valid_options.empty:
                            # Verifica CALLs e PUTs
                            calls = valid_options[valid_options['option_type'] == 'call']
                            puts = valid_options[valid_options['option_type'] == 'put']
                            logger.info(f"      CALLs: {len(calls)}, PUTs: {len(puts)}")
                            
                            # Mostra alguns strikes
                            if not calls.empty:
                                logger.info(f"      CALL strikes: {calls['strike'].min():.2f} - {calls['strike'].max():.2f}")
                            if not puts.empty:
                                logger.info(f"      PUT strikes: {puts['strike'].min():.2f} - {puts['strike'].max():.2f}")
                        else:
                            logger.warning(f"      ⚠️ Nenhuma opção com preços válidos")
                    else:
                        logger.warning(f"      ⚠️ Nenhuma opção com vencimento até 90 dias")
                else:
                    logger.warning(f"      ⚠️ Nenhuma opção encontrada")
                    
            except Exception as e:
                logger.error(f"      ❌ Erro ao testar {ticker}: {e}")
        
        logger.info("\n✅ Teste concluído!")
        
    except Exception as e:
        logger.error(f"❌ Erro geral: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scanner()
