#!/usr/bin/env python3
"""
Debug do Iron Condor Scanner para identificar problemas.
"""

import os
import sys
import traceback
import pandas as pd

# Adiciona o diretório do projeto ao path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def debug_iron_condor():
    """Debug do Iron Condor Scanner."""
    
    print("🔍 Debug do Iron Condor Scanner...")
    print("=" * 60)
    
    try:
        # Carrega variáveis de ambiente
        print("1️⃣ Carregando variáveis de ambiente...")
        # As variáveis já estão carregadas pelo script bash
        print("✅ Variáveis carregadas!")
        
        # Testa cliente OpLab
        print("\n2️⃣ Testando cliente OpLab...")
        from src.core.data.oplab_client import OpLabClient
        client = OpLabClient()
        print("✅ Cliente inicializado!")
        
        # Testa ticker
        ticker = "PETR4"
        print(f"\n3️⃣ Testando ticker: {ticker}")
        
        # Busca preço
        try:
            price = client.get_underlying_price(ticker)
            print(f"✅ Preço atual: R$ {price:.2f}")
        except Exception as e:
            print(f"❌ Erro ao buscar preço: {e}")
            return
        
        # Busca opções
        try:
            option_chain = client.get_option_chain(ticker)
            print(f"✅ Opções encontradas: {len(option_chain)}")
            
            if option_chain.empty:
                print("❌ Nenhuma opção encontrada!")
                return
            
            print(f"📊 Colunas disponíveis: {list(option_chain.columns)}")
            print(f"📊 Tipos de opção: {option_chain['option_type'].unique()}")
            print(f"📊 Vencimentos: {option_chain['expiration'].nunique()}")
            
            # Mostra algumas opções
            print("\n📋 Primeiras 5 opções:")
            print(option_chain.head()[['symbol', 'option_type', 'strike', 'expiration', 'bid', 'ask', 'last']])
            
        except Exception as e:
            print(f"❌ Erro ao buscar opções: {e}")
            traceback.print_exc()
            return
        
        # Testa filtro de vencimento
        print(f"\n4️⃣ Testando filtro de vencimento (30 dias)...")
        from datetime import datetime, timedelta
        
        option_chain['expiration'] = pd.to_datetime(option_chain['expiration'])
        cutoff_date = datetime.now() + timedelta(days=30)
        filtered_options = option_chain[option_chain['expiration'] <= cutoff_date]
        
        print(f"✅ Opções com vencimento <= 30 dias: {len(filtered_options)}")
        
        if filtered_options.empty:
            print("❌ Nenhuma opção com vencimento adequado!")
            return
        
        # Testa separação CALL/PUT
        print(f"\n5️⃣ Testando separação CALL/PUT...")
        calls = filtered_options[filtered_options['option_type'] == 'CALL']
        puts = filtered_options[filtered_options['option_type'] == 'PUT']
        
        print(f"✅ CALLs: {len(calls)}")
        print(f"✅ PUTs: {len(puts)}")
        
        if calls.empty or puts.empty:
            print("❌ Faltam CALLs ou PUTs!")
            return
        
        # Testa cálculo de probabilidade
        print(f"\n6️⃣ Testando cálculo de probabilidade...")
        from scipy.stats import norm
        import numpy as np
        
        # Testa com range simples
        range_bounds = [price * 0.95, price * 1.05]  # ±5% do preço atual
        iv = 0.30  # 30% volatilidade
        days_to_exp = 30
        
        time_to_exp = days_to_exp / 365.0
        sigma = iv * np.sqrt(time_to_exp)
        
        log_lower = np.log(range_bounds[0] / price) / sigma
        log_upper = np.log(range_bounds[1] / price) / sigma
        
        prob_lower = norm.cdf(log_lower)
        prob_upper = norm.cdf(log_upper)
        probability = prob_upper - prob_lower
        
        print(f"✅ Probabilidade para range ±5%: {probability:.1%}")
        
        # Testa scanner
        print(f"\n7️⃣ Testando scanner...")
        from src.features.radar.iron_condor_scanner import IronCondorScanner
        
        scanner = IronCondorScanner(client, min_probability=0.60)  # Reduz probabilidade mínima
        
        try:
            opportunities = scanner.scan_iron_condor_opportunities(ticker, max_days=60)
            print(f"✅ Scanner executado: {len(opportunities)} oportunidades")
            
            if opportunities:
                print("🎯 Primeira oportunidade:")
                opp = opportunities[0]
                for key, value in opp.items():
                    print(f"   {key}: {value}")
            else:
                print("❌ Nenhuma oportunidade encontrada!")
                
                # Debug mais profundo
                print("\n🔍 Debug mais profundo...")
                print(f"   - Probabilidade mínima: {scanner.min_probability}")
                print(f"   - Vencimento máximo: 60 dias")
                print(f"   - CALLs disponíveis: {len(calls)}")
                print(f"   - PUTs disponíveis: {len(puts)}")
                
        except Exception as e:
            print(f"❌ Erro no scanner: {e}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Erro geral: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    debug_iron_condor()
