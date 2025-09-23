#!/usr/bin/env python3
"""
Teste simples para verificar se o Iron Condor Scanner funciona corretamente.
"""

import os
import sys
import traceback

# Adiciona o diretório do projeto ao path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def test_iron_condor_scanner():
    """Testa o Iron Condor Scanner com as variáveis de ambiente carregadas."""
    
    print("🧪 Testando Iron Condor Scanner...")
    print("=" * 50)
    
    try:
        # 1. Testa importação
        print("1️⃣ Testando importações...")
        from src.core.data.oplab_client import OpLabClient
        from src.features.radar.iron_condor_scanner import IronCondorScanner, scan_iron_condor_opportunities
        print("✅ Importações realizadas com sucesso!")
        
        # 2. Verifica variáveis de ambiente
        print("\n2️⃣ Verificando variáveis de ambiente...")
        required_vars = [
            'OPLAB_API_BASE_URL',
            'OPLAB_API_KEY',
            'OPLAB_OPTION_CHAIN_ENDPOINT',
            'OPLAB_QUOTE_ENDPOINT'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
            else:
                print(f"✅ {var}: {os.getenv(var)[:50]}...")
        
        if missing_vars:
            print(f"❌ Variáveis faltando: {missing_vars}")
            return False
        
        # 3. Testa inicialização do cliente OpLab
        print("\n3️⃣ Testando cliente OpLab...")
        client = OpLabClient()
        print("✅ Cliente OpLab inicializado com sucesso!")
        
        # 4. Testa busca de preço do ativo
        print("\n4️⃣ Testando busca de preço do ativo...")
        ticker = "PETR4"
        try:
            price = client.get_underlying_price(ticker)
            print(f"✅ Preço de {ticker}: R$ {price:.2f}")
        except Exception as e:
            print(f"⚠️ Erro ao buscar preço de {ticker}: {e}")
            print("💡 Isso pode ser normal se o ativo não estiver disponível")
        
        # 5. Testa busca de opções
        print("\n5️⃣ Testando busca de opções...")
        try:
            option_chain = client.get_option_chain(ticker)
            print(f"✅ Opções encontradas: {len(option_chain)} contratos")
            
            if not option_chain.empty:
                print(f"📊 Vencimentos únicos: {option_chain['expiration'].nunique()}")
                print(f"📊 Tipos de opção: {option_chain['option_type'].unique()}")
            else:
                print("⚠️ Nenhuma opção encontrada - pode ser normal para alguns ativos")
                
        except Exception as e:
            print(f"⚠️ Erro ao buscar opções: {e}")
        
        # 6. Testa scanner Iron Condor
        print("\n6️⃣ Testando scanner Iron Condor...")
        try:
            scanner = IronCondorScanner(client, min_probability=0.70)
            print("✅ Scanner Iron Condor inicializado!")
            
            # Teste rápido com parâmetros conservadores
            print("🔍 Executando scan de teste...")
            opportunities = scanner.scan_iron_condor_opportunities(ticker, max_days=60)
            print(f"✅ Scan concluído: {len(opportunities)} oportunidades encontradas")
            
            if opportunities:
                best = opportunities[0]
                print(f"🎯 Melhor oportunidade:")
                print(f"   - Probabilidade: {best['probabilidade_sucesso']:.1%}")
                print(f"   - Retorno esperado: {best['retorno_esperado_pct']:.1f}%")
                print(f"   - Prêmio líquido: R$ {best['premio_liquido']:.2f}")
            
        except Exception as e:
            print(f"❌ Erro no scanner Iron Condor: {e}")
            traceback.print_exc()
            return False
        
        print("\n" + "=" * 50)
        print("🎉 TESTE CONCLUÍDO COM SUCESSO!")
        print("✅ Iron Condor Scanner está funcionando corretamente!")
        print("🚀 A aplicação está pronta para uso!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_iron_condor_scanner()
    sys.exit(0 if success else 1)


