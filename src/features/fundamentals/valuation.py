"""
Módulo de análise de fundamentos e valuation de ações brasileiras.
Implementa múltiplos métodos de cálculo de preço justo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.core.data.oplab_client import OpLabClient


@dataclass
class FundamentalData:
    """Dados fundamentais de uma ação."""
    ticker: str
    preco_atual: float
    lpa: float  # Lucro por ação
    vpa: float  # Valor patrimonial por ação
    dps: float  # Dividendos por ação
    dividend_yield: float  # Dividend yield atual
    payout: float  # Payout ratio
    crescimento_esperado: float  # Crescimento esperado (%)
    roe: float  # Return on Equity
    pl: float  # P/L atual
    pvp: float  # P/VPA atual
    peg_ratio: float  # PEG ratio


@dataclass
class ValuationResult:
    """Resultado de valuation com múltiplos métodos."""
    ticker: str
    preco_atual: float
    preco_graham: float
    preco_dividendos: float
    preco_pl10: float
    preco_pl12: float
    preco_pl15: float
    preco_pvp1: float
    preco_pvp1_5: float
    preco_bazin: float
    peg_ratio: float
    media_precos_justos: float
    desconto: bool
    caro: bool
    margem_seguranca: Dict[str, float]


def calculate_graham_price(lpa: float, vpa: float) -> float:
    """
    Calcula preço justo usando fórmula de Graham.
    Fórmula: sqrt(22.5 × LPA × VPA)
    """
    if lpa <= 0 or vpa <= 0:
        return 0.0
    
    return math.sqrt(22.5 * lpa * vpa)


def calculate_dividend_price(dps: float, yield_min: float = 6.0) -> float:
    """
    Calcula preço teto baseado no dividend yield desejado.
    Fórmula: DPS / yield_min
    """
    if dps <= 0 or yield_min <= 0:
        return 0.0
    
    return dps / (yield_min / 100)


def calculate_pl_price(lpa: float, pl_target: float) -> float:
    """
    Calcula preço baseado em P/L alvo.
    Fórmula: LPA × P/L_alvo
    """
    if lpa <= 0 or pl_target <= 0:
        return 0.0
    
    return lpa * pl_target


def calculate_pvp_price(vpa: float, pvp_target: float) -> float:
    """
    Calcula preço baseado em P/VPA alvo.
    Fórmula: VPA × P/VPA_alvo
    """
    if vpa <= 0 or pvp_target <= 0:
        return 0.0
    
    return vpa * pvp_target


def calculate_bazin_price(dps: float, crescimento: float, taxa_desconto: float = 6.0) -> float:
    """
    Calcula preço usando fórmula de Bazin.
    Fórmula: DPS / taxa_desconto
    """
    if dps <= 0 or taxa_desconto <= 0:
        return 0.0
    
    taxa_decimal = taxa_desconto / 100
    
    return dps / taxa_decimal


def calculate_peg_ratio(pl: float, crescimento: float) -> float:
    """
    Calcula PEG ratio.
    Fórmula: P/L / crescimento
    """
    if crescimento <= 0:
        return float('inf')
    
    return pl / crescimento


def calculate_margin_of_safety(preco_atual: float, preco_justo: float) -> float:
    """
    Calcula margem de segurança.
    Fórmula: (preco_justo - preco_atual) / preco_justo × 100
    """
    if preco_justo <= 0:
        return 0.0
    
    return ((preco_justo - preco_atual) / preco_justo) * 100


def analyze_fundamentals(
    ticker: str,
    fundamental_data: FundamentalData,
    yield_min: float = 6.0,
    pl_targets: List[float] = None,
    pvp_targets: List[float] = None,
    taxa_desconto: float = 6.0
) -> ValuationResult:
    """
    Analisa fundamentos de uma ação usando múltiplos métodos de valuation.
    
    Args:
        ticker: Código da ação
        fundamental_data: Dados fundamentais
        yield_min: Yield mínimo desejado (%)
        pl_targets: Lista de P/Ls alvo
        pvp_targets: Lista de P/VPA alvos
        taxa_desconto: Taxa de desconto para Bazin (%)
    
    Returns:
        ValuationResult com todos os cálculos
    """
    if pl_targets is None:
        pl_targets = [10, 12, 15]
    if pvp_targets is None:
        pvp_targets = [1.0, 1.5]
    
    # Cálculos de preço justo
    preco_graham = calculate_graham_price(fundamental_data.lpa, fundamental_data.vpa)
    preco_dividendos = calculate_dividend_price(fundamental_data.dps, yield_min)
    
    # Preços por P/L alvo
    preco_pl10 = calculate_pl_price(fundamental_data.lpa, pl_targets[0])
    preco_pl12 = calculate_pl_price(fundamental_data.lpa, pl_targets[1])
    preco_pl15 = calculate_pl_price(fundamental_data.lpa, pl_targets[2])
    
    # Preços por P/VPA alvo
    preco_pvp1 = calculate_pvp_price(fundamental_data.vpa, pvp_targets[0])
    preco_pvp1_5 = calculate_pvp_price(fundamental_data.vpa, pvp_targets[1])
    
    # Preço Bazin
    preco_bazin = calculate_bazin_price(
        fundamental_data.dps, 
        fundamental_data.crescimento_esperado, 
        taxa_desconto
    )
    
    # PEG ratio
    peg_ratio = calculate_peg_ratio(fundamental_data.pl, fundamental_data.crescimento_esperado)
    
    # Lista de preços justos válidos (excluindo zeros)
    precos_justos = [
        preco_graham, preco_dividendos, preco_pl10, preco_pl12, preco_pl15,
        preco_pvp1, preco_pvp1_5, preco_bazin
    ]
    precos_validos = [p for p in precos_justos if p > 0]
    
    # Média dos preços justos
    media_precos_justos = sum(precos_validos) / len(precos_validos) if precos_validos else 0.0
    
    # Flags de avaliação
    desconto = fundamental_data.preco_atual < media_precos_justos
    caro = all(fundamental_data.preco_atual > p for p in precos_validos) if precos_validos else False
    
    # Margens de segurança
    margem_seguranca = {
        "graham": calculate_margin_of_safety(fundamental_data.preco_atual, preco_graham),
        "dividendos": calculate_margin_of_safety(fundamental_data.preco_atual, preco_dividendos),
        "pl10": calculate_margin_of_safety(fundamental_data.preco_atual, preco_pl10),
        "pl12": calculate_margin_of_safety(fundamental_data.preco_atual, preco_pl12),
        "pl15": calculate_margin_of_safety(fundamental_data.preco_atual, preco_pl15),
        "pvp1": calculate_margin_of_safety(fundamental_data.preco_atual, preco_pvp1),
        "pvp1_5": calculate_margin_of_safety(fundamental_data.preco_atual, preco_pvp1_5),
        "bazin": calculate_margin_of_safety(fundamental_data.preco_atual, preco_bazin),
        "media": calculate_margin_of_safety(fundamental_data.preco_atual, media_precos_justos)
    }
    
    return ValuationResult(
        ticker=ticker,
        preco_atual=fundamental_data.preco_atual,
        preco_graham=preco_graham,
        preco_dividendos=preco_dividendos,
        preco_pl10=preco_pl10,
        preco_pl12=preco_pl12,
        preco_pl15=preco_pl15,
        preco_pvp1=preco_pvp1,
        preco_pvp1_5=preco_pvp1_5,
        preco_bazin=preco_bazin,
        peg_ratio=peg_ratio,
        media_precos_justos=media_precos_justos,
        desconto=desconto,
        caro=caro,
        margem_seguranca=margem_seguranca
    )


def get_real_fundamental_data(ticker: str, client: Optional[OpLabClient] = None) -> FundamentalData:
    """
    Busca dados fundamentais reais via APIs.
    SEMPRE busca dados reais - não há fallback para dados simulados.
    """
    client = client or OpLabClient()
    
    try:
        # Buscar preço atual real
        preco_atual = client.get_underlying_price(ticker)
        
        if preco_atual <= 0:
            raise ValueError(f"Preço não encontrado para {ticker}")
        
        # Buscar dados fundamentais reais via yfinance
        fundamental_data = _fetch_real_fundamentals(ticker, preco_atual)
        
        return fundamental_data
        
    except Exception as e:
        raise ValueError(f"Erro ao buscar dados reais para {ticker}: {e}")


def _fetch_real_fundamentals(ticker: str, preco_atual: float) -> FundamentalData:
    """
    Busca dados fundamentais reais via StatusInvest (scraping).
    Fallback para yfinance se StatusInvest falhar.
    """
    try:
        # Tentar StatusInvest primeiro (mais confiável para ações brasileiras)
        return _fetch_from_statusinvest(ticker, preco_atual)
    except Exception as e:
        print(f"StatusInvest falhou para {ticker}: {e}")
        try:
            # Fallback para yfinance
            return _fetch_from_yfinance(ticker, preco_atual)
        except Exception as e2:
            raise ValueError(f"Não foi possível obter dados fundamentais reais para {ticker}. StatusInvest: {e}, yfinance: {e2}")


def _fetch_from_statusinvest(ticker: str, preco_atual: float) -> FundamentalData:
    """
    Busca dados fundamentais via scraping do StatusInvest.
    """
    import requests
    from bs4 import BeautifulSoup
    import re
    
    try:
        # URL do StatusInvest para a ação
        url = f"https://statusinvest.com.br/acoes/{ticker.lower()}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Função auxiliar para extrair valores
        def extract_value(soup, label_text):
            try:
                # Procurar por elementos que contenham o label
                elements = soup.find_all(string=re.compile(label_text, re.IGNORECASE))
                for element in elements:
                    parent = element.parent
                    if parent:
                        # Procurar pelo valor próximo (diferentes estruturas HTML)
                        value_element = None
                        
                        # Tentar diferentes seletores
                        if parent.find_next_sibling():
                            value_element = parent.find_next_sibling()
                        elif parent.parent and parent.parent.find_next_sibling():
                            value_element = parent.parent.find_next_sibling()
                        elif parent.find_next():
                            value_element = parent.find_next()
                        
                        # Procurar por elementos com classes específicas
                        if not value_element:
                            value_element = parent.find_next(class_=re.compile('value|number|price|data', re.IGNORECASE))
                        
                        if value_element:
                            value_text = value_element.get_text(strip=True)
                            # Extrair número (pode ter R$ ou %)
                            value_match = re.search(r'[\d,.-]+', value_text.replace(',', '.'))
                            if value_match:
                                return float(value_match.group())
            except:
                pass
            return 0.0
        
        # Extrair dados fundamentais usando mapeamento correto
        lpa, vpa, dps, roe = _extract_fundamentals_correctly(soup)
        
        # Debug: mostrar valores extraídos (removido para produção)
        # print(f"StatusInvest {ticker}: LPA={lpa}, VPA={vpa}, DPS={dps}, ROE={roe}")
        
        # Se não encontrou via scraping, tentar via API interna do StatusInvest
        if lpa <= 0 or vpa <= 0:
            api_data = _fetch_from_statusinvest_api(ticker)
            if api_data:
                lpa = api_data.get('lpa', lpa)
                vpa = api_data.get('vpa', vpa)
                dps = api_data.get('dps', dps)
                roe = api_data.get('roe', roe)
        
        # Calcular métricas derivadas
        pl = preco_atual / lpa if lpa > 0 else 0.0
        pvp = preco_atual / vpa if vpa > 0 else 0.0
        dividend_yield = (dps / preco_atual) * 100 if preco_atual > 0 else 0.0
        payout = (dps / lpa) * 100 if lpa > 0 else 0.0
        
        # Estimativa de crescimento (baseada em ROE e payout)
        if payout > 0 and payout < 100 and roe > 0:
            # Se há dividendos, usar fórmula: ROE * (1 - payout/100)
            crescimento_esperado = roe * (1 - payout/100)
        elif roe > 0:
            # Se não há dividendos, usar uma estimativa mais conservadora
            crescimento_esperado = roe * 0.3  # 30% do ROE como crescimento
        else:
            crescimento_esperado = 5.0  # Crescimento padrão conservador
        
        crescimento_esperado = max(0.0, min(crescimento_esperado, 20.0))  # Limitar entre 0-20%
        
        peg_ratio = pl / crescimento_esperado if crescimento_esperado > 0 else 0.0
        
        return FundamentalData(
            ticker=ticker,
            preco_atual=preco_atual,
            lpa=lpa,
            vpa=vpa,
            dps=dps,
            dividend_yield=dividend_yield,
            payout=payout,
            crescimento_esperado=crescimento_esperado,
            roe=roe,
            pl=pl,
            pvp=pvp,
            peg_ratio=peg_ratio
        )
        
    except Exception as e:
        raise ValueError(f"Erro ao buscar dados do StatusInvest para {ticker}: {e}")


def _extract_fundamental_value(soup, metric_name):
    """
    Extrai valor fundamental específico do StatusInvest.
    """
    import re
    
    try:
        # Procurar por elementos que contenham o nome da métrica
        elements = soup.find_all(string=re.compile(metric_name, re.IGNORECASE))
        
        for element in elements:
            parent = element.parent
            if parent:
                # Procurar pelo valor próximo
                value_element = None
                
                # Tentar diferentes seletores
                if parent.find_next_sibling():
                    value_element = parent.find_next_sibling()
                elif parent.parent and parent.parent.find_next_sibling():
                    value_element = parent.parent.find_next_sibling()
                elif parent.find_next():
                    value_element = parent.find_next()
                
                # Procurar por elementos com classes específicas
                if not value_element:
                    value_element = parent.find_next(class_=re.compile('value|number|price|data', re.IGNORECASE))
                
                if value_element:
                    value_text = value_element.get_text(strip=True)
                    # Extrair número (pode ter R$ ou %)
                    value_match = re.search(r'-?[\d,.-]+', value_text.replace(',', '.'))
                    if value_match:
                        try:
                            return float(value_match.group())
                        except ValueError:
                            continue
        
        # Se não encontrou, tentar buscar em tabelas ou cards específicos
        return _extract_from_cards(soup, metric_name)
        
    except Exception as e:
        print(f"Erro ao extrair {metric_name}: {e}")
        return 0.0


def _extract_from_statusinvest_improved(soup, metric_name):
    """
    Versão melhorada para extrair dados específicos do StatusInvest.
    """
    import re
    
    try:
        # Procurar por diferentes padrões de texto
        patterns = [
            f"{metric_name}",
            f"{metric_name}:",
            f"{metric_name} -",
            f"{metric_name}="
        ]
        
        for pattern in patterns:
            elements = soup.find_all(string=re.compile(pattern, re.IGNORECASE))
            
            for element in elements:
                parent = element.parent
                if parent:
                    # Procurar pelo valor próximo
                    value_element = None
                    
                    # Tentar diferentes seletores
                    if parent.find_next_sibling():
                        value_element = parent.find_next_sibling()
                    elif parent.parent and parent.parent.find_next_sibling():
                        value_element = parent.parent.find_next_sibling()
                    elif parent.find_next():
                        value_element = parent.find_next()
                    
                    # Procurar por elementos com classes específicas
                    if not value_element:
                        value_element = parent.find_next(class_=re.compile('value|number|price|data', re.IGNORECASE))
                    
                    if value_element:
                        value_text = value_element.get_text(strip=True)
                        # Extrair número (pode ter R$ ou %)
                        value_match = re.search(r'-?[\d,.-]+', value_text.replace(',', '.'))
                        if value_match:
                            try:
                                return float(value_match.group())
                            except ValueError:
                                continue
        
        return 0.0
        
    except Exception as e:
        print(f"Erro ao extrair {metric_name} (melhorado): {e}")
        return 0.0


def _extract_from_cards(soup, metric_name):
    """
    Extrai dados de cards/tabelas específicas do StatusInvest.
    """
    import re
    
    try:
        # Procurar por cards com dados fundamentais
        cards = soup.find_all(class_=re.compile('card|indicator|metric', re.IGNORECASE))
        
        for card in cards:
            card_text = card.get_text()
            if metric_name.lower() in card_text.lower():
                # Procurar por números no card
                numbers = re.findall(r'-?[\d,.-]+', card_text.replace(',', '.'))
                if numbers:
                    # Retornar o primeiro número válido encontrado
                    for num_str in numbers:
                        try:
                            return float(num_str)
                        except ValueError:
                            continue
        
        return 0.0
        
    except Exception as e:
        print(f"Erro ao extrair {metric_name} de cards: {e}")
        return 0.0


def _extract_fundamentals_correctly(soup):
    """
    Extrai dados fundamentais corretamente mapeando cada valor para sua métrica.
    """
    import re
    
    try:
        # Buscar todos os elementos com classe 'value'
        value_elements = soup.find_all(class_='value')
        
        lpa = 0.0
        vpa = 0.0
        dps = 0.0
        roe = 0.0
        
        for element in value_elements:
            text = element.get_text(strip=True)
            parent = element.parent
            if parent:
                context = parent.get_text(strip=True)
                
                # Extrair número do texto
                value_match = re.search(r'-?[\d,.-]+', text.replace(',', '.'))
                if value_match:
                    try:
                        value = float(value_match.group())
                        
                        # Mapear baseado no contexto
                        if 'lucro líquido' in context.lower() and 'nº de ações' in context.lower():
                            lpa = value
                        elif 'valor patrimonial' in context.lower() and 'patrimônio líquido' in context.lower():
                            vpa = value
                        elif 'dividendos' in context.lower() and 'proventos' in context.lower():
                            # Este é o dividend yield, não DPS
                            pass
                        elif 'lucro líquido' in context.lower() and 'patrimônio líquido' in context.lower():
                            roe = value
                        elif 'ano atual' in context.lower() and 'dividendos' in context.lower():
                            dps = value
                            
                    except ValueError:
                        continue
        
        # Se não encontrou DPS, tentar buscar por dividendos provisionados
        if dps == 0.0:
            for element in value_elements:
                text = element.get_text(strip=True)
                parent = element.parent
                if parent:
                    context = parent.get_text(strip=True)
                    if 'provisionado' in context.lower() and 'dividendos' in context.lower():
                        value_match = re.search(r'-?[\d,.-]+', text.replace(',', '.'))
                        if value_match:
                            try:
                                dps = float(value_match.group())
                                break
                            except ValueError:
                                continue
        
        return lpa, vpa, dps, roe
        
    except Exception as e:
        print(f"Erro ao extrair fundamentais: {e}")
        return 0.0, 0.0, 0.0, 0.0


def _fetch_from_statusinvest_api(ticker: str) -> dict:
    """
    Busca dados via API interna do StatusInvest.
    """
    import requests
    import json
    
    try:
        # API interna do StatusInvest para dados fundamentais
        api_url = "https://statusinvest.com.br/acoes/indicatorhistoric"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': f'https://statusinvest.com.br/acoes/{ticker.lower()}',
            'Content-Type': 'application/json'
        }
        
        # Dados para a requisição
        data = {
            'codes': [ticker.upper()],
            'time': 12
        }
        
        response = requests.post(api_url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        
        if result and len(result) > 0:
            stock_data = result[0]
            return {
                'lpa': stock_data.get('lpa', 0.0),
                'vpa': stock_data.get('vpa', 0.0),
                'dps': stock_data.get('dps', 0.0),
                'roe': stock_data.get('roe', 0.0)
            }
        
        return None
        
    except Exception as e:
        print(f"Erro na API do StatusInvest: {e}")
        return None


def _fetch_from_yfinance(ticker: str, preco_atual: float) -> FundamentalData:
    """
    Fallback: busca dados fundamentais via yfinance.
    """
    import yfinance as yf
    
    # Adicionar .SA para ações brasileiras se necessário
    yf_ticker = ticker if ticker.endswith('.SA') else f"{ticker}.SA"
    
    stock = yf.Ticker(yf_ticker)
    info = stock.info
    
    # Extrair dados fundamentais
    lpa = info.get('trailingEps', 0.0) or info.get('forwardEps', 0.0) or 0.0
    vpa = info.get('bookValue', 0.0) or 0.0
    dps = info.get('dividendRate', 0.0) or 0.0
    
    # Se não encontrou dados, tentar dados trimestrais
    if lpa <= 0:
        try:
            financials = stock.financials
            if not financials.empty:
                # Pegar último trimestre
                latest_quarter = financials.columns[0]
                net_income = financials.loc['Net Income', latest_quarter] if 'Net Income' in financials.index else 0
                shares = info.get('sharesOutstanding', 1)
                lpa = net_income / shares if shares > 0 else 0.0
        except:
            lpa = 0.0
    
    # Calcular métricas derivadas
    pl = preco_atual / lpa if lpa > 0 else 0.0
    pvp = preco_atual / vpa if vpa > 0 else 0.0
    dividend_yield = (dps / preco_atual) * 100 if preco_atual > 0 else 0.0
    payout = (dps / lpa) * 100 if lpa > 0 else 0.0
    roe = (lpa / vpa) * 100 if vpa > 0 else 0.0
    
    # Estimativa de crescimento (baseada em ROE e payout)
    crescimento_esperado = roe * (1 - payout/100) if payout < 100 else 5.0
    crescimento_esperado = max(0.0, min(crescimento_esperado, 20.0))  # Limitar entre 0-20%
    
    peg_ratio = pl / crescimento_esperado if crescimento_esperado > 0 else 0.0
    
    return FundamentalData(
        ticker=ticker,
        preco_atual=preco_atual,
        lpa=lpa,
        vpa=vpa,
        dps=dps,
        dividend_yield=dividend_yield,
        payout=payout,
        crescimento_esperado=crescimento_esperado,
        roe=roe,
        pl=pl,
        pvp=pvp,
        peg_ratio=peg_ratio
    )


def get_sample_fundamental_data(ticker: str) -> FundamentalData:
    """
    Retorna dados fundamentais de exemplo para teste.
    Em produção, estes dados viriam de uma API de fundamentos.
    """
    # Dados de exemplo para BBAS3
    sample_data = {
        "BBAS3": FundamentalData(
            ticker="BBAS3",
            preco_atual=22.16,
            lpa=2.85,
            vpa=18.50,
            dps=1.20,
            dividend_yield=5.4,
            payout=42.1,
            crescimento_esperado=8.0,
            roe=15.4,
            pl=7.8,
            pvp=1.2,
            peg_ratio=0.98
        ),
        "PETR4": FundamentalData(
            ticker="PETR4",
            preco_atual=28.45,
            lpa=4.20,
            vpa=15.80,
            dps=2.10,
            dividend_yield=7.4,
            payout=50.0,
            crescimento_esperado=5.0,
            roe=26.6,
            pl=6.8,
            pvp=1.8,
            peg_ratio=1.36
        ),
        "VALE3": FundamentalData(
            ticker="VALE3",
            preco_atual=58.20,
            lpa=8.90,
            vpa=25.40,
            dps=4.50,
            dividend_yield=7.7,
            payout=50.6,
            crescimento_esperado=3.0,
            roe=35.0,
            pl=6.5,
            pvp=2.3,
            peg_ratio=2.17
        ),
        "MOVI3": FundamentalData(
            ticker="MOVI3",
            preco_atual=9.08,
            lpa=1.20,  # Ajustado para realidade (era 2.85)
            vpa=11.50,  # Ajustado para realidade (era 18.50)
            dps=0.25,   # Ajustado para realidade (era 1.20)
            dividend_yield=2.75,
            payout=20.83,
            crescimento_esperado=8.0,
            roe=10.43,
            pl=7.57,
            pvp=0.79,
            peg_ratio=0.95
        ),
        "ITUB4": FundamentalData(
            ticker="ITUB4",
            preco_atual=32.50,
            lpa=3.80,
            vpa=28.20,
            dps=1.50,
            dividend_yield=4.62,
            payout=39.47,
            crescimento_esperado=7.0,
            roe=13.48,
            pl=8.55,
            pvp=1.15,
            peg_ratio=1.22
        ),
        "BBDC4": FundamentalData(
            ticker="BBDC4",
            preco_atual=28.90,
            lpa=3.20,
            vpa=24.50,
            dps=1.30,
            dividend_yield=4.50,
            payout=40.63,
            crescimento_esperado=6.5,
            roe=13.06,
            pl=9.03,
            pvp=1.18,
            peg_ratio=1.39
        )
    }
    
    return sample_data.get(ticker.upper(), FundamentalData(
        ticker=ticker.upper(),
        preco_atual=0.0,
        lpa=0.0,
        vpa=0.0,
        dps=0.0,
        dividend_yield=0.0,
        payout=0.0,
        crescimento_esperado=0.0,
        roe=0.0,
        pl=0.0,
        pvp=0.0,
        peg_ratio=0.0
    ))


def analyze_multiple_tickers(
    tickers: List[str],
    yield_min: float = 6.0,
    pl_targets: List[float] = None,
    pvp_targets: List[float] = None,
    taxa_desconto: float = 6.0,
    client: Optional[OpLabClient] = None
) -> List[ValuationResult]:
    """
    Analisa múltiplos tickers e retorna ordenado por maior desconto.
    Usa dados reais quando disponíveis.
    """
    results = []
    client = client or OpLabClient()
    
    for ticker in tickers:
        try:
            # Tentar buscar dados reais primeiro
            fundamental_data = get_real_fundamental_data(ticker, client)
            
            if fundamental_data.preco_atual > 0:  # Só inclui se tem dados
                result = analyze_fundamentals(
                    ticker, fundamental_data, yield_min, pl_targets, pvp_targets, taxa_desconto
                )
                results.append(result)
        except Exception as e:
            print(f"Erro ao analisar {ticker}: {e}")
            continue
    
    # Ordena por maior margem de segurança (desconto)
    results.sort(key=lambda x: x.margem_seguranca["media"], reverse=True)
    
    return results


__all__ = [
    "FundamentalData",
    "ValuationResult", 
    "analyze_fundamentals",
    "analyze_multiple_tickers",
    "get_sample_fundamental_data"
]
