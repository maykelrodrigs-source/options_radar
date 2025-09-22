import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import math
from scipy.stats import norm

from src.core.data.oplab_client import OpLabClient


def get_most_active_stocks(client: OpLabClient, limit: int = 20) -> List[str]:
    """
    Busca os ativos mais líquidos/ativos do mercado.
    
    Tenta usar o endpoint da API primeiro, se falhar usa lista predefinida.
    """
    try:
        # Tenta usar o endpoint da API
        return client.get_most_active_stocks(limit)
    except RuntimeError:
        # Fallback para lista predefinida dos papéis mais líquidos com opções na B3
        liquid_stocks = [
            "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "ABEV3", "B3SA3", "WEGE3",
            "RENT3", "LREN3", "MGLU3", "VIIA3", "RADL3", "CSAN3", "GGBR4", "USIM5",
            "CSNA3", "JBSS3", "BEEF3", "SUZB3", "RAIL3", "CCRO3", "TOTS3", "EMBR3",
            "GOAU4", "PRIO3", "KLBN11", "NTCO3", "ELET3", "CMIG4", "VIVT3", "TIMS3"
        ]
        return liquid_stocks[:limit]


def calculate_black_scholes_delta(spot_price: float, strike: float, time_to_expiry_years: float, 
                                 risk_free_rate: float = 0.1275, volatility: float = 0.30, 
                                 option_type: str = "CALL") -> float:
    """
    Calcula delta usando modelo Black-Scholes.
    
    Args:
        spot_price: Preço atual do ativo subjacente
        strike: Preço de exercício da opção
        time_to_expiry_years: Tempo até vencimento em anos
        risk_free_rate: Taxa livre de risco (padrão 12.75% - Selic)
        volatility: Volatilidade implícita (padrão 30%)
        option_type: 'CALL' ou 'PUT'
    
    Returns:
        Delta da opção
    """
    if time_to_expiry_years <= 0 or spot_price <= 0 or strike <= 0:
        return 0.0
    
    try:
        d1 = (math.log(spot_price / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry_years) / \
             (volatility * math.sqrt(time_to_expiry_years))
        
        if option_type == "CALL":
            return norm.cdf(d1)
        else:  # PUT
            return norm.cdf(d1) - 1.0
    except (ValueError, ZeroDivisionError):
        # Em caso de erro, retorna delta neutro
        return 0.5 if option_type == "CALL" else -0.5


def calculate_annualized_return(premium: float, strike: float, days_to_expiration: int) -> float:
    """
    Calcula o retorno anualizado da opção.
    
    Fórmula corrigida:
    retorno_dia = (prêmio ÷ strike) ÷ dias_para_vencimento
    retorno_aa = retorno_dia × 252
    
    Args:
        premium: Prêmio da opção
        strike: Strike da opção (sempre usar strike como base)
        days_to_expiration: Dias até o vencimento
    
    Returns:
        Retorno anualizado em percentual
    """
    if days_to_expiration <= 0 or strike <= 0 or premium <= 0:
        return 0.0
    
    daily_return = (premium / strike) / days_to_expiration
    annualized_return = daily_return * 252 * 100
    
    return annualized_return


def calculate_exercise_probability(delta: float, option_type: str) -> float:
    """
    Estima a probabilidade de exercício baseada no delta.
    
    Para CALLs: probabilidade ≈ delta
    Para PUTs: probabilidade ≈ |delta|
    
    Args:
        delta: Delta da opção
        option_type: 'CALL' ou 'PUT'
    
    Returns:
        Probabilidade de exercício em percentual (0-100)
    """
    if pd.isna(delta):
        return 0.0
    
    if option_type == "CALL":
        # Para CALL, delta positivo indica probabilidade de exercício
        return abs(delta) * 100
    else:  # PUT
        # Para PUT, delta negativo indica probabilidade de exercício
        return abs(delta) * 100


def filter_liquid_options(df: pd.DataFrame, min_volume: int = 50, min_open_interest: int = 500) -> pd.DataFrame:
    """
    Aplica filtros de liquidez nas opções.
    
    Args:
        df: DataFrame com opções
        min_volume: Volume mínimo de contratos (padrão 50)
        min_open_interest: Open Interest mínimo (padrão 500)
    
    Returns:
        DataFrame filtrado
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Verifica se temos Open Interest separado
    has_oi = 'open_interest' in df.columns and df['open_interest'].notna().any()
    
    if has_oi:
        # Usa Volume e Open Interest separadamente
        liquid_mask = (
            (df['volume'].fillna(0) >= min_volume) &
            (df['open_interest'].fillna(0) >= min_open_interest)
        )
    else:
        # Fallback: usa volume para ambos os filtros
        liquid_mask = (
            (df['volume'].fillna(0) >= min_volume) &
            (df['volume'].fillna(0) >= min_open_interest)
        )
    
    return df[liquid_mask]


def filter_realistic_strikes(df: pd.DataFrame, stock_price: float) -> pd.DataFrame:
    """
    Filtra opções com strikes realistas baseados no preço atual.
    
    Regras:
    - PUT: aceitar apenas strikes até -10% do preço atual
    - CALL: aceitar apenas strikes até +20% do preço atual
    
    Args:
        df: DataFrame com opções
        stock_price: Preço atual da ação
    
    Returns:
        DataFrame filtrado
    """
    if df.empty or stock_price <= 0:
        return df
    
    df = df.copy()
    
    # Define limites
    put_max_strike = stock_price * 0.90  # -10%
    call_max_strike = stock_price * 1.20  # +20%
    
    # Aplica filtros por tipo
    realistic_mask = (
        ((df['option_type'] == 'PUT') & (df['strike'] <= put_max_strike)) |
        ((df['option_type'] == 'CALL') & (df['strike'] <= call_max_strike))
    )
    
    return df[realistic_mask]


def filter_reasonable_premiums(df: pd.DataFrame, stock_price: float, max_premium_pct: float = 20.0) -> pd.DataFrame:
    """
    Filtra opções com prêmios razoáveis.
    
    Args:
        df: DataFrame com opções
        stock_price: Preço atual da ação
        max_premium_pct: Percentual máximo do prêmio em relação ao preço da ação
    
    Returns:
        DataFrame filtrado
    """
    if df.empty or stock_price <= 0:
        return df
    
    df = df.copy()
    max_premium = stock_price * (max_premium_pct / 100)
    
    reasonable_mask = df['last'] <= max_premium
    
    return df[reasonable_mask]


def filter_low_risk_options(df: pd.DataFrame, stock_price: float, max_exercise_prob: float = 5.0) -> pd.DataFrame:
    """
    Filtra opções com baixo risco de exercício.
    
    Args:
        df: DataFrame com opções
        stock_price: Preço atual da ação (para calcular delta se necessário)
        max_exercise_prob: Probabilidade máxima de exercício (%)
    
    Returns:
        DataFrame filtrado
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Calcula dias até vencimento em anos
    today = datetime.now()
    df['days_to_expiration'] = (df['expiration'] - today).dt.days
    df['time_to_expiry_years'] = df['days_to_expiration'] / 365.0
    
    # Calcula delta se não estiver disponível ou for inválido
    def get_or_calculate_delta(row):
        # Verifica se delta da API é válido
        if pd.notna(row['delta']) and isinstance(row['delta'], (int, float)) and row['delta'] != 0:
            # print(f"DEBUG: Usando delta da API: {row['delta']} para {row['symbol']}")  # Debug removido para produção
            return float(row['delta'])
        
        # Calcula usando Black-Scholes
        calculated_delta = calculate_black_scholes_delta(
            spot_price=stock_price,
            strike=row['strike'],
            time_to_expiry_years=row['time_to_expiry_years'],
            option_type=row['option_type']
        )
        # print(f"DEBUG: Calculando delta para {row['symbol']}: {calculated_delta}")  # Debug removido para produção
        return calculated_delta
    
    df['calculated_delta'] = df.apply(get_or_calculate_delta, axis=1)
    
    # Calcula probabilidade de exercício usando delta calculado
    df['exercise_probability'] = df.apply(
        lambda row: calculate_exercise_probability(row['calculated_delta'], row['option_type']),
        axis=1
    )
    
    # Filtra por baixo risco
    low_risk_mask = df['exercise_probability'] <= max_exercise_prob
    
    return df[low_risk_mask]


def filter_implied_volatility(df: pd.DataFrame, stock_price: float, max_iv: float = 0.80) -> pd.DataFrame:
    """
    Filtra opções por volatilidade implícita máxima.
    
    Args:
        df: DataFrame com opções
        stock_price: Preço atual da ação
        max_iv: Volatilidade implícita máxima (decimal, ex: 0.80 = 80%)
    
    Returns:
        DataFrame filtrado
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Calcula dias até vencimento em anos (se não existir)
    if 'time_to_expiry_years' not in df.columns:
        today = datetime.now()
        df['days_to_expiration'] = (df['expiration'] - today).dt.days
        df['time_to_expiry_years'] = df['days_to_expiration'] / 365.0
    
    # Estima volatilidade implícita
    df['implied_volatility'] = df.apply(
        lambda row: calculate_implied_volatility_estimate(
            row['last'], stock_price, row['strike'], 
            row['time_to_expiry_years'], row['option_type']
        ),
        axis=1
    )
    
    # Filtra por IV máxima
    iv_mask = df['implied_volatility'] <= max_iv
    
    return df[iv_mask]


def filter_good_return_options(df: pd.DataFrame, min_annual_return: float = 2.0, max_annual_return: float = 50.0) -> pd.DataFrame:
    """
    Filtra opções com retorno anualizado realista.
    
    Args:
        df: DataFrame com opções
        min_annual_return: Retorno anualizado mínimo (%) - padrão 2%
        max_annual_return: Retorno anualizado máximo (%) - padrão 50%
    
    Returns:
        DataFrame filtrado
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Calcula dias até vencimento
    today = datetime.now()
    df['days_to_expiration'] = (df['expiration'] - today).dt.days
    
    # Filtro de vencimento: entre 15 e 60 dias
    df = df[(df['days_to_expiration'] >= 15) & (df['days_to_expiration'] <= 60)]
    
    if df.empty:
        return df
    
    # Calcula retorno anualizado usando sempre o strike como base
    df['annualized_return'] = df.apply(
        lambda row: calculate_annualized_return(
            row['last'],
            row['strike'],
            row['days_to_expiration']
        ),
        axis=1
    )
    
    # Filtra por faixa de retorno realista (2% a 50% a.a.)
    good_return_mask = (
        (df['annualized_return'] >= min_annual_return) &
        (df['annualized_return'] <= max_annual_return)
    )
    
    return df[good_return_mask]


def calculate_implied_volatility_estimate(premium: float, stock_price: float, strike: float, 
                                         time_to_expiry_years: float, option_type: str = "CALL") -> float:
    """
    Estima volatilidade implícita usando aproximação simples.
    
    Args:
        premium: Prêmio da opção
        stock_price: Preço atual da ação
        strike: Strike da opção
        time_to_expiry_years: Tempo até vencimento em anos
        option_type: Tipo da opção
    
    Returns:
        Volatilidade implícita estimada (em decimal, ex: 0.30 = 30%)
    """
    if time_to_expiry_years <= 0 or premium <= 0:
        return 0.0
    
    # Aproximação simples: IV ≈ (prêmio / preço_subjacente) / sqrt(tempo)
    # Ajustado para opções brasileiras
    if option_type == "CALL":
        moneyness = stock_price / strike
    else:  # PUT
        moneyness = strike / stock_price
    
    # Fórmula aproximada para IV
    iv_estimate = (premium / stock_price) / (math.sqrt(time_to_expiry_years) * moneyness)
    
    # Limita entre 10% e 200% para valores realistas
    return max(0.10, min(2.0, iv_estimate))


def get_sector_mapping() -> Dict[str, str]:
    """
    Mapeamento de tickers para setores.
    
    Returns:
        Dicionário com ticker -> setor
    """
    return {
        # Energia
        "PETR4": "Energia", "PETR3": "Energia", "PRIO3": "Energia",
        
        # Mineração
        "VALE3": "Mineração", "CSNA3": "Mineração", "GGBR4": "Mineração", "USIM5": "Mineração", "GOAU4": "Mineração",
        
        # Bancos
        "ITUB4": "Financeiro", "BBDC4": "Financeiro", "BBAS3": "Financeiro", "BPAC11": "Financeiro",
        
        # Varejo
        "MGLU3": "Varejo", "LREN3": "Varejo", "VIIA3": "Varejo", "AMER3": "Varejo",
        
        # Consumo
        "ABEV3": "Consumo", "JBSS3": "Consumo", "BEEF3": "Consumo", "SUZB3": "Consumo",
        
        # Infraestrutura
        "B3SA3": "Infraestrutura", "CCRO3": "Infraestrutura", "RAIL3": "Infraestrutura",
        
        # Industrial
        "WEGE3": "Industrial", "RENT3": "Industrial", "KLBN11": "Industrial", "EMBR3": "Industrial",
        
        # Saúde
        "RADL3": "Saúde", "HAPV3": "Saúde", "RDOR3": "Saúde",
        
        # Utilities
        "ELET3": "Utilities", "CMIG4": "Utilities", "CPLE6": "Utilities",
        
        # Telecomunicações
        "VIVT3": "Telecom", "TIMS3": "Telecom"
    }


def calculate_quality_score(row: pd.Series, max_volume: float) -> float:
    """
    Calcula score de qualidade híbrido.
    
    Args:
        row: Linha do DataFrame com dados da opção
        max_volume: Volume máximo para normalização
    
    Returns:
        Score de qualidade (0-100)
    """
    # Componentes do score
    return_component = (row['Retorno a.a. (%)'] / 50.0) * 50  # Máx 50 pontos (assumindo 50% como teto)
    risk_component = (1 / max(row['Prob. Exercício (%)'], 0.1)) * 30  # Máx 30 pontos (quanto menor risco, melhor)
    volume_component = (row['Volume'] / max_volume) * 20  # Máx 20 pontos
    
    # Limita cada componente
    return_component = min(50, return_component)
    risk_component = min(30, risk_component)
    volume_component = min(20, volume_component)
    
    return return_component + risk_component + volume_component


def diversify_opportunities(df: pd.DataFrame, max_per_asset: int = 2, min_sectors: int = 3) -> pd.DataFrame:
    """
    Aplica diversificação por ativo e setor.
    
    Args:
        df: DataFrame com oportunidades
        max_per_asset: Máximo de opções por ativo
        min_sectors: Mínimo de setores diferentes
    
    Returns:
        DataFrame diversificado
    """
    if df.empty:
        return df
    
    sector_mapping = get_sector_mapping()
    df = df.copy()
    
    # Adiciona setor
    df['Setor'] = df['Ativo'].map(sector_mapping).fillna('Outros')
    
    # Calcula volume máximo para normalização do score
    max_volume = df['Volume'].max() if not df.empty else 1
    
    # Calcula score de qualidade
    df['Score_Qualidade'] = df.apply(lambda row: calculate_quality_score(row, max_volume), axis=1)
    
    # Ordena por score de qualidade (não só retorno)
    df = df.sort_values('Score_Qualidade', ascending=False)
    
    # Aplica diversificação por ativo
    diversified = []
    asset_count = {}
    
    for _, row in df.iterrows():
        asset = row['Ativo']
        
        # Limita opções por ativo
        if asset_count.get(asset, 0) < max_per_asset:
            diversified.append(row)
            asset_count[asset] = asset_count.get(asset, 0) + 1
    
    diversified_df = pd.DataFrame(diversified)
    
    if diversified_df.empty:
        return diversified_df
    
    # Verifica diversidade setorial
    sectors_present = diversified_df['Setor'].nunique()
    
    if sectors_present < min_sectors:
        # Tenta forçar mais setores
        remaining_df = df[~df.index.isin(diversified_df.index)]
        
        sectors_in_result = set(diversified_df['Setor'].unique())
        
        for _, row in remaining_df.iterrows():
            if row['Setor'] not in sectors_in_result and len(diversified) < 15:  # Limite flexível
                diversified.append(row)
                sectors_in_result.add(row['Setor'])
                
                if len(sectors_in_result) >= min_sectors:
                    break
        
        diversified_df = pd.DataFrame(diversified)
    
    # Remove coluna auxiliar do score
    if 'Score_Qualidade' in diversified_df.columns:
        diversified_df = diversified_df.drop('Score_Qualidade', axis=1)
    
    # Reordena por retorno anualizado para exibição final
    diversified_df = diversified_df.sort_values('Retorno a.a. (%)', ascending=False)
    
    return diversified_df


def create_justification(row: pd.Series, stock_price: float) -> str:
    """
    Cria justificativa para a oportunidade de PUT.
    
    Args:
        row: Linha do DataFrame com dados da PUT
        stock_price: Preço atual da ação
    
    Returns:
        String com justificativa
    """
    monthly_return = row['annualized_return'] / 12
    discount_pct = ((stock_price - row['strike']) / stock_price) * 100
    
    return f"PUT gera {monthly_return:.1f}% no mês, só compra ação {discount_pct:.0f}% mais barata"


def find_income_opportunities(
    client: OpLabClient,
    max_exercise_prob: float = 5.0,
    min_annual_return: float = 8.0,
    max_annual_return: float = 15.0,
    top_stocks: int = 20,
    top_opportunities: int = 10
) -> pd.DataFrame:
    """
    Busca as melhores oportunidades de renda no mercado de opções.
    
    Args:
        client: Cliente da OpLab
        max_exercise_prob: Probabilidade máxima de exercício (%)
        min_annual_return: Retorno anualizado mínimo (%)
        max_annual_return: Retorno anualizado máximo (%)
        top_stocks: Número de ações mais líquidas para analisar
        top_opportunities: Número de melhores oportunidades para retornar
    
    Returns:
        DataFrame com as melhores oportunidades ordenadas por retorno
    """
    # Busca ativos mais líquidos
    liquid_stocks = get_most_active_stocks(client, top_stocks)
    
    all_opportunities = []
    
    for ticker in liquid_stocks:
        try:
            # Busca preço da ação
            stock_price = client.get_underlying_price(ticker)
            
            # Busca grade de opções
            options_df = client.get_option_chain(ticker)
            
            if options_df.empty:
                continue
            
            # Remove opções sem preço (bid/ask/last)
            options_df = options_df.dropna(subset=['last'])
            options_df = options_df[options_df['last'] > 0]
            
            # Filtra apenas PUTs (estratégia de renda focada em PUT)
            options_df = options_df[options_df['option_type'] == 'PUT']
            
            if options_df.empty:
                continue
            
            # Aplica filtros sequencialmente
            # 1. Strikes realistas
            realistic_options = filter_realistic_strikes(options_df, stock_price)
            
            if realistic_options.empty:
                continue
            
            # 2. Prêmios razoáveis
            reasonable_options = filter_reasonable_premiums(realistic_options, stock_price)
            
            if reasonable_options.empty:
                continue
            
            # 3. Liquidez
            liquid_options = filter_liquid_options(reasonable_options)
            
            if liquid_options.empty:
                continue
            
            # 4. Volatilidade implícita
            iv_filtered_options = filter_implied_volatility(liquid_options, stock_price)
            
            if iv_filtered_options.empty:
                continue
            
            # 5. Baixo risco
            low_risk_options = filter_low_risk_options(iv_filtered_options, stock_price, max_exercise_prob)
            
            if low_risk_options.empty:
                continue
            
            # 6. Retorno realista
            good_return_options = filter_good_return_options(low_risk_options, min_annual_return, max_annual_return)
            
            if good_return_options.empty:
                continue
            
            # Prepara dados para resultado final (apenas PUTs)
            for _, row in good_return_options.iterrows():
                opportunity = {
                    'Ativo': ticker,
                    'Opção': row['symbol'],
                    'Tipo': 'Vender PUT',
                    'Strike': row['strike'],
                    'Vencimento': row['expiration'].strftime('%d/%m/%y'),
                    'Prêmio (R$)': row['last'],
                    'Retorno a.a. (%)': row['annualized_return'],
                    'Prob. Exercício (%)': row['exercise_probability'],
                    'Justificativa': create_justification(row, stock_price),
                    'Volume': row['volume']
                }
                
                # Adiciona Open Interest se disponível
                if 'open_interest' in row and pd.notna(row['open_interest']):
                    opportunity['Open Interest'] = row['open_interest']
                
                all_opportunities.append(opportunity)
                
        except Exception as e:
            # Continua para próximo ticker em caso de erro
            print(f"Erro ao processar {ticker}: {e}")
            continue
    
    if not all_opportunities:
        return pd.DataFrame()
    
    # Converte para DataFrame
    result_df = pd.DataFrame(all_opportunities)
    
    # Aplica diversificação inteligente
    diversified_df = diversify_opportunities(result_df, max_per_asset=1, min_sectors=3)
    
    # Retorna apenas os top N após diversificação
    return diversified_df.head(top_opportunities)


def render_income_opportunities_page():
    """
    Renderiza a página de Oportunidades de Renda no Streamlit.
    """
    import streamlit as st
    
    st.markdown("### 💰 Oportunidades de Renda")
    st.markdown("Automatiza a busca de PUTs líquidas com bom prêmio e baixo risco de exercício")
    
    # Formulário de parâmetros
    with st.form("income_params_form"):
        st.subheader("⚙️ Configurações")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            max_exercise_prob = st.number_input(
                "Prob. máx. exercício (%)",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.5,
                help="Probabilidade máxima de exercício da opção",
                key="income_max_exercise"
            )
        
        with col2:
            min_annual_return = st.number_input(
                "Retorno a.a. mínimo (%)",
                min_value=2.0,
                max_value=50.0,
                value=8.0,
                step=0.5,
                help="Retorno anualizado mínimo esperado",
                key="income_min_return"
            )
        
        with col3:
            max_annual_return = st.number_input(
                "Retorno a.a. máximo (%)",
                min_value=5.0,
                max_value=100.0,
                value=15.0,
                step=0.5,
                help="Retorno anualizado máximo (filtra oportunidades muito especulativas)",
                key="income_max_return"
            )
        
        with col4:
            top_opportunities = st.number_input(
                "Nº oportunidades",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="Número de melhores oportunidades para exibir",
                key="income_top_opportunities"
            )
        
        submitted = st.form_submit_button("🔍 Buscar Oportunidades", use_container_width=True)
    
    if submitted:
        with st.spinner("Analisando mercado de opções... Isso pode levar alguns minutos."):
            try:
                client = OpLabClient()
                
                opportunities_df = find_income_opportunities(
                    client=client,
                    max_exercise_prob=max_exercise_prob,
                    min_annual_return=min_annual_return,
                    max_annual_return=max_annual_return,
                    top_opportunities=top_opportunities
                )
                
                if opportunities_df.empty:
                    st.warning("⚠️ Nenhuma oportunidade encontrada com os critérios atuais.")
                    st.info("💡 Tente relaxar os filtros (menor retorno mínimo ou maior probabilidade de exercício)")
                else:
                    st.success(f"✅ Encontradas {len(opportunities_df)} oportunidades!")
                    
                    # Configuração da tabela
                    column_config = {
                        "Tipo": st.column_config.TextColumn(
                            "Tipo",
                            help="Estratégia cash-secured PUT: venda de PUT para gerar renda mensal"
                        ),
                        "Prêmio (R$)": st.column_config.NumberColumn(
                            "Prêmio (R$)",
                            help="Valor recebido pela venda da opção",
                            format="R$ %.3f"
                        ),
                        "Strike": st.column_config.NumberColumn(
                            "Strike",
                            help="Preço de exercício da opção",
                            format="R$ %.2f"
                        ),
                        "Retorno a.a. (%)": st.column_config.NumberColumn(
                            "Retorno a.a. (%)",
                            help="Retorno anualizado sobre o capital investido",
                            format="%.1f%%"
                        ),
                        "Prob. Exercício (%)": st.column_config.NumberColumn(
                            "Prob. Exercício (%)",
                            help="Probabilidade estimada de exercício da opção",
                            format="%.1f%%"
                        ),
                        "Volume": st.column_config.NumberColumn(
                            "Volume",
                            help="Volume de contratos negociados (liquidez)",
                            format="%d"
                        ),
                        "Open Interest": st.column_config.NumberColumn(
                            "Open Interest",
                            help="Contratos em aberto (interesse em exercício)",
                            format="%d"
                        )
                    }
                    
                    # DataFrame já está no formato correto para exibição
                    display_df = opportunities_df
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        column_config=column_config,
                        hide_index=True
                    )
                    
                    # Estatísticas resumidas
                    st.subheader("📊 Resumo")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Oportunidades", len(opportunities_df))
                    
                    with col2:
                        avg_return = opportunities_df['Retorno a.a. (%)'].mean()
                        st.metric("Retorno médio", f"{avg_return:.1f}%")
                    
                    with col3:
                        avg_prob = opportunities_df['Prob. Exercício (%)'].mean()
                        st.metric("Risco médio", f"{avg_prob:.1f}%")
                    
                    with col4:
                        unique_assets = opportunities_df['Ativo'].nunique()
                        st.metric("Ativos únicos", unique_assets)
                    
                    # Diversificação setorial
                    if 'Setor' in opportunities_df.columns:
                        st.subheader("🏭 Diversificação Setorial")
                        sector_counts = opportunities_df['Setor'].value_counts()
                        
                        # Exibe setores em colunas
                        sector_cols = st.columns(min(len(sector_counts), 4))
                        for i, (sector, count) in enumerate(sector_counts.head(4).items()):
                            with sector_cols[i % 4]:
                                st.metric(sector, f"{count} opções")
                        
            except Exception as e:
                st.error(f"❌ Erro ao buscar oportunidades: {e}")
    else:
        st.info("👆 Configure os parâmetros e clique em 'Buscar Oportunidades' para começar")
        
        # Explicação da estratégia
        with st.expander("ℹ️ Como funciona"):
            st.markdown("""
            **Fluxo de execução:**
            
            1. **Busca ativos líquidos** - Analisa os 20 papéis mais negociados da B3
            2. **Coleta grade de opções** - Busca todas as PUTs disponíveis para cada ativo
            3. **Filtra strikes realistas** - PUT ≤ -10% do preço atual
            4. **Filtra prêmios razoáveis** - Prêmio ≤ 20% do preço da ação
            5. **Aplica filtros de liquidez** - Volume ≥ 50 contratos e OI ≥ 500
            6. **Filtra volatilidade** - Volatilidade implícita ≤ 80% (evita ativos especulativos)
            7. **Aplica filtros de risco** - Probabilidade de exercício ≤ 5%
            8. **Filtra vencimento** - Entre 15 e 60 dias
            9. **Filtra retorno** - Retorno anualizado entre mínimo e máximo configurados
            10. **Aplica diversificação** - Máx 1 opção por ativo, mín 3 setores diferentes
            11. **Score de qualidade** - Combina retorno (50%), risco (30%) e liquidez (20%)
            
            **Fórmula de retorno:**
            - retorno_dia = (prêmio ÷ strike) ÷ dias_para_vencimento
            - retorno_aa = retorno_dia × 252
            
            **Score de qualidade:**
            - 50% baseado no retorno anualizado
            - 30% baseado no risco (quanto menor, melhor)
            - 20% baseado na liquidez (volume normalizado)
            
            **Diversificação inteligente:**
            - **Máximo 1 opção por ativo** - Evita concentração em um papel
            - **Mínimo 3 setores** - Garante diversificação setorial
            - **Ranking híbrido** - Não considera apenas retorno, mas qualidade geral
            
            **Estratégia de venda contemplada:**
            - **Vender PUT (Cash-Secured)**: Venda de PUT com delta baixo (≥ -0,10) - Gera renda mensal
            
            **Interpretação dos resultados:**
            - **Prob. Exercício < 5%**: Baixo risco de exercício
            - **Retorno a.a. configurável**: Faixa personalizada de retornos (padrão 8-15%)
            - **Diversificação**: Portfolio equilibrado entre ativos e setores
            - **Justificativa**: Explica o potencial da estratégia
            
            **Filtros de retorno:**
            - **Mínimo 8%**: Garante retorno superior à poupança/CDI
            - **Máximo 15%**: Filtra oportunidades muito especulativas
            - **Ajustável**: Pode ser modificado conforme perfil de risco
            - **Equilibrado**: Mantém foco em retornos realistas e sustentáveis
            """)
