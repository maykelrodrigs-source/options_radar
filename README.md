# Options Radar (Dividendos Sintéticos)

Módulo em Python para buscar a grade de opções (B3) via OpLab, filtrar CALLs e PUTs conforme critérios de dividendos sintéticos e exibir sugestões em uma interface Streamlit.

## Requisitos
- Python 3.11+
- Conta e chave de API da OpLab

## Instalação
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuração

### Método 1: Script automático (Recomendado)
```bash
./start_app.sh
```

### Método 2: Configuração manual
```bash
# Opção A: Usar arquivo de configuração
source env_config.sh
streamlit run app.py

# Opção B: Definir variáveis diretamente
export OPLAB_API_BASE_URL="https://api.oplab.com.br/v3"
export OPLAB_API_KEY="sua_chave_aqui"
export OPLAB_API_AUTH_HEADER="Access-Token"
export OPLAB_API_AUTH_SCHEME=""
export OPLAB_OPTION_CHAIN_ENDPOINT="/market/options/{ticker}"
export OPLAB_QUOTE_ENDPOINT="/market/stocks/{ticker}"
streamlit run app.py
```

### Método 3: Arquivo de configuração
1. Copie `config_example.py` para `config.py`
2. Configure suas credenciais no arquivo
3. Execute a aplicação normalmente

> **Nota**: A API Key fornecida é válida para testes. Para produção, use sua própria chave da OpLab.

## Assunções de mapeamento de campos
A resposta da OpLab precisa conter (direta ou indiretamente) as colunas abaixo, que são normalizadas pelo cliente:
- `symbol`
- `option_type` (ou `type`), valores aceitos: `CALL`/`PUT` (ou `C`/`P`)
- `strike` (ou `strikePrice`)
- `expiration` (ou `expirationDate`), idealmente ISO 8601
- `bid`, `ask`, `last` (pode usar `book.bid`/`book.ask`)
- `volume`
- `delta` (ou `greeks.delta`)

Exemplo mínimo de item aceito:
```json
{
  "symbol": "PETRK123",
  "option_type": "CALL",
  "strike": 45.0,
  "expiration": "2025-10-21",
  "bid": 0.45,
  "ask": 0.52,
  "last": 0.50,
  "volume": 350,
  "greeks": {"delta": 0.18}
}
```

## Critérios de seleção
- CALL coberta:
  - Strike ≥ 15% acima do preço atual
  - Delta ≤ 0,20
  - Vencimento entre 30 e 45 dias
  - Liquidez: volume > 100
- PUT coberta:
  - Strike ≤ 10% abaixo do preço atual
  - Delta ≥ -0,20
  - Vencimento entre 30 e 45 dias
  - Liquidez: volume > 100

Nota: A probabilidade de exercício é aproximada a partir do delta (heurística). O retorno (%) considera `prêmio_mid / spot`.

## Uso (biblioteca)
```python
from synthetic_dividends import find_synthetic_dividend_options

# Defina as variáveis OPLAB_* no ambiente

df = find_synthetic_dividend_options("PETR4")
print(df)
```

## Uso (Streamlit)
```bash
# Método 1: Script automático
./start_app.sh

# Método 2: Manual
streamlit run app.py
```

**Acesse**: http://localhost:8501

Insira o ticker (ex.: `PETR4`) e visualize a tabela com sugestões.

## Funcionalidades

### 💰 Dividendos Sintéticos
Busca opções para estratégia de dividendos sintéticos em um ticker específico.

### 📈 Radar de Direção
Análise técnica básica com indicadores SMA, RSI, MACD para decisão CALL/PUT.

### 🎯 Radar Profissional (NOVO!)
**Análise institucional em 6 camadas:**

**🏗️ Camada 1: Tendência Estrutural**
- Médias móveis múltiplas (SMA 10, 50, 100, 200)
- Golden/Death Cross
- Regime de volatilidade (ATR vs histórico)

**⚡ Camada 2: Momentum e Força**
- RSI múltiplos períodos (7, 14, 21)
- MACD histograma
- ADX (força da tendência)
- ROC (Rate of Change)

**📈 Camada 3: Volume e Fluxo**
- Volume ratios (5d vs 20d)
- OBV (On-Balance Volume)
- Accumulation/Distribution
- Direção do fluxo

**🎭 Camada 4: Sentimento via Opções**
- Put/Call Ratio
- Volatility Skew
- Volume distribution CALL vs PUT
- Market Bias

**🌍 Camada 5: Contexto Macro e Setorial**
- Análise setorial (PETR, VALE, bancos)
- Contexto macro (taxa de juros, commodities)
- Bias contextual

**🎯 Camada 6: Modelo de Decisão**
- Score normalizado (-1 a +1)
- Ponderação inteligente por camada
- Critérios claros: ≥ +0.3 CALL, ≤ -0.3 PUT, entre -0.3 e +0.3 NEUTRO
- Estratégias específicas por confiança

### 💸 Oportunidades de Renda
Automatiza a busca de opções líquidas com bom prêmio e baixo risco de exercício em todo o mercado:

**Fluxo de execução:**
1. Busca ativos mais líquidos (top 20 papéis)
2. Coleta grade de opções de cada ativo
3. Aplica filtros de liquidez (volume ≥ R$ 50k, OI ≥ 500)
4. Aplica filtros de risco (probabilidade exercício ≤ 5%)
5. Aplica filtros de retorno (retorno anualizado ≥ 6%)
6. Ordena por melhor retorno anualizado

**Parâmetros configuráveis:**
- Probabilidade máxima de exercício (padrão 5%)
- Retorno anualizado mínimo (padrão 6%)
- Número de oportunidades exibidas (padrão 10)

## Estrutura
- `oplab_client.py`: cliente para autenticação e consultas (cotação, option chain, most actives)
- `synthetic_dividends.py`: filtros e tabela de sugestões para dividendos sintéticos
- `direction_radar.py`: análise técnica básica e radar de direção
- `professional_analysis.py`: sistema de análise profissional em 6 camadas
- `professional_radar.py`: interface do radar profissional
- `income_opportunities.py`: busca automatizada de oportunidades de renda
- `app.py`: UI (Streamlit) com 4 abas

## Roadmap
- Função opcional de gráfico de payoff (Plotly)
- Ajustes finos de liquidez (ex.: considerar open interest, spreads)
- Parametrização dos limites (%) e janelas de vencimento
- Integração com mais endpoints da OpLab para dados em tempo real
