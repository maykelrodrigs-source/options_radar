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

## Variáveis de ambiente (OpLab)
Defina no seu ambiente (ex.: `.env` + `export`, ou no shell):

- `OPLAB_API_BASE_URL` (ex.: `https://api.oplab.com.br`)
- `OPLAB_API_KEY` (sua chave/token)
- `OPLAB_API_AUTH_HEADER` (padrão: `Authorization`)
- `OPLAB_API_AUTH_SCHEME` (padrão: `Bearer`)
- `OPLAB_OPTION_CHAIN_ENDPOINT` (ex.: `/v1/options/chain?symbol={ticker}`)
- `OPLAB_QUOTE_ENDPOINT` (ex.: `/v1/quotes/{ticker}`)

> Observação: Os caminhos exatos podem variar conforme o plano/versão da API da OpLab. Use placeholders `{ticker}` para interpolação automática.

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
streamlit run app.py
```
Insira o ticker (ex.: `PETR4`) e visualize a tabela com sugestões.

## Estrutura
- `oplab_client.py`: cliente para autenticação e consultas (cotação e option chain)
- `synthetic_dividends.py`: filtros e tabela de sugestões
- `app.py`: UI (Streamlit)

## Roadmap
- Função opcional de gráfico de payoff (Plotly)
- Ajustes finos de liquidez (ex.: considerar open interest, spreads)
- Parametrização dos limites (%) e janelas de vencimento
