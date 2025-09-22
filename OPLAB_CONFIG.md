# 🔧 Configuração da API OpLab para Dados Reais

## 📋 Variáveis de Ambiente Necessárias

Para usar dados históricos reais da B3 via OpLab, configure as seguintes variáveis de ambiente:

### **Configuração Básica**
```bash
# URL base da API OpLab
export OPLAB_API_BASE_URL="https://api.oplab.com.br"

# Sua chave de API OpLab
export OPLAB_API_KEY="sua_chave_aqui"

# Endpoint para candles históricos
export OPLAB_CANDLES_ENDPOINT="/v1/candles/{ticker}"

# Endpoints existentes (já configurados)
export OPLAB_OPTION_CHAIN_ENDPOINT="/v1/options/chain?symbol={ticker}"
export OPLAB_QUOTE_ENDPOINT="/v1/quotes/{ticker}"
export OPLAB_MOST_ACTIVES_ENDPOINT="/v1/stocks/most_actives"
```

### **Configuração via arquivo .env**
Crie um arquivo `.env` na raiz do projeto:

```env
OPLAB_API_BASE_URL=https://api.oplab.com.br
OPLAB_API_KEY=sua_chave_aqui
OPLAB_CANDLES_ENDPOINT=/v1/candles/{ticker}
OPLAB_OPTION_CHAIN_ENDPOINT=/v1/options/chain?symbol={ticker}
OPLAB_QUOTE_ENDPOINT=/v1/quotes/{ticker}
OPLAB_MOST_ACTIVES_ENDPOINT=/v1/stocks/most_actives
```

## 🔍 Como Obter sua Chave da API

1. Acesse o portal do OpLab
2. Faça login em sua conta
3. Vá para seção "API" ou "Desenvolvedores"
4. Gere uma nova chave de API
5. Copie a chave e configure na variável `OPLAB_API_KEY`

## 🧪 Testando a Configuração

Execute o teste de integração:

```bash
python test_oplab_integration.py
```

### **Resultado Esperado:**
```
🚀 Iniciando testes de integração OpLab
✅ Variáveis de ambiente configuradas
✅ Provider criado com sucesso
✅ 30 candles carregados
✅ Colunas: ['date', 'open', 'high', 'low', 'close', 'volume']
🎉 Todos os testes passaram! Integração OpLab está funcionando.
```

## 📊 Endpoints da API OpLab

### **Candles Históricos**
```
GET /v1/candles/{ticker}?start=2024-01-01&end=2024-12-31&interval=1d
```

**Parâmetros:**
- `ticker`: Código da ação (ex: PETR4)
- `start`: Data inicial (YYYY-MM-DD)
- `end`: Data final (YYYY-MM-DD)  
- `interval`: Granularidade (1d para diário)

**Resposta Esperada:**
```json
{
  "results": [
    {
      "date": "2024-01-01",
      "open": 35.50,
      "high": 36.20,
      "low": 35.10,
      "close": 35.80,
      "volume": 1500000
    }
  ]
}
```

## ⚠️ Troubleshooting

### **Erro: "OPLAB_API_KEY não configurado"**
- Verifique se a variável de ambiente está definida
- Se usando .env, instale: `pip install python-dotenv`

### **Erro: "Sem dados para o período"**
- Verifique se o ticker existe (ex: PETR4, VALE3)
- Confirme se as datas estão corretas
- Teste com período menor primeiro

### **Erro: "401 Unauthorized"**
- Chave de API inválida ou expirada
- Verifique se a chave está correta
- Confirme se tem permissões para dados históricos

### **Erro: "404 Not Found"**
- Endpoint incorreto
- Verifique `OPLAB_CANDLES_ENDPOINT`
- Consulte documentação da OpLab

## 🔄 Fallback para Simulação

Se a API não estiver disponível, o sistema pode usar dados simulados:

```python
# Use o backup de simulação temporariamente
from data_simulation_backup import get_price_history
```

## 📈 Benefícios dos Dados Reais

### **Antes (Simulação):**
- Dados artificiais com padrões previsíveis
- Mesma "performance" sempre
- Não reflete realidade do mercado

### **Depois (OpLab):**
- ✅ Candles reais da B3
- ✅ Backtests com dados históricos verdadeiros  
- ✅ Validação real do ProfessionalAnalyzer
- ✅ Resultados confiáveis para produção

## 🎯 Próximos Passos

1. Configure as variáveis de ambiente
2. Execute o teste de integração
3. Rode um backtest com dados reais
4. Compare resultados com simulação anterior
5. Ajuste parâmetros do ProfessionalAnalyzer se necessário

---

**Dados reais = Backtests confiáveis = Decisões melhores** 🎯
