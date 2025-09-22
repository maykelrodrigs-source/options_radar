# 🔬 Módulo de Backtest do ProfessionalAnalyzer

Este módulo implementa um sistema completo de backtest para validar a eficácia do ProfessionalAnalyzer em dados históricos.

## 📁 Arquivos Criados

### `backtest.py`
- **BacktestEngine**: Engine principal para execução de backtests
- **BacktestResult**: Estrutura de dados com resultados completos
- **BacktestSignal**: Representa cada sinal individual gerado
- Funcionalidades:
  - Execução de backtest em janelas móveis (rolling window)
  - Validação de sinais CALL/PUT/NEUTRAL
  - Cálculo de estatísticas de acurácia
  - Exportação para DataFrame/CSV

### `plots.py`
- **BacktestPlotter**: Classe para gerar visualizações interativas
- Gráficos disponíveis:
  - Preço do ativo com marcadores de sinais
  - Breakdown de acurácia por tipo e confiança
  - Distribuição de retornos
  - Confiança vs Resultado (scatter plot)
  - Performance temporal (rolling accuracy)

### `run_backtest.py`
- Interface Streamlit completa para execução de backtests
- Configuração de parâmetros via sidebar
- Visualização interativa dos resultados
- Exportação de dados e relatórios
- Insights automáticos baseados na performance

## 🚀 Como Usar

### 1. Interface Streamlit (Recomendado)

```bash
streamlit run run_backtest.py
```

Ou acesse pela aba "🔬 Backtest" no app principal:

```bash
streamlit run app.py
```

### 2. Uso Programático

```python
from datetime import datetime, timedelta
from backtest import BacktestEngine
from plots import BacktestPlotter

# Cria engine
engine = BacktestEngine(success_threshold=3.0)

# Configura período
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # 2 anos

# Executa backtest
result = engine.run_backtest(
    ticker="PETR4",
    start_date=start_date,
    end_date=end_date,
    evaluation_days=20,
    rolling_window=5
)

# Visualiza resultados
engine.print_summary(result)

# Gera gráficos
plotter = BacktestPlotter()
fig = plotter.plot_price_with_signals(result)
fig.show()
```

## ⚙️ Parâmetros de Configuração

### BacktestEngine
- **ticker**: Código do ativo (ex: PETR4)
- **start_date/end_date**: Período do backtest
- **evaluation_days**: Dias úteis para avaliar se sinal acertou (padrão: 20)
- **rolling_window**: Intervalo entre análises em dias úteis (padrão: 5)
- **success_threshold**: Percentual mínimo para considerar acerto (padrão: 3.0%)

### Critérios de Validação
- **CALL**: Acerto se preço subiu > +threshold% no horizonte
- **PUT**: Acerto se preço caiu > -threshold% no horizonte
- **NEUTRAL**: Acerto se preço ficou entre ±threshold%

## 📊 Métricas Calculadas

### Estatísticas Gerais
- Taxa de acerto geral (%)
- Acurácia por tipo de sinal (CALL/PUT/NEUTRAL)
- Acurácia por nível de confiança (Alta ≥70%, Média 50-69%, Baixa <50%)

### Performance Qualitativa
- **Excelente**: ≥60% de acurácia
- **Boa**: 50-59% de acurácia
- **Regular**: 40-49% de acurácia
- **Precisa Melhorar**: <40% de acurácia

## 📈 Visualizações Disponíveis

1. **Preço com Sinais**: Gráfico de preço com marcadores coloridos para cada sinal
2. **Breakdown de Acurácia**: Barras comparando acurácia por tipo e confiança
3. **Distribuição de Retornos**: Histogramas dos retornos por tipo de sinal
4. **Confiança vs Resultado**: Scatter plot para identificar padrões
5. **Performance Temporal**: Acurácia móvel ao longo do tempo

## 🔧 Configuração para Produção

Para usar com dados reais da API OpLab, configure as variáveis de ambiente:

```bash
export OPLAB_API_BASE_URL="https://api.oplab.com.br"
export OPLAB_API_KEY="sua_chave_aqui"
# ... outras variáveis conforme oplab_client.py
```

## 📋 Exemplo de Saída

```
📊 RESUMO DO BACKTEST - PETR4
============================================================
📅 Período: 01/01/2023 a 01/01/2025
⏱️ Horizonte de avaliação: 20 dias úteis
🎯 Threshold de sucesso: ±3.0%

📈 ESTATÍSTICAS GERAIS:
• Total de sinais: 87
• Acurácia geral: 62.1%

📊 ACURÁCIA POR SINAL:
• CALL (29 sinais): 65.5%
• PUT (31 sinais): 58.1%
• NEUTRAL (27 sinais): 63.0%

🎯 ACURÁCIA POR CONFIANÇA:
• Alta confiança (≥70%, 32 sinais): 75.0%
• Média confiança (50-69%, 41 sinais): 56.1%
• Baixa confiança (<50%, 14 sinais): 42.9%

🎉 PERFORMANCE: EXCELENTE (≥60%)
============================================================
```

## 🎯 Insights Automáticos

O sistema gera insights automáticos baseados nos resultados:
- Performance geral vs benchmarks
- Eficácia de sinais de alta confiança
- Melhor tipo de sinal para o ativo
- Sugestões de otimização

## 📁 Exportação

- **CSV**: Tabela completa de sinais com resultados
- **PNG**: Gráficos interativos (via botão de câmera)
- **Texto**: Resumo executivo formatado para relatórios

## 🔍 Limitações Atuais

1. **Dados Simulados**: Por padrão usa dados simulados (módulo `data.py`)
2. **API Dependency**: Requer configuração da API OpLab para dados reais
3. **Horizonte Fixo**: Avalia todos os sinais no mesmo horizonte temporal
4. **Sem Custos**: Não considera custos de transação ou spread

## 🚀 Próximas Melhorias

- [ ] Integração com dados reais de múltiplas fontes
- [ ] Análise de custos de transação
- [ ] Backtests multi-ativo simultâneos
- [ ] Otimização automática de parâmetros
- [ ] Comparação com benchmarks de mercado
- [ ] Análise de drawdown e Sharpe ratio

---

**Desenvolvido para validar e otimizar o ProfessionalAnalyzer** 🎯
