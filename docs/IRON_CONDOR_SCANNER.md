# Scanner de Iron Condor

## Visão Geral

O Scanner de Iron Condor é uma ferramenta que busca automaticamente oportunidades de estratégias Iron Condor baseadas em probabilidade de sucesso. A estratégia é ideal para mercados neutros onde esperamos que o preço do ativo permaneça dentro de um range específico.

## O que é Iron Condor?

Iron Condor é uma estratégia neutra de opções que combina:

- **Venda de CALL OTM** + **Compra de CALL mais alta** (trava de alta)
- **Venda de PUT OTM** + **Compra de PUT mais baixa** (trava de baixa)

### Características:
- **Lucro máximo**: Prêmio líquido recebido
- **Perda máxima**: Diferença entre strikes menos prêmio recebido
- **Probabilidade de sucesso**: Geralmente 75%+ (configurável)
- **Estratégia**: Neutra (não depende de direção do mercado)

## Como Usar

### 1. Acesso
- Abra a aplicação Options Radar
- Clique na aba **"🦅 Iron Condor"**

### 2. Configuração
- **Ticker**: Digite o código do ativo (ex: PETR4, VALE3, ITUB4)
- **Prob. Mínima**: Probabilidade mínima de sucesso (60% - 95%)
- **Vencimento Máx**: Vencimento máximo em dias (15, 30, 45, 60)

### 3. Filtros de Qualidade
- **Prêmio Mínimo**: Prêmio líquido mínimo em R$ (R$ 0,20+)
- **Filtros Automáticos Aplicados**:
  - ✅ Relação risco/retorno ≥ 80%
  - ✅ Largura das asas: 2-8% do preço
  - ✅ Probabilidade ≥ configurada
  - ✅ Prêmio mínimo ≥ configurado

### 4. Análise
Clique em **"🔍 Buscar Oportunidades"** para:
- Consultar opções do ativo via OpLab
- Filtrar opções com vencimento adequado
- Calcular probabilidades usando distribuição normal
- Montar estruturas Iron Condor viáveis
- Ordenar por retorno esperado

## Resultados

### Tabela de Oportunidades
A tabela mostra:
- **Prob. Sucesso**: Chance do preço ficar no range
- **Range Sucesso**: Faixa de preços para lucro
- **Prêmio Líquido**: Valor recebido pela estratégia
- **Risco Máximo**: Perda máxima possível
- **Ret. Esperado**: Retorno esperado anualizado

### Melhor Oportunidade
Detalhes da estrutura com melhor retorno esperado:
- Métricas principais
- Estrutura detalhada (strikes e preços)
- Gráfico de payoff
- Range de sucesso

### Exportação
- Download em JSON com todas as oportunidades
- Dados completos para análise externa

## Métricas Calculadas

### Probabilidade de Sucesso
Baseada em:
- Preço atual do ativo
- Volatilidade implícita das opções
- Distribuição normal acumulada
- Tempo até vencimento

### Retorno Esperado
```
Retorno Esperado = (Prob. Sucesso × Prêmio) - (Prob. Perda × Risco)
```

### Anualização
Retorno anualizado considerando o tempo até vencimento.

## Parâmetros Recomendados

### Para Iniciantes
- **Prob. Mínima**: 75%
- **Vencimento**: 30-45 dias
- **Ativos**: PETR4, VALE3, ITUB4 (mais líquidos)

### Para Experientes
- **Prob. Mínima**: 70-80%
- **Vencimento**: 15-60 dias (dependendo da volatilidade)
- **Ativos**: Qualquer com boa liquidez

## Exemplo de Estrutura

```
Ativo: PETR4
Vencimento: 2025-01-17 (30 dias)
Probabilidade: 76%

CALLs (Trava de Alta):
- Venda CALL: Strike R$ 32.00 → Prêmio R$ 1.20
- Compra CALL: Strike R$ 35.00 → Custo R$ 0.30

PUTs (Trava de Baixa):
- Venda PUT: Strike R$ 28.00 → Prêmio R$ 1.10
- Compra PUT: Strike R$ 25.00 → Custo R$ 0.20

Range de Sucesso: R$ 28.00 - R$ 32.00
Prêmio Líquido: R$ 1.80
Risco Máximo: R$ 1.20
Retorno Esperado: 12.5% a.a.
```

## Gestão de Risco

### Antes da Operação
- Verificar liquidez das opções
- Confirmar spreads bid-ask aceitáveis
- Validar margem disponível
- Definir stop-loss

### Durante a Operação
- Monitorar preço do ativo
- Ajustar posição se necessário
- Fechar antecipadamente se favorável
- Gerenciar exposição

### Após o Vencimento
- Analisar resultado
- Documentar aprendizado
- Ajustar parâmetros futuros

## Limitações

- Depende da disponibilidade de dados do OpLab
- Não considera comissões e impostos
- Baseado em volatilidade histórica
- Não garante resultados futuros

## Dicas

1. **Liquidez**: Priorize ativos com maior volume de opções
2. **Timing**: Evite vencimentos muito próximos
3. **Diversificação**: Não concentre todo capital em uma operação
4. **Educação**: Estude antes de operar com dinheiro real
5. **Simulação**: Pratique com valores pequenos inicialmente

## Melhorias Implementadas

### Filtros de Qualidade Avançados
- **Prêmio Mínimo**: Filtra oportunidades de baixa qualidade (R$ 0,20+)
- **Relação Risco/Retorno**: Prêmio deve ser ≥ 80% do risco máximo
- **Largura das Asas**: Otimizada entre 2-8% do preço do ativo
- **Probabilidade Configurável**: Flexibilidade de 60% a 95%

### Resultados Esperados
- **Menos Oportunidades**: Scanner retorna apenas as melhores
- **Maior Qualidade**: Todas as oportunidades têm potencial de lucro
- **Retorno Positivo**: Expectativa de retorno esperado positivo
- **Risco Controlado**: Relação risco/retorno adequada

### Comparação Antes vs Depois
| Aspecto | Antes | Depois |
|---------|-------|--------|
| Oportunidades | 2-3 por ticker | 1-2 de alta qualidade |
| Prêmio Médio | R$ 0,14 | R$ 0,30+ |
| Retorno Esperado | -21,2% | +10%+ |
| Relação R/R | < 50% | ≥ 80% |

## Suporte

Para dúvidas ou problemas:
- Verifique a configuração do OpLab
- Consulte a documentação das APIs
- Teste com diferentes parâmetros
- Analise os logs de erro

---

**⚠️ Aviso Legal**: Este scanner é uma ferramenta educacional. Não constitui recomendação de investimento. Operações com opções envolvem risco significativo. Sempre consulte um profissional qualificado antes de investir.


