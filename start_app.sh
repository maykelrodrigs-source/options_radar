#!/bin/bash

# Script para iniciar Options Radar com variáveis de ambiente configuradas
# Uso: ./start_app.sh

echo "🎯 Iniciando Options Radar..."

# Carrega variáveis de ambiente
if [ -f "src/config/env_config.sh" ]; then
    echo "🔧 Carregando configurações de ambiente..."
    source src/config/env_config.sh
else
    echo "❌ Arquivo env_config.sh não encontrado!"
    echo "💡 Certifique-se de que o arquivo existe em src/config/env_config.sh."
    exit 1
fi

# Verifica se as variáveis essenciais estão configuradas
if [ -z "$OPLAB_API_BASE_URL" ] || [ -z "$OPLAB_API_KEY" ]; then
    echo "❌ Variáveis OPLAB_API_BASE_URL ou OPLAB_API_KEY não configuradas!"
    echo "💡 Verifique o arquivo env_config.sh"
    exit 1
fi

echo "✅ Configurações carregadas com sucesso!"
echo "🚀 Iniciando Streamlit..."

# Garante que o pacote src/ seja importável
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Inicia o Streamlit
streamlit run src/app/app.py --server.headless true