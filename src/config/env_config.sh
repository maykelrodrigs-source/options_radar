#!/bin/bash

# Arquivo de configuração de variáveis de ambiente para Options Radar
# Execute: source env_config.sh

export OPLAB_API_BASE_URL="https://api.oplab.com.br/v3"
export OPLAB_API_KEY="s6Yrt1NqoroPyww/xirujxxy/sJSRL71kkkGscqad7Y/tm7bnhZ/kcb3Y4xZ3Cwa--eWKOcjng2AqZTiV5YIU69g==--ZDA4YWE0ODZmNTAwZWNiNzA3NmMzN2M0ZWNmNWYxYzA="
export OPLAB_API_AUTH_HEADER="Access-Token"
export OPLAB_API_AUTH_SCHEME=""
export OPLAB_OPTION_CHAIN_ENDPOINT="/market/options/{ticker}"
export OPLAB_QUOTE_ENDPOINT="/market/stocks/{ticker}"
export OPLAB_CANDLES_ENDPOINT="/v1/candles/{ticker}"
export OPLAB_MOST_ACTIVES_ENDPOINT="/market/most_actives"

echo "✅ Variáveis de ambiente configuradas para Options Radar"
echo "🔧 Para usar: source env_config.sh"
