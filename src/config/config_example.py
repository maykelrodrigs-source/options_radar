"""
Arquivo de exemplo de configuração para o Options Radar.
Copie este arquivo para config.py e configure suas variáveis.
"""

# Configuração da API OpLab
OPLAB_API_BASE_URL = "https://api.oplab.com.br/v3"
OPLAB_API_KEY = "sua_chave_aqui"
OPLAB_API_AUTH_HEADER = "Access-Token"
OPLAB_API_AUTH_SCHEME = ""

# Endpoints da API
OPLAB_OPTION_CHAIN_ENDPOINT = "/market/options/{ticker}"
OPLAB_QUOTE_ENDPOINT = "/market/stocks/{ticker}"

# Configurações opcionais
OPLAB_REQUEST_TIMEOUT = 20
OPLAB_MAX_RETRIES = 3
