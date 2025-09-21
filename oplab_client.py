import os
import json
from typing import Any, Dict, List, Optional, Tuple
import requests
import pandas as pd
from datetime import datetime


class OpLabClient:
    """
    Cliente configurável para a API do OpLab.

    Importante: Como a documentação pública de endpoints pode variar por plano/versão,
    os caminhos devem ser configurados por variáveis de ambiente para evitar suposições erradas.

    Variáveis de ambiente suportadas:
    - OPLAB_API_BASE_URL: Base da API (ex.: https://api.oplab.com.br)
    - OPLAB_API_KEY: Token/chave de acesso
    - OPLAB_API_AUTH_HEADER: Nome do header de autenticação (padrão: Authorization)
    - OPLAB_API_AUTH_SCHEME: Esquema do header (padrão: Bearer). Ex.: Bearer, Token, vazio
    - OPLAB_OPTION_CHAIN_ENDPOINT: Caminho para chain de opções. Suporta format: {ticker}
      Ex.: /v1/options/chain?symbol={ticker}
    - OPLAB_QUOTE_ENDPOINT: Caminho para cotação do subjacente. Suporta format: {ticker}
      Ex.: /v1/quotes/{ticker}

    Convenções esperadas nos dados de opções:
    - Cada opção deve conter campos que possamos normalizar para os seguintes nomes:
      symbol, option_type (CALL/PUT), strike, expiration, bid, ask, last, volume, delta
    - Campos aninhados como greeks.delta são aceitos.
    """

    def __init__(self,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 auth_header: Optional[str] = None,
                 auth_scheme: Optional[str] = None,
                 option_chain_endpoint: Optional[str] = None,
                 quote_endpoint: Optional[str] = None,
                 timeout_seconds: int = 20) -> None:
        self.base_url = (base_url or os.getenv("OPLAB_API_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("OPLAB_API_KEY", "")
        self.auth_header = auth_header or os.getenv("OPLAB_API_AUTH_HEADER", "Authorization")
        self.auth_scheme = auth_scheme or os.getenv("OPLAB_API_AUTH_SCHEME", "Bearer")
        self.option_chain_endpoint = option_chain_endpoint or os.getenv("OPLAB_OPTION_CHAIN_ENDPOINT", "")
        self.quote_endpoint = quote_endpoint or os.getenv("OPLAB_QUOTE_ENDPOINT", "")
        self.timeout_seconds = timeout_seconds

        if not self.base_url:
            raise ValueError("OPLAB_API_BASE_URL não configurado. Defina a variável de ambiente.")
        if not self.api_key:
            raise ValueError("OPLAB_API_KEY não configurado. Defina a variável de ambiente.")
        if not self.option_chain_endpoint:
            raise ValueError("OPLAB_OPTION_CHAIN_ENDPOINT não configurado. Defina a variável de ambiente.")
        if not self.quote_endpoint:
            raise ValueError("OPLAB_QUOTE_ENDPOINT não configurado. Defina a variável de ambiente.")

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self.auth_header:
            if self.auth_scheme:
                headers[self.auth_header] = f"{self.auth_scheme} {self.api_key}".strip()
            else:
                headers[self.auth_header] = self.api_key
        return headers

    def _get(self, endpoint_template: str, ticker: str) -> requests.Response:
        endpoint = endpoint_template.format(ticker=ticker)
        url = f"{self.base_url}{endpoint}" if endpoint.startswith("/") else f"{self.base_url}/{endpoint}"
        resp = requests.get(url, headers=self._build_headers(), timeout=self.timeout_seconds)
        if not resp.ok:
            raise RuntimeError(f"Falha ao consultar {url}: {resp.status_code} - {resp.text}")
        return resp

    def get_underlying_price(self, ticker: str) -> float:
        """Retorna o preço atual (último) do papel subjacente.

        Tenta identificar campos comuns: price, last, lastPrice, close, regularMarketPrice.
        Pode aceitar payloads onde a cotação venha em { 'symbol': ..., 'price': ... }
        ou em listas.
        """
        resp = self._get(self.quote_endpoint, ticker)
        data: Any = resp.json()

        # Se vier lista, use o primeiro elemento
        if isinstance(data, list) and data:
            data = data[0]

        candidates = [
            ("price", None), ("last", None), ("lastPrice", None), ("last_price", None), ("close", None), ("regularMarketPrice", None)
        ]

        # Busca em primeiro nível
        for key, _ in candidates:
            if isinstance(data, dict) and key in data:
                value = data[key]
                if isinstance(value, (int, float)):
                    return float(value)

        # Alguns provedores retornam preços em sub-objetos
        nested_candidates: List[Tuple[str, str]] = [
            ("quote", "last"), ("quote", "price"), ("data", "last"), ("data", "price")
        ]
        for outer, inner in nested_candidates:
            if isinstance(data, dict) and outer in data and isinstance(data[outer], dict) and inner in data[outer]:
                value = data[outer][inner]
                if isinstance(value, (int, float)):
                    return float(value)

        raise RuntimeError("Não foi possível identificar o preço atual no payload de cotação da OpLab.")

    def get_option_chain(self, ticker: str) -> pd.DataFrame:
        """Retorna DataFrame normalizado com a grade de opções (CALL e PUT) do ticker.

        Colunas normalizadas: symbol, option_type, strike, expiration, bid, ask, last, volume, delta
        """
        resp = self._get(self.option_chain_endpoint, ticker)
        
        # Verifica se a resposta está vazia
        if not resp.text.strip():
            return pd.DataFrame()
        
        data: Any = resp.json()

        # Muitos endpoints retornam {'results': [...]} ou lista direta
        options_payload: List[Dict[str, Any]]
        if isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list):
                options_payload = data["results"]
            elif "data" in data and isinstance(data["data"], list):
                options_payload = data["data"]
            else:
                # Tenta interpretar o dicionário como um único contrato (pouco comum)
                options_payload = [data]
        elif isinstance(data, list):
            options_payload = data
        else:
            raise RuntimeError("Formato inesperado do payload de opções da OpLab.")

        normalized: List[Dict[str, Any]] = []
        for raw in options_payload:
            normalized.append(self._normalize_option_record(raw))

        df = pd.DataFrame(normalized)

        # Conversões e limpeza básicas
        if not df.empty:
            # Tipos numéricos
            for col in ["strike", "bid", "ask", "last", "volume", "delta"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Datas
            if "expiration" in df.columns:
                df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")

            # Tipos (CALL/PUT)
            if "option_type" in df.columns:
                df["option_type"] = df["option_type"].astype(str).str.upper().str.strip()
                df["option_type"] = df["option_type"].replace({"C": "CALL", "CALL": "CALL", "P": "PUT", "PUT": "PUT"})

        return df

    def _normalize_option_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Tenta mapear campos comuns do payload para o esquema normalizado."""
        def get_nested(d: Dict[str, Any], *keys: str) -> Optional[Any]:
            cur: Any = d
            for k in keys:
                if not isinstance(cur, dict) or k not in cur:
                    return None
                cur = cur[k]
            return cur

        symbol = raw.get("symbol") or raw.get("ticker") or raw.get("optionSymbol")
        option_type = raw.get("option_type") or raw.get("type") or raw.get("optionType")
        strike = raw.get("strike") or raw.get("strikePrice")
        # v3 costuma usar due_date
        expiration = (
            raw.get("expiration")
            or raw.get("expirationDate")
            or raw.get("expiration_date")
            or raw.get("due_date")
            or raw.get("maturity")
        )
        bid = raw.get("bid") or raw.get("bestBid") or get_nested(raw, "book", "bid")
        ask = raw.get("ask") or raw.get("bestAsk") or get_nested(raw, "book", "ask")
        # usar close (último) quando disponível
        last = (
            raw.get("last")
            or raw.get("lastPrice")
            or raw.get("last_price")
            or raw.get("close")
            or raw.get("premium")
            or raw.get("tradePrice")
        )
        volume = raw.get("volume") or raw.get("open_interest") or raw.get("vol") or raw.get("totalVolume")
        delta = raw.get("delta") or get_nested(raw, "greeks", "delta")

        return {
            "symbol": symbol,
            "option_type": option_type,
            "strike": strike,
            "expiration": expiration,
            "bid": bid,
            "ask": ask,
            "last": last,
            "volume": volume,
            "delta": delta,
        }


__all__ = ["OpLabClient"]


