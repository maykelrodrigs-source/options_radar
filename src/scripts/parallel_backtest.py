import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from typing import Dict, Tuple, List

import pandas as pd

from src.core.backtesting.backtest import BacktestEngine, BacktestProfessionalAnalyzer
from src.core.data.data import get_historical_data


# Cache simples em memória por (ticker, start, end)
@lru_cache(maxsize=256)
def cached_get(ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
    df = get_historical_data(ticker, start_str, end_str)
    # Converte para tipos básicos para ser cacheável via lru_cache (usa pickle interno)
    return df


def run_cfg(args: Tuple[str, datetime, datetime, int, float, Dict[str, float]]) -> Tuple[Tuple, float, int]:
    ticker, start, end, eval_days, threshold, weights = args
    analyzer = BacktestProfessionalAnalyzer(horizon='médio')
    analyzer.decision_threshold = threshold
    analyzer.layer_weights = weights

    engine = BacktestEngine(
        success_threshold=3.0,
        analyzer=analyzer,
        horizon='médio',
        data_provider=cached_get,
    )

    result = engine.run_backtest(
        ticker=ticker,
        start_date=start,
        end_date=end,
        evaluation_days=eval_days,
        rolling_window=5,
        min_history_days=300,
    )
    return (ticker, eval_days, threshold, tuple(sorted(weights.items()))), result.overall_accuracy, result.total_signals


def run_parallel(
    tickers: List[str],
    start: datetime,
    end: datetime,
    eval_days_list: List[int],
    thresholds: List[float],
    weights_list: List[Dict[str, float]],
    max_workers: int = 4,
):
    configs = []
    for t in tickers:
        for e in eval_days_list:
            for th in thresholds:
                for w in weights_list:
                    configs.append((t, start, end, e, th, w))

    results = []
    start_ts = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_cfg, cfg): cfg for cfg in configs}
        for fut in as_completed(futures):
            try:
                cfg_sig, acc, num = fut.result()
                results.append((cfg_sig, acc, num))
                t, e, th, w = cfg_sig[0], cfg_sig[1], cfg_sig[2], dict(cfg_sig[3])
                print(f"done: {t} eval={e} thr={th} -> acc={acc:.1f}% signals={num} weights={w}")
            except Exception as e:
                print("error on cfg:", futures[fut], e)

    dur = time.time() - start_ts
    print(f"Completed {len(results)} configs in {dur:.1f}s")
    if results:
        best = max(results, key=lambda x: x[1])
        print("BEST:", best)
    return results


if __name__ == "__main__":
    end = datetime(2024, 12, 31)
    start = datetime(2024, 1, 1)
    tickers = ["ITUB4", "PETR4", "VALE3"]
    eval_days_list = [10, 12, 15, 20]
    thresholds = [0.2, 0.25, 0.3, 0.35]
    weights_list = [
        {'trend': 0.45, 'momentum': 0.25, 'volume': 0.15, 'sentiment': 0.10, 'macro': 0.05},
        {'trend': 0.25, 'momentum': 0.45, 'volume': 0.15, 'sentiment': 0.10, 'macro': 0.05},
        {'trend': 0.30, 'momentum': 0.30, 'volume': 0.20, 'sentiment': 0.15, 'macro': 0.05},
    ]
    run_parallel(tickers, start, end, eval_days_list, thresholds, weights_list, max_workers=4)










