from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from typing import Dict, Tuple

from src.core.backtesting.backtest import BacktestEngine, BacktestProfessionalAnalyzer
from src.core.data.data import get_historical_data_cached


def run_cfg(args: Tuple[str, datetime, datetime, int, float, Dict[str, float]]):
    ticker, start, end, eval_days, threshold, weights = args
    analyzer = BacktestProfessionalAnalyzer(horizon='curto')
    analyzer.decision_threshold = threshold
    analyzer.layer_weights = weights
    engine = BacktestEngine(success_threshold=3.0, analyzer=analyzer, horizon='curto', data_provider=get_historical_data_cached)
    res = engine.run_backtest(ticker=ticker, start_date=start, end_date=end, evaluation_days=eval_days, rolling_window=5, min_history_days=300)
    return (ticker, eval_days, threshold, weights), res.overall_accuracy, res.total_signals


if __name__ == "__main__":
    TICKERS = ["ITUB4", "PETR4", "VALE3"]
    end = datetime(2024, 12, 31)
    start = datetime(2024, 1, 1)
    EVALS = [10, 12, 15]
    THRS = [0.2, 0.25, 0.3]
    WEIGHTS = [
        {'trend': 0.30, 'momentum': 0.30, 'volume': 0.20, 'sentiment': 0.15, 'macro': 0.05},
        {'trend': 0.45, 'momentum': 0.25, 'volume': 0.15, 'sentiment': 0.10, 'macro': 0.05},
        {'trend': 0.25, 'momentum': 0.45, 'volume': 0.15, 'sentiment': 0.10, 'macro': 0.05},
    ]

    configs = [(t, start, end, e, th, w) for t, e, th, w in product(TICKERS, EVALS, THRS, WEIGHTS)]

    results = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_cfg, cfg): cfg for cfg in configs}
        for fut in as_completed(futs):
            try:
                (t, e, th, w), acc, num = fut.result()
                results.append(((t, e, th, w), acc, num))
                print(f"done: {t} eval={e} thr={th} -> acc={acc:.1f}% signals={num}")
            except Exception as e:
                print("error:", futs[fut], e)

    if results:
        best = max(results, key=lambda x: x[1])
        (t, e, th, w), acc, num = best
        print("BEST:", {"ticker": t, "eval_days": e, "threshold": th, "overall_accuracy": acc, "signals": num, "weights": w})
    else:
        print("NO_RESULTS")








