from datetime import datetime
from typing import Dict, Tuple

from src.core.backtesting.backtest import BacktestEngine, BacktestProfessionalAnalyzer
from src.core.data.data import get_historical_data_cached


def run_backtest_once(ticker: str, start: datetime, end: datetime, eval_days: int, roll: int, thr: float, weights: Dict[str, float]) -> Tuple[float, int]:
    analyzer = BacktestProfessionalAnalyzer(horizon='curto')
    analyzer.decision_threshold = thr
    analyzer.layer_weights = weights
    engine = BacktestEngine(success_threshold=3.0, analyzer=analyzer, horizon='curto', data_provider=get_historical_data_cached)
    res = engine.run_backtest(ticker, start, end, evaluation_days=eval_days, rolling_window=roll, min_history_days=300)
    return float(res.overall_accuracy), int(res.total_signals)


def walk_forward_itub4():
    ticker = "ITUB4"
    # 2023 como treino, 2024 como teste
    train_start, train_end = datetime(2023,1,1), datetime(2023,12,31)
    test_start, test_end = datetime(2024,1,1), datetime(2024,12,31)

    evals = [10, 12]
    rolls = [4, 5]
    thrs = [0.20, 0.22, 0.25, 0.28]
    weights_list = [
        ('balanced', {'trend':0.30,'momentum':0.30,'volume':0.20,'sentiment':0.15,'macro':0.05}),
        ('trend_heavy', {'trend':0.40,'momentum':0.30,'volume':0.15,'sentiment':0.10,'macro':0.05}),
        ('momentum_heavy', {'trend':0.25,'momentum':0.45,'volume':0.15,'sentiment':0.10,'macro':0.05}),
    ]

    best = (-1.0, None)
    print("🧪 Otimizando no treino (2023)...")
    for e in evals:
        for r in rolls:
            for t in thrs:
                for name, w in weights_list:
                    acc, sig = run_backtest_once(ticker, train_start, train_end, e, r, t, w)
                    print(f"train acc={acc:.2f}% sig={sig} | eval={e} roll={r} thr={t} w={name}")
                    if acc > best[0]:
                        best = (acc, (e, r, t, name, w))

    print("🏆 Melhor no treino:", best)
    if best[1] is None:
        print("Sem configuração vencedora")
        return

    e, r, t, name, w = best[1]
    print("🔁 Validando no teste (2024)...")
    test_acc, test_sig = run_backtest_once(ticker, test_start, test_end, e, r, t, w)
    print({
        'ticker': ticker,
        'train_best_acc': round(best[0],2),
        'test_acc': round(test_acc,2),
        'test_signals': test_sig,
        'eval_days': e,
        'rolling': r,
        'threshold': t,
        'weights': name
    })


if __name__ == "__main__":
    walk_forward_itub4()








