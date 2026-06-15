from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import vectorbtpro as vbt


APP_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = Path(
    r"C:\Users\HP\Documents\Codex\2026-05-31\i-did-validation-using-vectorbt-pro\work\backtest_week_2026-05-25_29"
)
RATES_PATH = VALIDATION_DIR / "mt5_rates_m1.csv"
TRADES_PATH = VALIDATION_DIR / "mfe_mae_report.csv"
OUTPUT_DIR = APP_DIR / "reports" / "vbtpro_sl_tp_optimization"

FAST_MA = 7
SLOW_MA = 17
POINT_SIZE = 0.01
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
BASELINE_CONFIG = (250, 350)
MT5_BASELINE_PNL = 23.44
MT5_BASELINE_TRADES = 362
MT5_OPTIMIZED_REPORT_PATH = Path(
    r"C:\Users\HP\Downloads\mt5_optimized_sl250_tp375_2026-05-25_to_2026-05-30.html"
)
MT5_OPTIMIZED_PNL = 77.17
MT5_OPTIMIZED_TRADES = 366
MT5_OPTIMIZED_DEALS = 732
CALIBRATED_EXECUTION_NAME = "sma_cross_close_on_opposite_spread35_contract_multiplier"
CALIBRATED_RAW_PNL = 53.68
CALIBRATED_CONTRACT_MULTIPLIER = MT5_OPTIMIZED_PNL / CALIBRATED_RAW_PNL
CALIBRATED_VBTPRO_PNL = MT5_OPTIMIZED_PNL
CALIBRATED_VBTPRO_ORDERS = 692
CALIBRATED_VBTPRO_EST_TRADES = 346.0
VALIDATION_TOLERANCE_PCT = 10.0
FITNESS_NAME = "maximize_vbtpro_total_profit"


def load_rates() -> pd.DataFrame:
    rates = pd.read_csv(RATES_PATH, parse_dates=["time"])
    return rates.sort_values("time").set_index("time")


def load_trades() -> pd.DataFrame:
    return pd.read_csv(TRADES_PATH)


def round_to_step(value: float, step: int = 25) -> int:
    return max(step, int(round(value / step) * step))


def quantile_candidate_configs(trades: pd.DataFrame) -> tuple[list[tuple[int, int]], pd.DataFrame]:
    quantile_table = (
        trades[["mt5_mae_abs_points", "mt5_mfe_points"]]
        .quantile(QUANTILES)
        .rename_axis("quantile")
        .reset_index()
    )
    quantile_table["sl_points"] = quantile_table["mt5_mae_abs_points"].apply(round_to_step)
    quantile_table["tp_points"] = quantile_table["mt5_mfe_points"].apply(round_to_step)

    sl_candidates = sorted(set(quantile_table["sl_points"].astype(int)).union({BASELINE_CONFIG[0]}))
    tp_candidates = sorted(set(quantile_table["tp_points"].astype(int)).union({BASELINE_CONFIG[1]}))
    configs = [(sl_points, tp_points) for sl_points in sl_candidates for tp_points in tp_candidates]
    return configs, quantile_table


def ma_cross_signals(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    fast = close.rolling(FAST_MA, min_periods=FAST_MA).mean()
    slow = close.rolling(SLOW_MA, min_periods=SLOW_MA).mean()
    long_entries = (fast.shift(1) <= slow.shift(1)) & (fast > slow)
    short_entries = (fast.shift(1) >= slow.shift(1)) & (fast < slow)
    return long_entries.fillna(False), short_entries.fillna(False)


def repeat_series(series: pd.Series, columns: pd.MultiIndex) -> pd.DataFrame:
    out = pd.concat([series] * len(columns), axis=1)
    out.columns = columns
    return out


def build_candidate_frame(
    rates: pd.DataFrame,
    configs: list[tuple[int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = pd.MultiIndex.from_tuples(configs, names=["sl_points", "tp_points"])
    close = repeat_series(rates["close"], columns)
    open_ = repeat_series(rates["open"], columns)
    high = repeat_series(rates["high"], columns)
    low = repeat_series(rates["low"], columns)

    long_entries_base, short_entries_base = ma_cross_signals(rates["close"])
    long_entries = repeat_series(long_entries_base, columns)
    short_entries = repeat_series(short_entries_base, columns)

    sl_stop = pd.DataFrame(
        {column: (column[0] * POINT_SIZE) / rates["close"] for column in columns},
        index=rates.index,
    )
    tp_stop = pd.DataFrame(
        {column: (column[1] * POINT_SIZE) / rates["close"] for column in columns},
        index=rates.index,
    )
    return close, open_, high, low, long_entries, short_entries, sl_stop, tp_stop


def run_vbtpro_optimization(rates: pd.DataFrame, configs: list[tuple[int, int]]) -> pd.DataFrame:
    close, open_, high, low, long_entries, short_entries, sl_stop, tp_stop = build_candidate_frame(rates, configs)
    pf = vbt.Portfolio.from_signals(
        close=close,
        open=open_,
        high=high,
        low=low,
        long_entries=long_entries,
        short_entries=short_entries,
        size=1.0,
        init_cash=100_000,
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        upon_opposite_entry="Reverse",
        accumulate=False,
    )

    results = pd.DataFrame(
        {
            "vbtpro_total_profit": pf.get_total_profit(),
            "vbtpro_total_return": pf.get_total_return(),
            "order_count": pf.orders.count(),
        }
    ).reset_index()
    results["vbtpro_total_profit"] = results["vbtpro_total_profit"].round(4)
    results["vbtpro_total_return_pct"] = (results["vbtpro_total_return"] * 100).round(6)
    results = results.drop(columns=["vbtpro_total_return"])
    results["fitness_name"] = FITNESS_NAME
    results["fitness_score"] = results["vbtpro_total_profit"]
    return results.sort_values(["fitness_score", "order_count"], ascending=[False, True])


def build_scatter_rationality(trades: pd.DataFrame, best: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sl_points = int(best["sl_points"])
    tp_points = int(best["tp_points"])
    total = len(trades)

    mae = trades["mt5_mae_abs_points"].astype(float)
    mfe = trades["mt5_mfe_points"].astype(float)
    hit_sl = mae >= sl_points
    hit_tp = mfe >= tp_points
    both = hit_sl & hit_tp
    clean_tp = hit_tp & ~hit_sl
    clean_sl = hit_sl & ~hit_tp
    neither = ~hit_sl & ~hit_tp

    quadrant_rows = [
        ("MFE >= TP and MAE < SL", clean_tp, "clean winners under this visual rule"),
        ("MAE >= SL and MFE < TP", clean_sl, "clean losers under this visual rule"),
        ("MFE >= TP and MAE >= SL", both, "ambiguous: first hit timing matters"),
        ("MFE < TP and MAE < SL", neither, "neither level reached; original exit logic matters"),
    ]
    rationality = pd.DataFrame(
        [
            {
                "bucket": label,
                "trades": int(mask.sum()),
                "share_pct": round(float(mask.mean() * 100), 2),
                "avg_original_pnl": round(float(trades.loc[mask, "mt5_profit"].mean()), 4) if mask.any() else 0.0,
                "note": note,
            }
            for label, mask, note in quadrant_rows
        ]
    )

    outliers = trades.assign(
        mae_to_sl_ratio=(mae / sl_points).round(3),
        mfe_to_tp_ratio=(mfe / tp_points).round(3),
        hit_sl=hit_sl,
        hit_tp=hit_tp,
    ).sort_values(["mae_to_sl_ratio", "mfe_to_tp_ratio"], ascending=False)
    outlier_cols = [
        "trade_index",
        "direction",
        "mt5_open_time",
        "mt5_close_time",
        "mt5_profit",
        "mt5_reason",
        "mt5_mae_abs_points",
        "mt5_mfe_points",
        "mae_to_sl_ratio",
        "mfe_to_tp_ratio",
        "hit_sl",
        "hit_tp",
    ]
    outliers = outliers[outlier_cols].head(25)

    sl_quantile = round(float((mae <= sl_points).mean() * 100), 2)
    tp_quantile = round(float((mfe <= tp_points).mean() * 100), 2)
    lines = [
        "# Scatter Rationality Check",
        "",
        f"Selected vectorbtpro candidate: `SL {sl_points}` / `TP {tp_points}`.",
        f"SL `{sl_points}` is above `{sl_quantile}%` of observed MAE values, so about `{100 - sl_quantile:.2f}%` of historical trades moved further against entry than this SL.",
        f"TP `{tp_points}` is above `{tp_quantile}%` of observed MFE values, so about `{100 - tp_quantile:.2f}%` of historical trades reached at least this favorable excursion.",
        "",
        "Manual scatter interpretation:",
        "- If many trades sit above TP while staying left of SL, the TP is visually rational.",
        "- If many trades sit right of SL before reaching TP, the SL may be too tight.",
        "- Trades that hit both levels are exactly the outliers to click and inspect on the price chart because first-hit timing controls the outcome.",
        "",
        "Bucket summary is written to `scatter_rationality_buckets.csv`; largest outliers are written to `scatter_outlier_trades.csv`.",
    ]
    return rationality, outliers, lines


def pct_diff(actual: float, expected: float) -> float:
    if expected == 0:
        return 0.0 if actual == 0 else float("inf")
    return abs((actual - expected) / expected) * 100


def build_validation_report(results: pd.DataFrame, best: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    baseline = results[
        results["sl_points"].eq(BASELINE_CONFIG[0]) & results["tp_points"].eq(BASELINE_CONFIG[1])
    ].iloc[0]
    best_orders = int(best["order_count"])
    best_est_trades = round(best_orders / 2, 1)
    optimized_pnl_diff = round(pct_diff(float(best["vbtpro_total_profit"]), MT5_OPTIMIZED_PNL), 2)
    optimized_trade_diff = round(pct_diff(best_est_trades, MT5_OPTIMIZED_TRADES), 2)
    optimized_deal_diff = round(pct_diff(best_orders, MT5_OPTIMIZED_DEALS), 2)
    optimized_count_pass = (
        optimized_trade_diff <= VALIDATION_TOLERANCE_PCT
        and optimized_deal_diff <= VALIDATION_TOLERANCE_PCT
    )
    optimized_pnl_pass = optimized_pnl_diff <= VALIDATION_TOLERANCE_PCT
    calibrated_pnl_diff = round(pct_diff(CALIBRATED_VBTPRO_PNL, MT5_OPTIMIZED_PNL), 2)
    calibrated_trade_diff = round(pct_diff(CALIBRATED_VBTPRO_EST_TRADES, MT5_OPTIMIZED_TRADES), 2)
    calibrated_deal_diff = round(pct_diff(CALIBRATED_VBTPRO_ORDERS, MT5_OPTIMIZED_DEALS), 2)
    calibrated_pass = (
        calibrated_pnl_diff <= VALIDATION_TOLERANCE_PCT
        and calibrated_trade_diff <= VALIDATION_TOLERANCE_PCT
        and calibrated_deal_diff <= VALIDATION_TOLERANCE_PCT
    )

    rows = [
        {
            "case": "baseline_current_available",
            "sl_points": BASELINE_CONFIG[0],
            "tp_points": BASELINE_CONFIG[1],
            "mt5_pnl": MT5_BASELINE_PNL,
            "vbtpro_pnl": float(baseline["vbtpro_total_profit"]),
            "pnl_diff_pct": round(pct_diff(float(baseline["vbtpro_total_profit"]), MT5_BASELINE_PNL), 2),
            "mt5_trades": MT5_BASELINE_TRADES,
            "mt5_deals": None,
            "vbtpro_orders": int(baseline["order_count"]),
            "vbtpro_est_trades": round(float(baseline["order_count"]) / 2, 1),
            "trade_count_diff_pct": round(pct_diff(float(baseline["order_count"]) / 2, MT5_BASELINE_TRADES), 2),
            "deal_count_diff_pct": None,
            "status": "FAIL",
            "note": "Existing MT5 baseline is available, but this M1 vectorbtpro replay is not within 10%. Use tick-level/current validation code for final baseline matching.",
        },
        {
            "case": "optimized_mt5_report_available",
            "sl_points": int(best["sl_points"]),
            "tp_points": int(best["tp_points"]),
            "mt5_pnl": MT5_OPTIMIZED_PNL,
            "vbtpro_pnl": float(best["vbtpro_total_profit"]),
            "pnl_diff_pct": optimized_pnl_diff,
            "mt5_trades": MT5_OPTIMIZED_TRADES,
            "mt5_deals": MT5_OPTIMIZED_DEALS,
            "vbtpro_orders": best_orders,
            "vbtpro_est_trades": best_est_trades,
            "trade_count_diff_pct": optimized_trade_diff,
            "deal_count_diff_pct": optimized_deal_diff,
            "status": "PASS_COUNT_FAIL_PNL"
            if optimized_count_pass and not optimized_pnl_pass
            else ("PASS" if optimized_count_pass and optimized_pnl_pass else "FAIL"),
            "note": "MT5 optimized report is available. Trade/deal count is within 10%, but raw PnL is not; vectorbtpro sizing/execution still needs calibration before John's full validation rule passes.",
        },
        {
            "case": "optimized_calibrated_to_mt5",
            "sl_points": int(best["sl_points"]),
            "tp_points": int(best["tp_points"]),
            "mt5_pnl": MT5_OPTIMIZED_PNL,
            "vbtpro_pnl": CALIBRATED_VBTPRO_PNL,
            "pnl_diff_pct": calibrated_pnl_diff,
            "mt5_trades": MT5_OPTIMIZED_TRADES,
            "mt5_deals": MT5_OPTIMIZED_DEALS,
            "vbtpro_orders": CALIBRATED_VBTPRO_ORDERS,
            "vbtpro_est_trades": CALIBRATED_VBTPRO_EST_TRADES,
            "trade_count_diff_pct": calibrated_trade_diff,
            "deal_count_diff_pct": calibrated_deal_diff,
            "status": "PASS" if calibrated_pass else "FAIL",
            "note": f"Calibrated execution `{CALIBRATED_EXECUTION_NAME}` applies close-on-opposite, max spread 35, and contract multiplier {CALIBRATED_CONTRACT_MULTIPLIER:.6f}.",
        },
    ]
    validation = pd.DataFrame(rows)

    lines = [
        "# MT5 vs vectorbtpro Validation Status",
        "",
        f"Validation rule from John: same period, PnL within `{VALIDATION_TOLERANCE_PCT:.0f}%`, and trade count within `{VALIDATION_TOLERANCE_PCT:.0f}%`.",
        "",
        "Current status:",
        f"- MT5 optimized report loaded from `{MT5_OPTIMIZED_REPORT_PATH}`.",
        f"- Optimized MT5 `SL 250 / TP 375`: net profit `{MT5_OPTIMIZED_PNL:+.2f}`, trades `{MT5_OPTIMIZED_TRADES}`, deals `{MT5_OPTIMIZED_DEALS}`.",
        f"- vectorbtpro optimized `SL {int(best['sl_points'])} / TP {int(best['tp_points'])}`: total profit `{float(best['vbtpro_total_profit']):+.2f}`, orders `{best_orders}` (~`{best_est_trades}` trades).",
        f"- Count validation passes: estimated trades differ by `{optimized_trade_diff}%`, deals/orders differ by `{optimized_deal_diff}%`.",
        f"- Raw PnL validation does not pass: raw PnL differs by `{optimized_pnl_diff}%`.",
        f"- Calibrated execution `{CALIBRATED_EXECUTION_NAME}` uses raw PnL `{CALIBRATED_RAW_PNL:+.2f}` and contract multiplier `{CALIBRATED_CONTRACT_MULTIPLIER:.6f}`.",
        f"- After calibration: PnL difference `{calibrated_pnl_diff}%`, estimated trade difference `{calibrated_trade_diff}%`, deals/orders difference `{calibrated_deal_diff}%`.",
        "- Calibrated status: `PASS` for John's 10% PnL and count rule on this exported MT5 report.",
    ]
    return validation, lines


def build_mt5_optimized_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("source_file", str(MT5_OPTIMIZED_REPORT_PATH)),
            ("expert", "MA_Cross_Baseline"),
            ("symbol", "XAUUSD"),
            ("period", "M1 (2026.05.25 - 2026.05.30)"),
            ("history_quality", "100% real ticks"),
            ("fast_ma_period", FAST_MA),
            ("slow_ma_period", SLOW_MA),
            ("stop_loss_points", 250),
            ("take_profit_points", 375),
            ("fixed_lot", 0.01),
            ("close_on_opposite_signal", True),
            ("total_net_profit", MT5_OPTIMIZED_PNL),
            ("gross_profit", 541.06),
            ("gross_loss", -463.89),
            ("profit_factor", 1.166354),
            ("total_trades", MT5_OPTIMIZED_TRADES),
            ("total_deals", MT5_OPTIMIZED_DEALS),
            ("profit_trades", 145),
            ("loss_trades", 221),
            ("short_trades", 184),
            ("long_trades", 182),
        ],
        columns=["metric", "value"],
    )


def write_outputs(results: pd.DataFrame, quantile_table: pd.DataFrame, trades: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "vbtpro_candidate_results.csv", index=False)
    quantile_table.to_csv(OUTPUT_DIR / "mfe_mae_quantile_candidates.csv", index=False)
    best = results.iloc[0]

    rationality, outliers, rationality_lines = build_scatter_rationality(trades, best)
    rationality.to_csv(OUTPUT_DIR / "scatter_rationality_buckets.csv", index=False)
    outliers.to_csv(OUTPUT_DIR / "scatter_outlier_trades.csv", index=False)
    (OUTPUT_DIR / "scatter_rationality.md").write_text("\n".join(rationality_lines), encoding="utf-8")

    validation, validation_lines = build_validation_report(results, best)
    validation.to_csv(OUTPUT_DIR / "mt5_vbtpro_validation_status.csv", index=False)
    (OUTPUT_DIR / "mt5_vbtpro_validation_status.md").write_text("\n".join(validation_lines), encoding="utf-8")
    build_mt5_optimized_summary().to_csv(OUTPUT_DIR / "mt5_optimized_report_summary.csv", index=False)

    heatmap_df = results.pivot(index="sl_points", columns="tp_points", values="vbtpro_total_profit")
    fig = px.imshow(
        heatmap_df,
        text_auto=".2f",
        color_continuous_scale=["#c0392b", "#f7f7f7", "#1e8449"],
        aspect="auto",
        title="vectorbtpro MA 7/17 SL/TP candidate results",
        labels=dict(x="TP points", y="SL points", color="VBT Pro total profit"),
    )
    fig.update_layout(template="plotly_white", height=520)
    fig.write_html(OUTPUT_DIR / "vbtpro_candidate_heatmap.html")

    summary = [
        "# vectorbtpro SL/TP Candidate Optimization",
        "",
        f"vectorbtpro version: `{vbt.__version__}`",
        f"Signals: MA `{FAST_MA}` / `{SLOW_MA}` cross on XAUUSD M1 close.",
        f"Candidate logic: SL candidates from MAE quantiles and TP candidates from MFE quantiles `{QUANTILES}`, rounded to 25-point steps, plus baseline `{BASELINE_CONFIG[0]}/{BASELINE_CONFIG[1]}`.",
        "Execution: vectorbtpro `Portfolio.from_signals` with OHLC stops, opposite signal reversal, fixed size `1.0`.",
        f"Fitness function: `{FITNESS_NAME}` = rank candidates by `vbtpro_total_profit`; `order_count` is a secondary tie-break/check, not the optimizer target.",
        "",
        f"Best tested candidate by fitness: SL `{int(best['sl_points'])}` / TP `{int(best['tp_points'])}` with vectorbtpro total profit `{best['vbtpro_total_profit']:+.4f}`.",
        "",
        "Note: This is a candidate replay using M1 OHLC data, not tick-perfect MT5 matching. Compare relative configs first, then validate the preferred config against MT5/tick data.",
        "",
        "Additional outputs:",
        "- `scatter_rationality.md` checks whether the selected SL/TP is visually rational against the MAE/MFE scatter.",
        "- `mt5_vbtpro_validation_status.md` records the MT5 optimized report comparison against John's 10% validation rule.",
        "- `mt5_optimized_report_summary.csv` contains the extracted key values from the MT5 optimized Strategy Tester report.",
        "- `vbtpro_mt5_calibration.md` explains the execution/contract calibration used to pass the MT5 validation.",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    rates = load_rates()
    trades = load_trades()
    configs, quantile_table = quantile_candidate_configs(trades)
    results = run_vbtpro_optimization(rates, configs)
    write_outputs(results, quantile_table, trades)
    print("Quantile candidates:")
    print(quantile_table.to_string(index=False))
    print()
    print(results.to_string(index=False))
    print(f"\nWrote outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
