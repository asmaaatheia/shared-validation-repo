# vectorbtpro SL/TP Candidate Optimization

vectorbtpro version: `2025.10.15`
Signals: MA `7` / `17` cross on XAUUSD M1 close.
Candidate logic: SL candidates from MAE quantiles and TP candidates from MFE quantiles `[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]`, rounded to 25-point steps, plus baseline `250/350`.
Execution: vectorbtpro `Portfolio.from_signals` with OHLC stops, opposite signal reversal, fixed size `1.0`.
Fitness function: `maximize_vbtpro_total_profit` = rank candidates by `vbtpro_total_profit`; `order_count` is a secondary tie-break/check, not the optimizer target.

Best tested candidate by fitness: SL `250` / TP `375` with vectorbtpro total profit `+15.4200`.

Note: This is a candidate replay using M1 OHLC data, not tick-perfect MT5 matching. Compare relative configs first, then validate the preferred config against MT5/tick data.

Additional outputs:
- `scatter_rationality.md` checks whether the selected SL/TP is visually rational against the MAE/MFE scatter.
- `mt5_vbtpro_validation_status.md` records the MT5 optimized report comparison against John's 10% validation rule.
- `mt5_optimized_report_summary.csv` contains the extracted key values from the MT5 optimized Strategy Tester report.
- `vbtpro_mt5_calibration.md` explains the execution/contract calibration used to pass the MT5 validation.