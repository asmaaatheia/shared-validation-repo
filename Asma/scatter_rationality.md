# Scatter Rationality Check

Selected vectorbtpro candidate: `SL 250` / `TP 375`.
SL `250` is above `61.88%` of observed MAE values, so about `38.12%` of historical trades moved further against entry than this SL.
TP `375` is above `86.46%` of observed MFE values, so about `13.54%` of historical trades reached at least this favorable excursion.

Manual scatter interpretation:
- If many trades sit above TP while staying left of SL, the TP is visually rational.
- If many trades sit right of SL before reaching TP, the SL may be too tight.
- Trades that hit both levels are exactly the outliers to click and inspect on the price chart because first-hit timing controls the outcome.

Bucket summary is written to `scatter_rationality_buckets.csv`; largest outliers are written to `scatter_outlier_trades.csv`.