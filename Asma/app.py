from __future__ import annotations

from pathlib import Path

import dash_ag_grid as dag
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from plotly.subplots import make_subplots


APP_DIR = Path(__file__).resolve().parent
MT5_DATA_PATH = APP_DIR / "data" / "mt5_news_events.csv"
FALLBACK_DATA_PATH = APP_DIR / "data" / "news_events.csv"
DATA_PATH = MT5_DATA_PATH if MT5_DATA_PATH.exists() else FALLBACK_DATA_PATH
DATA_SOURCE_NAME = "MT5 Economic Calendar" if DATA_PATH == MT5_DATA_PATH else "temporary non-MT5 fallback"
VALIDATION_DIR = Path(
    r"C:\Users\HP\Documents\Codex\2026-05-31\i-did-validation-using-vectorbt-pro\work\backtest_week_2026-05-25_29"
)
MFE_MAE_PATH = VALIDATION_DIR / "mfe_mae_report.csv"
RATES_PATH = VALIDATION_DIR / "mt5_rates_m1.csv"

IMPACT_ORDER = ["High", "Moderate", "Low", "None"]
IMPACT_COLOR = {
    "High": "#c0392b",
    "Moderate": "#d68910",
    "Low": "#2874a6",
    "None": "#7f8c8d",
}
IMPACT_SCORE = {"High": 3, "Moderate": 2, "Low": 1, "None": 0}
FAST_MA_PERIOD = 7
SLOW_MA_PERIOD = 17
BASE_STOP_LOSS_POINTS = 250
BASE_TAKE_PROFIT_POINTS = 350


def parse_calendar_number(value: str) -> float:
    text = str(value).strip()
    if not text or text.lower() == "nan" or "|" in text:
        return float("nan")
    text = text.replace("%", "").replace(",", "")
    suffix = text[-1:].upper()
    if suffix in {"K", "M", "B", "T"}:
        text = text[:-1]
    return pd.to_numeric(text, errors="coerce")


def parse_news_csv(path: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        header = handle.readline().strip().lstrip("\ufeff")
        if header != "Time,Event,Currency,Impact,Forecast,Result":
            raise ValueError(f"Unexpected news CSV header: {header!r}")

        for line_no, raw_line in enumerate(handle, start=2):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) < 6:
                continue

            records.append(
                {
                    "source_line": line_no,
                    "time_raw": parts[0],
                    "event": ", ".join(parts[1:-4]),
                    "currency": parts[-4],
                    "impact": parts[-3] if parts[-3] in IMPACT_SCORE else "None",
                    "forecast_raw": parts[-2],
                    "result_raw": parts[-1],
                }
            )

    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time_raw"], format="%Y.%m.%d %H:%M", errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    df["forecast_display"] = df["forecast_raw"].replace({"nan": "", "None": ""}).fillna("")
    df["result_display"] = df["result_raw"].replace({"nan": "", "None": ""}).fillna("")
    df["forecast"] = df["forecast_display"].apply(parse_calendar_number)
    df["result"] = df["result_display"].apply(parse_calendar_number)
    df["surprise"] = df["result"] - df["forecast"]
    df["surprise_abs"] = df["surprise"].abs()
    df["impact_score"] = df["impact"].map(IMPACT_SCORE).fillna(0).astype(int)
    df["date"] = df["time"].dt.date.astype(str)
    df["hour"] = df["time"].dt.hour
    df["display_time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M")
    df["event_id"] = df.index.astype(int)
    return df


def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    trades = pd.read_csv(
        path,
        parse_dates=[
            "mt5_open_time",
            "mt5_close_time",
            "mt5_mfe_time",
            "mt5_mae_time",
            "vbtpro_open_time",
            "vbtpro_close_time",
        ],
    )
    trades["trade_index"] = trades["trade_index"].astype(int)
    trades["display_open_time"] = trades["mt5_open_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    trades["display_close_time"] = trades["mt5_close_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    trades["duration_min"] = (
        (trades["mt5_close_time"] - trades["mt5_open_time"]).dt.total_seconds() / 60
    ).round(1)
    trades["profit_display"] = trades["mt5_profit"].round(2)
    return trades.sort_values("mt5_open_time").reset_index(drop=True)


def load_rates(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rates = pd.read_csv(path, parse_dates=["time"])
    return rates.sort_values("time").reset_index(drop=True)


NEWS = parse_news_csv(DATA_PATH)
TRADES = load_trades(MFE_MAE_PATH)
RATES = load_rates(RATES_PATH)
INITIAL_FILTERED = None


def mark_news_window(
    trades: pd.DataFrame,
    events: pd.DataFrame,
    before_minutes: int,
    after_minutes: int,
) -> pd.Series:
    if trades.empty or events.empty:
        return pd.Series(False, index=trades.index)

    open_times = trades["mt5_open_time"]
    in_window = pd.Series(False, index=trades.index)
    for event_time in events["time"].dropna().sort_values():
        start = event_time - pd.Timedelta(minutes=before_minutes)
        end = event_time + pd.Timedelta(minutes=after_minutes)
        in_window |= open_times.between(start, end, inclusive="both")
    return in_window


def mark_trade_context_news_window(
    trades: pd.DataFrame,
    events: pd.DataFrame,
    before_entry_minutes: int,
    after_exit_minutes: int,
) -> pd.Series:
    if trades.empty or events.empty:
        return pd.Series(False, index=trades.index)
    event_times = events["time"].dropna().sort_values()
    in_window = pd.Series(False, index=trades.index)
    for idx, trade in trades.iterrows():
        start = trade["mt5_open_time"] - pd.Timedelta(minutes=before_entry_minutes)
        end = trade["mt5_close_time"] + pd.Timedelta(minutes=after_exit_minutes)
        in_window.loc[idx] = event_times.between(start, end, inclusive="both").any()
    return in_window


def summarize_trade_bucket(label: str, trades: pd.DataFrame) -> dict:
    count = int(len(trades))
    pnl = float(trades["mt5_profit"].sum()) if count else 0.0
    wins = int((trades["mt5_profit"] > 0).sum()) if count else 0
    reasons = trades["mt5_reason"].astype(str).str.upper() if count else pd.Series(dtype=str)
    return {
        "bucket": label,
        "trades": count,
        "pnl": round(pnl, 2),
        "avg_pnl": round(pnl / count, 3) if count else 0.0,
        "win_rate": round((wins / count) * 100, 1) if count else 0.0,
        "tp_rate": round((reasons.eq("TP").sum() / count) * 100, 1) if count else 0.0,
        "sl_rate": round((reasons.eq("SL").sum() / count) * 100, 1) if count else 0.0,
        "avg_mfe_pts": round(float(trades["mt5_mfe_points"].mean()), 1) if count else 0.0,
        "avg_mae_pts": round(float(trades["mt5_mae_abs_points"].mean()), 1) if count else 0.0,
    }


def load_strategy_stats(news: pd.DataFrame) -> dict:
    if not MFE_MAE_PATH.exists():
        return {
            "available": False,
            "kpis": {},
            "daily": pd.DataFrame(),
            "windows": pd.DataFrame(),
            "notes": ["Strategy validation file was not found."],
        }

    trades = pd.read_csv(
        MFE_MAE_PATH,
        parse_dates=["mt5_open_time", "mt5_close_time", "mt5_mfe_time", "mt5_mae_time"],
    )
    trades["day"] = trades["mt5_open_time"].dt.date.astype(str)
    trade_count = int(len(trades))
    pnl = float(trades["mt5_profit"].sum())
    win_rate = float((trades["mt5_profit"] > 0).mean() * 100) if trade_count else 0.0

    daily = (
        trades.groupby("day", observed=True)
        .agg(
            trades=("trade_index", "count"),
            pnl=("mt5_profit", "sum"),
            wins=("mt5_profit", lambda values: int((values > 0).sum())),
            avg_mfe_pts=("mt5_mfe_points", "mean"),
            avg_mae_pts=("mt5_mae_abs_points", "mean"),
        )
        .reset_index()
    )
    daily["win_rate"] = (daily["wins"] / daily["trades"] * 100).round(1)
    daily["pnl"] = daily["pnl"].round(2)
    daily["avg_mfe_pts"] = daily["avg_mfe_pts"].round(1)
    daily["avg_mae_pts"] = daily["avg_mae_pts"].round(1)

    news_by_day = news.groupby("date", observed=True).size().rename("news_events").reset_index()
    high_by_day = (
        news[news["impact"].eq("High")]
        .groupby("date", observed=True)
        .size()
        .rename("high_news")
        .reset_index()
    )
    daily = daily.merge(news_by_day, left_on="day", right_on="date", how="left")
    daily = daily.merge(high_by_day, left_on="day", right_on="date", how="left", suffixes=("", "_high"))
    daily["news_events"] = daily["news_events"].fillna(0).astype(int)
    daily["high_news"] = daily["high_news"].fillna(0).astype(int)
    daily = daily.drop(columns=[col for col in ["date", "date_high"] if col in daily.columns])

    usd_news = news[news["currency"].eq("USD")]
    usd_highmod = usd_news[usd_news["impact"].isin(["High", "Moderate"])]
    usd_high = usd_news[usd_news["impact"].eq("High")]

    post_highmod = mark_news_window(trades, usd_highmod, before_minutes=0, after_minutes=60)
    spike_high = mark_news_window(trades, usd_high, before_minutes=15, after_minutes=15)
    windows = pd.DataFrame(
        [
            summarize_trade_bucket("USD High/Moderate 0-60m after", trades[post_highmod]),
            summarize_trade_bucket("Outside that window", trades[~post_highmod]),
            summarize_trade_bucket("USD High +/-15m", trades[spike_high]),
            summarize_trade_bucket("Outside High +/-15m", trades[~spike_high]),
        ]
    )

    post_row = windows[windows["bucket"].eq("USD High/Moderate 0-60m after")].iloc[0]
    outside_post_row = windows[windows["bucket"].eq("Outside that window")].iloc[0]
    spike_row = windows[windows["bucket"].eq("USD High +/-15m")].iloc[0]
    notes = [
        f"MT5 strategy result for the validated week: {trade_count} trades, {pnl:+.2f} PnL, {win_rate:.1f}% win rate.",
        f"After USD High/Moderate news, 0-60m trades made {post_row['pnl']:+.2f} versus {outside_post_row['pnl']:+.2f} outside that window.",
        f"The immediate USD High +/-15m window was {spike_row['pnl']:+.2f}, so the release spike itself was weaker than the post-news window.",
    ]

    return {
        "available": True,
        "kpis": {
            "trades": f"{trade_count:,}",
            "pnl": f"{pnl:+.2f}",
            "win_rate": f"{win_rate:.1f}%",
            "avg_mfe": f"{trades['mt5_mfe_points'].mean():.1f} pts",
            "avg_mae": f"{trades['mt5_mae_abs_points'].mean():.1f} pts",
            "validation": "MT5 +23.44 / VBT +4.17",
        },
        "daily": daily,
        "windows": windows,
        "notes": notes,
    }


STRATEGY_STATS = load_strategy_stats(NEWS)


def filter_news(
    currencies: list[str] | None,
    impacts: list[str] | None,
    start_date: str | None,
    end_date: str | None,
    search: str | None,
) -> pd.DataFrame:
    filtered = NEWS.copy()
    if currencies:
        filtered = filtered[filtered["currency"].isin(currencies)]
    if impacts:
        filtered = filtered[filtered["impact"].isin(impacts)]
    if start_date:
        filtered = filtered[filtered["time"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[filtered["time"] < pd.to_datetime(end_date) + pd.Timedelta(days=1)]
    if search:
        needle = search.strip().lower()
        if needle:
            filtered = filtered[filtered["event"].str.lower().str.contains(needle, regex=False)]
    return filtered.sort_values("time").reset_index(drop=True)


def grid_records(df: pd.DataFrame) -> list[dict]:
    out = df[
        [
            "event_id",
            "display_time",
            "event",
            "currency",
            "impact",
            "forecast_display",
            "result_display",
            "surprise",
        ]
    ].copy()
    out = out.where(pd.notna(out), None)
    return out.to_dict("records")


def trade_grid_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    out = df[
        [
            "trade_index",
            "direction",
            "display_open_time",
            "display_close_time",
            "duration_min",
            "profit_display",
            "mt5_reason",
            "mt5_entry_price",
            "mt5_exit_price",
            "mt5_mfe_points",
            "mt5_mae_abs_points",
        ]
    ].copy()
    out = out.rename(
        columns={
            "display_open_time": "open_time",
            "display_close_time": "close_time",
            "profit_display": "pnl",
            "mt5_reason": "reason",
            "mt5_entry_price": "entry",
            "mt5_exit_price": "exit",
            "mt5_mfe_points": "mfe_pts",
            "mt5_mae_abs_points": "mae_pts",
        }
    )
    out = out.where(pd.notna(out), None)
    return out.to_dict("records")


def selected_trade_from_rows(selected_rows: list[dict] | None) -> pd.Series | None:
    if TRADES.empty:
        return None
    if selected_rows:
        trade_index = int(selected_rows[0]["trade_index"])
        match = TRADES[TRADES["trade_index"].eq(trade_index)]
        if not match.empty:
            return match.iloc[0]
    return TRADES.iloc[0]


def selected_trade_from_index(trade_index: int | str | None) -> pd.Series | None:
    if TRADES.empty:
        return None
    if trade_index is None:
        return TRADES.iloc[0]
    match = TRADES[TRADES["trade_index"].eq(int(trade_index))]
    if match.empty:
        return TRADES.iloc[0]
    return match.iloc[0]


def selected_trade_from_click(click_data: dict | None) -> pd.Series | None:
    if TRADES.empty or not click_data or not click_data.get("points"):
        return None
    point = click_data["points"][0]
    customdata = point.get("customdata")
    if isinstance(customdata, list):
        trade_index = int(customdata[0])
    elif customdata is not None:
        trade_index = int(customdata)
    else:
        return None
    match = TRADES[TRADES["trade_index"].eq(trade_index)]
    if match.empty:
        return None
    return match.iloc[0]


def trade_options() -> list[dict]:
    if TRADES.empty:
        return []
    return [
        {
            "label": f"#{int(row.trade_index)} {row.direction} {row.mt5_open_time:%m-%d %H:%M} PnL {row.mt5_profit:+.2f}",
            "value": int(row.trade_index),
        }
        for row in TRADES.itertuples(index=False)
    ]


def filtered_news_events(
    currencies: list[str] | None,
    impacts: list[str] | None,
) -> pd.DataFrame:
    events = NEWS.copy()
    if currencies:
        events = events[events["currency"].isin(currencies)]
    if impacts:
        events = events[events["impact"].isin(impacts)]
    return events.sort_values("time").reset_index(drop=True)


def nearby_news_records(events: pd.DataFrame) -> list[dict]:
    if events.empty:
        return []
    columns = [
        col
        for col in [
            "display_time",
            "event",
            "currency",
            "impact",
            "forecast_display",
            "result_display",
            "minutes_from_entry",
            "minutes_from_exit",
            "surprise",
        ]
        if col in events.columns
    ]
    out = events[
        [
            *columns,
        ]
    ].copy()
    out = out.where(pd.notna(out), None)
    return out.to_dict("records")


def news_rows_for_trade_window(
    trade: pd.Series | None,
    before_minutes: int | float | None,
    after_minutes: int | float | None,
    currencies: list[str] | None,
    impacts: list[str] | None,
) -> pd.DataFrame:
    if trade is None:
        return pd.DataFrame()
    before = int(before_minutes or 0)
    after = int(after_minutes or 0)
    start = trade["mt5_open_time"] - pd.Timedelta(minutes=before)
    end = trade["mt5_close_time"] + pd.Timedelta(minutes=after)
    events = filtered_news_events(currencies, impacts)
    rows = events[events["time"].between(start, end, inclusive="both")].copy()
    if rows.empty:
        return rows
    rows["minutes_from_entry"] = ((rows["time"] - trade["mt5_open_time"]).dt.total_seconds() / 60).round(1)
    rows["minutes_from_exit"] = ((rows["time"] - trade["mt5_close_time"]).dt.total_seconds() / 60).round(1)
    return rows.sort_values("time").reset_index(drop=True)


def news_window_summary(nearby: pd.DataFrame) -> dict:
    if nearby.empty:
        return {
            "Nearby News": "0",
            "High Impact": "0",
            "Currencies": "0",
            "Nearest Event": "-",
            "Nearest Minutes": "-",
        }
    nearest = nearby.iloc[nearby["minutes_from_entry"].abs().sort_values().index[0]]
    return {
        "Nearby News": f"{len(nearby):,}",
        "High Impact": f"{int(nearby['impact'].eq('High').sum()):,}",
        "Currencies": f"{nearby['currency'].nunique():,}",
        "Nearest Event": str(nearest["event"])[:34] or "-",
        "Nearest Minutes": f"{abs(float(nearest['minutes_from_entry'])):.1f}",
    }


def news_window_kpis(summary: dict) -> html.Div:
    items = [
        ("Nearby News", summary["Nearby News"]),
        ("High Impact", summary["High Impact"]),
        ("Currencies", summary["Currencies"]),
        ("Nearest Event", summary["Nearest Event"]),
        ("Nearest Minutes", summary["Nearest Minutes"]),
    ]
    return html.Div(
        className="kpi-strip news-window-kpis",
        children=[
            html.Div([html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value")], className="kpi-card")
            for label, value in items
        ],
    )


def selected_trade_summary(trade: pd.Series | None) -> html.Div:
    if trade is None:
        return html.Div("No trade selected.", className="selected-card")
    return html.Div(
        className="selected-card trade-summary-card",
        children=[
            html.Div(
                f"Trade {int(trade['trade_index'])} | {trade['direction']} | {trade['mt5_reason']}",
                className=f"impact-pill {'trade-win' if trade['mt5_profit'] > 0 else 'trade-loss'}",
            ),
            html.H2(f"PnL {trade['mt5_profit']:+.2f}"),
            html.Div(
                [
                    html.Span(f"Entry {trade['display_open_time']}"),
                    html.Span(f"Exit {trade['display_close_time']}"),
                    html.Span(f"Duration {trade['duration_min']:.1f} min"),
                    html.Span(f"MFE {trade['mt5_mfe_points']:.0f} pts"),
                    html.Span(f"MAE {trade['mt5_mae_abs_points']:.0f} pts"),
                ],
                className="selected-meta",
            ),
        ],
    )


def trade_summary(label: str, trades: pd.DataFrame) -> dict:
    count = int(len(trades))
    pnl = float(trades["mt5_profit"].sum()) if count else 0.0
    wins = int((trades["mt5_profit"] > 0).sum()) if count else 0
    return {
        "set": label,
        "trades": count,
        "pnl": round(pnl, 2),
        "avg_pnl": round(pnl / count, 3) if count else 0.0,
        "win_rate": round((wins / count) * 100, 1) if count else 0.0,
        "avg_mfe_pts": round(float(trades["mt5_mfe_points"].mean()), 1) if count else 0.0,
        "avg_mae_pts": round(float(trades["mt5_mae_abs_points"].mean()), 1) if count else 0.0,
    }


def filter_simulation_stats(
    currencies: list[str] | None,
    impacts: list[str] | None,
    before_minutes: int | float | None,
    after_minutes: int | float | None,
) -> pd.DataFrame:
    if TRADES.empty:
        return pd.DataFrame()
    before = int(before_minutes or 0)
    after = int(after_minutes or 0)
    events = filtered_news_events(currencies, impacts)
    blocked_mask = mark_trade_context_news_window(TRADES, events, before, after)
    kept = TRADES[~blocked_mask]
    removed = TRADES[blocked_mask]
    return pd.DataFrame(
        [
            trade_summary("All trades", TRADES),
            trade_summary("Kept after news filter", kept),
            trade_summary("Removed by news filter", removed),
        ]
    )


def filter_simulation_stats_for_trade(
    selected_trade_index: int | str | None,
    currencies: list[str] | None,
    impacts: list[str] | None,
    before_minutes: int | float | None,
    after_minutes: int | float | None,
) -> pd.DataFrame:
    if TRADES.empty:
        return pd.DataFrame()
    before = int(before_minutes or 0)
    after = int(after_minutes or 0)
    trade = selected_trade_from_index(selected_trade_index)
    events = news_rows_for_trade_window(trade, before, after, currencies, impacts)
    blocked_mask = mark_trade_context_news_window(TRADES, events, before, after)
    kept = TRADES[~blocked_mask]
    removed = TRADES[blocked_mask]
    return pd.DataFrame(
        [
            trade_summary("All trades", TRADES),
            trade_summary("Kept after selected news", kept),
            trade_summary("Removed by selected news", removed),
        ]
    )


def empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=40, r=20, t=52, b=36),
        height=320,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def selected_trade_card(trade: pd.Series | None) -> html.Div:
    if trade is None:
        return html.Div("No trade data found.", className="selected-card")

    return html.Div(
        className="selected-card trade-card",
        children=[
            html.Div(
                f"Trade {int(trade['trade_index'])} | {trade['direction']} | {trade['mt5_reason']}",
                className=f"impact-pill {'trade-win' if trade['mt5_profit'] > 0 else 'trade-loss'}",
            ),
            html.H2(f"MT5 PnL {trade['mt5_profit']:+.2f}"),
            html.Div(
                [
                    html.Span(f"Open {trade['display_open_time']}"),
                    html.Span(f"Close {trade['display_close_time']}"),
                    html.Span(f"Entry {trade['mt5_entry_price']:.2f}"),
                    html.Span(f"Exit {trade['mt5_exit_price']:.2f}"),
                    html.Span(f"MFE {trade['mt5_mfe_points']:.0f} pts"),
                    html.Span(f"MAE {trade['mt5_mae_abs_points']:.0f} pts"),
                ],
                className="selected-meta",
            ),
        ],
    )


def trade_overview_figure(selected_trade_index: int | None = None) -> go.Figure:
    if TRADES.empty:
        return empty_figure("Trade overview")

    colors = ["#1e8449" if value > 0 else "#c0392b" for value in TRADES["mt5_profit"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=TRADES["mt5_open_time"],
            y=TRADES["mt5_profit"],
            mode="markers",
            customdata=TRADES[["trade_index"]],
            marker=dict(
                size=9,
                color=colors,
                line=dict(width=1, color="#ffffff"),
            ),
            text=[
                f"Trade {int(row.trade_index)} | {row.direction} | {row.mt5_reason}"
                for row in TRADES.itertuples(index=False)
            ],
            hovertemplate="%{text}<br>Open %{x}<br>PnL %{y:.2f}<extra></extra>",
            name="Trades",
        )
    )
    if selected_trade_index is not None:
        selected = TRADES[TRADES["trade_index"].eq(selected_trade_index)]
        if not selected.empty:
            row = selected.iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[row["mt5_open_time"]],
                    y=[row["mt5_profit"]],
                    mode="markers",
                    marker=dict(size=18, color="#17202a", symbol="circle-open", line=dict(width=3)),
                    name="Selected",
                    hoverinfo="skip",
                )
            )
    fig.add_hline(y=0, line_width=1, line_color="#95a5a6")
    fig.update_layout(
        title="Click a trade point to plot price and nearby news",
        template="plotly_white",
        height=300,
        margin=dict(l=44, r=18, t=52, b=38),
        showlegend=False,
        clickmode="event+select",
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Trade PnL")
    return fig


def mfe_mae_scatter_figure(selected_trade_index: int | str | None = None) -> go.Figure:
    if TRADES.empty:
        return empty_figure("MFE vs MAE scatter")

    plot_df = TRADES.copy()
    plot_df["outcome"] = plot_df["mt5_profit"].apply(lambda value: "Win" if value > 0 else "Loss")
    plot_df["size"] = plot_df["mt5_profit"].abs().clip(lower=1.0)
    fig = px.scatter(
        plot_df,
        x="mt5_mae_abs_points",
        y="mt5_mfe_points",
        color="outcome",
        symbol="direction",
        size="size",
        size_max=11,
        color_discrete_map={"Win": "#1e8449", "Loss": "#c0392b"},
        custom_data=["trade_index", "direction", "mt5_profit", "mt5_reason"],
        labels={
            "mt5_mae_abs_points": "MAE points",
            "mt5_mfe_points": "MFE points",
            "outcome": "Outcome",
        },
        title="MAE/MFE scatter: click a trade to plot it",
    )
    fig.update_traces(
        marker=dict(line=dict(color="#ffffff", width=0.8), opacity=0.85),
        hovertemplate=(
            "Trade %{customdata[0]} | %{customdata[1]} | %{customdata[3]}<br>"
            "MAE %{x:.0f} pts<br>"
            "MFE %{y:.0f} pts<br>"
            "PnL %{customdata[2]:+.2f}<extra></extra>"
        ),
    )
    if selected_trade_index is not None:
        selected = TRADES[TRADES["trade_index"].eq(int(selected_trade_index))]
        if not selected.empty:
            row = selected.iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[row["mt5_mae_abs_points"]],
                    y=[row["mt5_mfe_points"]],
                    mode="markers",
                    marker=dict(size=18, color="#17202a", symbol="circle-open", line=dict(width=3)),
                    name="Selected trade",
                    hovertemplate=f"Selected trade {int(row['trade_index'])}<extra></extra>",
                )
            )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=52, r=24, t=58, b=46),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        clickmode="event+select",
    )
    fig.update_xaxes(rangemode="tozero", showspikes=True)
    fig.update_yaxes(rangemode="tozero", showspikes=True)
    return fig


def trade_extreme_price(trade: pd.Series, kind: str) -> float:
    point_size = 0.01
    direction = str(trade["direction"]).upper()
    entry = float(trade["mt5_entry_price"])
    if kind == "mfe":
        move = float(trade["mt5_mfe_points"]) * point_size
        return entry + move if direction == "BUY" else entry - move
    move = float(trade["mt5_mae_abs_points"]) * point_size
    return entry - move if direction == "BUY" else entry + move


def trade_chart_figure(
    trade: pd.Series | None,
    window_minutes: int | float | None,
    currencies: list[str] | None,
    impacts: list[str] | None,
) -> go.Figure:
    if trade is None or RATES.empty:
        return empty_figure("Click a trade to chart it with nearby news")

    chart_after = int(window_minutes or 75)
    start = trade["mt5_open_time"] - pd.Timedelta(minutes=20)
    end = trade["mt5_close_time"] + pd.Timedelta(minutes=chart_after)
    rates = RATES[RATES["time"].between(start, end, inclusive="both")]
    if rates.empty:
        return empty_figure("No price data for selected trade window")
    rates = rates.copy()
    rates["ma_fast"] = rates["close"].rolling(FAST_MA_PERIOD, min_periods=FAST_MA_PERIOD).mean()
    rates["ma_slow"] = rates["close"].rolling(SLOW_MA_PERIOD, min_periods=SLOW_MA_PERIOD).mean()
    previous_fast = rates["ma_fast"].shift(1)
    previous_slow = rates["ma_slow"].shift(1)
    rates["bull_cross"] = (previous_fast <= previous_slow) & (rates["ma_fast"] > rates["ma_slow"])
    rates["bear_cross"] = (previous_fast >= previous_slow) & (rates["ma_fast"] < rates["ma_slow"])

    events = filtered_news_events(currencies, impacts)
    nearby = events[events["time"].between(start, end, inclusive="both")]

    fig = go.Figure()
    candle_up = "#16a085"
    candle_down = "#c0392b"
    fig.add_trace(
        go.Candlestick(
            x=rates["time"],
            open=rates["open"],
            high=rates["high"],
            low=rates["low"],
            close=rates["close"],
            name="XAUUSD M1",
            increasing_line_color=candle_up,
            decreasing_line_color=candle_down,
            increasing_fillcolor=candle_up,
            decreasing_fillcolor=candle_down,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rates["time"],
            y=rates["ma_fast"],
            mode="lines",
            line=dict(color="#2874a6", width=1.6),
            name=f"MA {FAST_MA_PERIOD}",
            hovertemplate=f"MA {FAST_MA_PERIOD}<br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rates["time"],
            y=rates["ma_slow"],
            mode="lines",
            line=dict(color="#d68910", width=1.6),
            name=f"MA {SLOW_MA_PERIOD}",
            hovertemplate=f"MA {SLOW_MA_PERIOD}<br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
        )
    )
    bull_cross = rates[rates["bull_cross"]]
    bear_cross = rates[rates["bear_cross"]]
    if not bull_cross.empty:
        fig.add_trace(
            go.Scatter(
                x=bull_cross["time"],
                y=bull_cross["close"],
                mode="markers",
                marker=dict(size=11, color="#1e8449", symbol="triangle-up", line=dict(color="#ffffff", width=1)),
                name="Bull cross",
                hovertemplate="Bull MA cross<br>%{x}<br>%{y:.2f}<extra></extra>",
            )
        )
    if not bear_cross.empty:
        fig.add_trace(
            go.Scatter(
                x=bear_cross["time"],
                y=bear_cross["close"],
                mode="markers",
                marker=dict(size=11, color="#c0392b", symbol="triangle-down", line=dict(color="#ffffff", width=1)),
                name="Bear cross",
                hovertemplate="Bear MA cross<br>%{x}<br>%{y:.2f}<extra></extra>",
            )
        )
    nearest_cross_time = None
    cross_mask = rates["bull_cross"] if str(trade["direction"]).upper() == "BUY" else rates["bear_cross"]
    matching_crosses = rates.loc[cross_mask, ["time", "close"]]
    if not matching_crosses.empty:
        nearest_idx = (matching_crosses["time"] - trade["mt5_open_time"]).abs().idxmin()
        nearest_cross_time = matching_crosses.loc[nearest_idx, "time"]
    trade_color = "#1e8449" if float(trade["mt5_profit"]) > 0 else "#c0392b"
    mfe_price = trade_extreme_price(trade, "mfe")
    mae_price = trade_extreme_price(trade, "mae")
    fig.add_trace(
        go.Scatter(
            x=[trade["mt5_open_time"], trade["mt5_close_time"]],
            y=[trade["mt5_entry_price"], trade["mt5_exit_price"]],
            mode="lines",
            line=dict(color=trade_color, width=3),
            name=f"{trade['direction']} trade",
            hovertemplate="Trade path<br>%{x}<br>%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[trade["mt5_open_time"]],
            y=[trade["mt5_entry_price"]],
            mode="markers",
            marker=dict(size=14, color="#17202a", symbol="triangle-up" if trade["direction"] == "BUY" else "triangle-down"),
            name=f"{trade['direction']} entry",
            hovertemplate=f"{trade['direction']} entry<br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[trade["mt5_close_time"]],
            y=[trade["mt5_exit_price"]],
            mode="markers",
            marker=dict(size=14, color="#6c3483", symbol="x"),
            name=f"Exit {trade['mt5_reason']}",
            hovertemplate=f"Exit {trade['mt5_reason']}<br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[trade["mt5_mfe_time"]],
            y=[mfe_price],
            mode="markers",
            marker=dict(size=10, color="#1e8449", symbol="circle"),
            name="MFE",
            hovertemplate="MFE<br>%{x}<br>%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[trade["mt5_mae_time"]],
            y=[mae_price],
            mode="markers",
            marker=dict(size=10, color="#c0392b", symbol="circle"),
            name="MAE",
            hovertemplate="MAE<br>%{x}<br>%{y:.2f}<extra></extra>",
        )
    )
    fig.add_vrect(
        x0=trade["mt5_open_time"],
        x1=trade["mt5_close_time"],
        fillcolor="#d6eaf8",
        opacity=0.25,
        line_width=0,
    )
    if nearest_cross_time is not None:
        fig.add_vline(
            x=nearest_cross_time,
            line_width=1,
            line_dash="dot",
            line_color="#34495e",
        )
    fig.add_hline(
        y=float(trade["mt5_entry_price"]),
        line_width=1,
        line_dash="dot",
        line_color="#17202a",
    )
    fig.add_hline(
        y=float(trade["mt5_exit_price"]),
        line_width=1,
        line_dash="dot",
        line_color=trade_color,
    )

    for _, event in nearby.iterrows():
        color = IMPACT_COLOR.get(event["impact"], "#7f8c8d")
        fig.add_vline(
            x=event["time"],
            line_width=2 if event["impact"] in {"High", "Moderate"} else 1,
            line_dash="dash",
            line_color=color,
        )

    fig.update_layout(
        title=dict(
            text=f"Trade {int(trade['trade_index'])}: XAUUSD M1 with MA {FAST_MA_PERIOD}/{SLOW_MA_PERIOD}",
            font=dict(size=15),
            y=0.98,
            x=0.02,
            xanchor="left",
        ),
        template="plotly_white",
        height=460,
        margin=dict(l=52, r=32, t=54, b=96),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            font=dict(size=10),
            itemwidth=34,
        ),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(title=None)
    price_values = pd.concat(
        [
            rates[["low", "high"]].stack(),
            pd.Series([trade["mt5_entry_price"], trade["mt5_exit_price"], mfe_price, mae_price]),
        ],
        ignore_index=True,
    ).dropna()
    y_min = float(price_values.min())
    y_max = float(price_values.max())
    padding = max((y_max - y_min) * 0.28, 1.5)
    fig.update_yaxes(title="XAUUSD", range=[y_min - padding, y_max + padding], tickformat=".2f")
    return fig


def news_filter_figure(stats: pd.DataFrame) -> go.Figure:
    if stats.empty:
        return empty_figure("News filter effect on strategy stats")
    fig = px.bar(
        stats,
        x="set",
        y="pnl",
        color="avg_pnl",
        color_continuous_scale=["#c0392b", "#f7f7f7", "#1e8449"],
        title="What happens if trades near selected news are filtered?",
        hover_data=["trades", "avg_pnl", "win_rate", "avg_mfe_pts", "avg_mae_pts"],
    )
    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=44, r=20, t=52, b=50),
        coloraxis_colorbar=dict(title="Avg PnL"),
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="MT5 PnL", zeroline=True, zerolinecolor="#95a5a6")
    return fig


def timeline_figure(df: pd.DataFrame, selected_event_id: int | None) -> go.Figure:
    if df.empty:
        return empty_figure("Events by day")

    daily = (
        df.groupby([pd.Grouper(key="time", freq="D"), "impact"], observed=True)
        .size()
        .reset_index(name="events")
    )
    fig = px.bar(
        daily,
        x="time",
        y="events",
        color="impact",
        category_orders={"impact": IMPACT_ORDER},
        color_discrete_map=IMPACT_COLOR,
        title="Events by day",
    )
    if selected_event_id is not None and selected_event_id in set(df["event_id"]):
        selected = df.loc[df["event_id"].eq(selected_event_id)].iloc[0]
        fig.add_vline(
            x=selected["time"],
            line_width=2,
            line_color="#17202a",
            annotation_text=selected["currency"],
            annotation_position="top",
        )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=18, t=52, b=36),
        height=330,
        legend_title_text="Impact",
        bargap=0.08,
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="News count", rangemode="tozero")
    return fig


def surprise_figure(df: pd.DataFrame, selected_event_id: int | None) -> go.Figure:
    scored = df.dropna(subset=["surprise"]).copy()
    if scored.empty:
        return empty_figure("Forecast vs result surprise")

    scored["selected"] = scored["event_id"].eq(selected_event_id)
    fig = px.scatter(
        scored,
        x="time",
        y="surprise",
        color="impact",
        size="impact_score",
        hover_data=["event", "currency", "forecast_display", "result_display"],
        category_orders={"impact": IMPACT_ORDER},
        color_discrete_map=IMPACT_COLOR,
        title="Forecast vs result surprise",
    )
    selected = scored[scored["selected"]]
    if not selected.empty:
        fig.add_trace(
            go.Scatter(
                x=selected["time"],
                y=selected["surprise"],
                mode="markers",
                marker=dict(size=18, color="#17202a", symbol="circle-open", line=dict(width=3)),
                name="Selected",
                hoverinfo="skip",
            )
        )
    fig.add_hline(y=0, line_width=1, line_color="#b2babb")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=44, r=18, t=52, b=36),
        height=330,
        legend_title_text="Impact",
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Result - forecast")
    return fig


def heatmap_figure(df: pd.DataFrame, selected_event_id: int | None) -> go.Figure:
    if df.empty:
        return empty_figure("Event heatmap by currency and hour")

    selected = None
    if selected_event_id is not None and selected_event_id in set(df["event_id"]):
        selected = df.loc[df["event_id"].eq(selected_event_id)].iloc[0]

    top_currencies = df["currency"].value_counts().head(14).index.tolist()
    if selected is not None and selected["currency"] not in top_currencies:
        top_currencies = top_currencies[:-1] + [selected["currency"]]

    heat_df = df[df["currency"].isin(top_currencies)].copy()
    heat = (
        heat_df.groupby(["currency", "hour"], observed=True)
        .size()
        .reset_index(name="events")
        .pivot(index="currency", columns="hour", values="events")
        .reindex(index=top_currencies, columns=list(range(24)))
        .fillna(0)
    )

    fig = px.imshow(
        heat,
        x=heat.columns,
        y=heat.index,
        color_continuous_scale=[
            [0.0, "#f7fbfc"],
            [0.35, "#b9d9e8"],
            [0.70, "#4f93b5"],
            [1.0, "#1f4e5f"],
        ],
        aspect="auto",
        title="Event heatmap by currency and hour",
        labels=dict(x="Hour", y="Currency", color="Events"),
    )

    if selected is not None and selected["currency"] in top_currencies:
        fig.add_trace(
            go.Scatter(
                x=[int(selected["hour"])],
                y=[selected["currency"]],
                mode="markers",
                marker=dict(
                    symbol="circle-open",
                    size=18,
                    color="#17202a",
                    line=dict(width=3),
                ),
                name="Selected",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=54, r=18, t=52, b=42),
        height=330,
        coloraxis_colorbar=dict(title="Events"),
    )
    fig.update_xaxes(dtick=2)
    return fig


def selected_card(selected_rows: list[dict] | None) -> html.Div:
    if not selected_rows:
        row = NEWS.iloc[0]
    else:
        event_id = int(selected_rows[0]["event_id"])
        row = NEWS.loc[NEWS["event_id"].eq(event_id)].iloc[0]

    forecast = row["forecast_display"] or "n/a"
    result = row["result_display"] or "n/a"
    surprise = "n/a" if pd.isna(row["surprise"]) else f"{row['surprise']:+,.3f}"
    return html.Div(
        className="selected-card",
        children=[
            html.Div(row["impact"], className=f"impact-pill impact-{row['impact'].lower()}"),
            html.H2(row["event"]),
            html.Div(
                [
                    html.Span(row["display_time"]),
                    html.Span(row["currency"]),
                    html.Span(f"Forecast {forecast}"),
                    html.Span(f"Result {result}"),
                    html.Span(f"Surprise {surprise}"),
                ],
                className="selected-meta",
            ),
        ],
    )


def kpi_strip(stats: dict) -> html.Div:
    if not stats["available"]:
        return html.Div(stats["notes"][0], className="stats-warning")

    kpis = [
        ("MT5 trades", stats["kpis"]["trades"]),
        ("MT5 PnL", stats["kpis"]["pnl"]),
        ("Win rate", stats["kpis"]["win_rate"]),
        ("Avg MFE", stats["kpis"]["avg_mfe"]),
        ("Avg MAE", stats["kpis"]["avg_mae"]),
        ("Validation", stats["kpis"]["validation"]),
    ]
    return html.Div(
        className="kpi-strip",
        children=[
            html.Div([html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value")], className="kpi-card")
            for label, value in kpis
        ],
    )


def daily_strategy_figure(stats: dict) -> go.Figure:
    daily = stats["daily"]
    if daily.empty:
        return empty_figure("Daily strategy PnL vs MT5 news count")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=daily["day"],
            y=daily["pnl"],
            name="MT5 PnL",
            marker_color=["#1e8449" if value >= 0 else "#c0392b" for value in daily["pnl"]],
            hovertemplate="%{x}<br>PnL: %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=daily["day"],
            y=daily["news_events"],
            mode="lines+markers",
            name="MT5 news events",
            line=dict(color="#2874a6", width=3),
            marker=dict(size=8),
            hovertemplate="%{x}<br>News: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=daily["day"],
            y=daily["high_news"],
            mode="markers",
            name="High news",
            marker=dict(color="#17202a", size=10, symbol="diamond"),
            hovertemplate="%{x}<br>High news: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Daily strategy PnL vs MT5 news count",
        template="plotly_white",
        height=330,
        margin=dict(l=44, r=54, t=52, b=42),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="MT5 PnL", secondary_y=False, zeroline=True, zerolinecolor="#95a5a6")
    fig.update_yaxes(title_text="News count", secondary_y=True, rangemode="tozero")
    fig.update_xaxes(title=None)
    return fig


def window_edge_figure(stats: dict) -> go.Figure:
    windows = stats["windows"]
    if windows.empty:
        return empty_figure("Strategy PnL by news window")

    fig = px.bar(
        windows,
        x="bucket",
        y="pnl",
        color="avg_pnl",
        color_continuous_scale=["#c0392b", "#f7f7f7", "#1e8449"],
        title="Strategy PnL by news window",
        hover_data=["trades", "avg_pnl", "win_rate", "avg_mfe_pts", "avg_mae_pts"],
    )
    fig.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=44, r=20, t=52, b=86),
        coloraxis_colorbar=dict(title="Avg PnL"),
    )
    fig.update_xaxes(title=None, tickangle=-18)
    fig.update_yaxes(title="MT5 PnL", zeroline=True, zerolinecolor="#95a5a6")
    return fig


def trade_news_timeline_figure(
    trade: pd.Series | None,
    nearby: pd.DataFrame,
    before_minutes: int | float | None,
    after_minutes: int | float | None,
) -> go.Figure:
    if trade is None:
        return empty_figure("News window timeline")

    before = int(before_minutes or 0)
    after = int(after_minutes or 0)
    window_start = trade["mt5_open_time"] - pd.Timedelta(minutes=before)
    window_end = trade["mt5_close_time"] + pd.Timedelta(minutes=after)
    fig = go.Figure()
    fig.add_vrect(
        x0=trade["mt5_open_time"],
        x1=trade["mt5_close_time"],
        fillcolor="#d6eaf8",
        opacity=0.35,
        line_width=0,
    )
    fig.add_trace(
        go.Scatter(
            x=[trade["mt5_open_time"], trade["mt5_close_time"]],
            y=[0, 0],
            mode="markers+text",
            marker=dict(size=13, color=["#17202a", "#6c3483"], symbol=["triangle-up", "x"]),
            text=["Entry", "Exit"],
            textposition=["bottom center", "top center"],
            name="Trade",
            hovertemplate="%{text}<br>%{x}<extra></extra>",
        )
    )

    if not nearby.empty:
        y_map = {"High": 4, "Moderate": 3, "Low": 2, "None": 1}
        fig.add_trace(
            go.Scatter(
                x=nearby["time"],
                y=nearby["impact"].map(y_map).fillna(1),
                mode="markers",
                marker=dict(
                    size=nearby["impact"].map({"High": 15, "Moderate": 12, "Low": 9, "None": 8}).fillna(8),
                    color=nearby["impact"].map(IMPACT_COLOR).fillna("#7f8c8d"),
                    line=dict(color="#ffffff", width=1),
                ),
                text=nearby["currency"] + " | " + nearby["event"],
                customdata=nearby[["impact", "minutes_from_entry", "minutes_from_exit"]],
                hovertemplate=(
                    "%{text}<br>"
                    "Impact %{customdata[0]}<br>"
                    "Minutes from entry %{customdata[1]:.1f}<br>"
                    "Minutes from exit %{customdata[2]:.1f}<extra></extra>"
                ),
                name="News",
            )
        )

    start = window_start - pd.Timedelta(minutes=10)
    end = window_end + pd.Timedelta(minutes=10)
    if not nearby.empty:
        start = min(start, nearby["time"].min() - pd.Timedelta(minutes=20))
        end = max(end, nearby["time"].max() + pd.Timedelta(minutes=20))

    fig.update_layout(
        title="News window timeline around selected trade",
        template="plotly_white",
        height=280,
        margin=dict(l=44, r=20, t=52, b=42),
        showlegend=False,
    )
    fig.update_xaxes(title=None, range=[start, end])
    fig.update_yaxes(
        title="Impact",
        tickmode="array",
        tickvals=[0, 1, 2, 3, 4],
        ticktext=["Trade", "None", "Low", "Moderate", "High"],
        range=[-0.5, 4.5],
    )
    return fig


def filter_decision(stats: pd.DataFrame) -> html.Div:
    if stats.empty or len(stats) < 2:
        return html.Div("No filter decision available.", className="decision-card neutral")
    all_row = stats[stats["set"].eq("All trades")].iloc[0]
    kept_rows = stats[stats["set"].astype(str).str.startswith("Kept after")]
    removed_rows = stats[stats["set"].astype(str).str.startswith("Removed by")]
    if kept_rows.empty or removed_rows.empty:
        return html.Div("No filter decision available.", className="decision-card neutral")
    kept_row = kept_rows.iloc[0]
    removed_row = removed_rows.iloc[0]
    if int(removed_row["trades"]) == 0:
        return html.Div(
            className="decision-card neutral",
            children=[
                html.Div("No matching news trades to remove", className="decision-title"),
                html.Div("The selected trade window has no matching MT5 news under the current filters.", className="decision-detail"),
            ],
        )
    pnl_delta = float(kept_row["pnl"]) - float(all_row["pnl"])
    win_delta = float(kept_row["win_rate"]) - float(all_row["win_rate"])
    improves = pnl_delta > 0 and win_delta >= 0
    class_name = "decision-card improve" if improves else "decision-card worsen"
    verdict = "Filtering improves this test" if improves else "Filtering does not improve this test"
    detail = (
        f"Kept-trade PnL changes by {pnl_delta:+.2f} and win rate by {win_delta:+.1f} points. "
        f"The removed bucket had {int(removed_row['trades'])} trades and {float(removed_row['pnl']):+.2f} PnL."
    )
    return html.Div(
        className=class_name,
        children=[html.Div(verdict, className="decision-title"), html.Div(detail, className="decision-detail")],
    )


def equity_curve_figure(
    currencies: list[str] | None,
    impacts: list[str] | None,
    before_minutes: int | float | None,
    after_minutes: int | float | None,
) -> go.Figure:
    if TRADES.empty:
        return empty_figure("Equity curve after news filter")

    before = int(before_minutes or 0)
    after = int(after_minutes or 0)
    events = filtered_news_events(currencies, impacts)
    blocked_mask = mark_trade_context_news_window(TRADES, events, before, after)
    all_trades = TRADES.sort_values("mt5_open_time").copy()
    kept_trades = all_trades.loc[~blocked_mask.loc[all_trades.index]].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=all_trades["mt5_close_time"],
            y=all_trades["mt5_profit"].cumsum(),
            mode="lines",
            line=dict(color="#17202a", width=2),
            name="All trades",
            hovertemplate="%{x}<br>Equity %{y:.2f}<extra></extra>",
        )
    )
    if not kept_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=kept_trades["mt5_close_time"],
                y=kept_trades["mt5_profit"].cumsum(),
                mode="lines",
                line=dict(color="#2874a6", width=2),
                name="After news filter",
                hovertemplate="%{x}<br>Equity %{y:.2f}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_width=1, line_color="#95a5a6")
    fig.update_layout(
        title="Equity curve: all trades vs after news filter",
        template="plotly_white",
        height=330,
        margin=dict(l=44, r=20, t=52, b=42),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Cumulative MT5 PnL")
    return fig


def equity_curve_figure_for_trade(
    selected_trade_index: int | str | None,
    currencies: list[str] | None,
    impacts: list[str] | None,
    before_minutes: int | float | None,
    after_minutes: int | float | None,
) -> go.Figure:
    if TRADES.empty:
        return empty_figure("Equity curve after selected news filter")

    before = int(before_minutes or 0)
    after = int(after_minutes or 0)
    trade = selected_trade_from_index(selected_trade_index)
    events = news_rows_for_trade_window(trade, before, after, currencies, impacts)
    blocked_mask = mark_trade_context_news_window(TRADES, events, before, after)
    all_trades = TRADES.sort_values("mt5_open_time").copy()
    kept_trades = all_trades.loc[~blocked_mask.loc[all_trades.index]].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=all_trades["mt5_close_time"],
            y=all_trades["mt5_profit"].cumsum(),
            mode="lines",
            line=dict(color="#17202a", width=2),
            name="All trades",
            hovertemplate="%{x}<br>Equity %{y:.2f}<extra></extra>",
        )
    )
    if not kept_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=kept_trades["mt5_close_time"],
                y=kept_trades["mt5_profit"].cumsum(),
                mode="lines",
                line=dict(color="#2874a6", width=2),
                name="After selected news filter",
                hovertemplate="%{x}<br>Equity %{y:.2f}<extra></extra>",
            )
        )
    if events.empty:
        title = "Equity curve: selected trade has no news to filter"
    else:
        title = f"Equity curve: removing trades around {len(events)} selected news event(s)"
    fig.add_hline(y=0, line_width=1, line_color="#95a5a6")
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=330,
        margin=dict(l=44, r=20, t=52, b=42),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Cumulative MT5 PnL")
    return fig


def stats_notes(stats: dict) -> html.Div:
    return html.Div(
        className="stats-notes",
        children=[
            html.Div("Combined interpretation", className="stats-notes-title"),
            html.Ul([html.Li(note) for note in stats["notes"]]),
        ],
    )


def window_stats_records(stats: dict) -> list[dict]:
    if stats["windows"].empty:
        return []
    return stats["windows"].to_dict("records")


currency_options = [{"label": value, "value": value} for value in sorted(NEWS["currency"].dropna().unique())]
impact_options = [{"label": value, "value": value} for value in IMPACT_ORDER]
initial_start = NEWS["time"].min().date().isoformat()
initial_end = NEWS["time"].max().date().isoformat()
DEFAULT_NEWS_CURRENCIES = ["USD"]
DEFAULT_NEWS_IMPACTS = ["High", "Moderate"]
DEFAULT_BEFORE_MINUTES = 15
DEFAULT_AFTER_MINUTES = 60
DEFAULT_CHART_WINDOW_MINUTES = 75


def default_trade_index() -> int | None:
    if TRADES.empty:
        return None
    for _, trade in TRADES.iterrows():
        nearby = news_rows_for_trade_window(
            trade,
            DEFAULT_BEFORE_MINUTES,
            DEFAULT_AFTER_MINUTES,
            DEFAULT_NEWS_CURRENCIES,
            DEFAULT_NEWS_IMPACTS,
        )
        if not nearby.empty:
            return int(trade["trade_index"])
    return int(TRADES.iloc[0]["trade_index"])


DEFAULT_TRADE_INDEX = default_trade_index()
INITIAL_FILTERED = filter_news(None, ["High", "Moderate"], initial_start, initial_end, None)
INITIAL_SELECTED = grid_records(INITIAL_FILTERED.head(1))
INITIAL_TRADE_SELECTED = trade_grid_records(TRADES.head(1))
INITIAL_FILTER_STATS = filter_simulation_stats_for_trade(
    DEFAULT_TRADE_INDEX,
    DEFAULT_NEWS_CURRENCIES,
    DEFAULT_NEWS_IMPACTS,
    DEFAULT_BEFORE_MINUTES,
    DEFAULT_AFTER_MINUTES,
)
INITIAL_GLOBAL_FILTER_STATS = filter_simulation_stats(
    DEFAULT_NEWS_CURRENCIES,
    DEFAULT_NEWS_IMPACTS,
    DEFAULT_BEFORE_MINUTES,
    DEFAULT_AFTER_MINUTES,
)
INITIAL_TRADE = selected_trade_from_index(DEFAULT_TRADE_INDEX)
INITIAL_NEARBY_NEWS = (
    news_rows_for_trade_window(
        INITIAL_TRADE,
        DEFAULT_BEFORE_MINUTES,
        DEFAULT_AFTER_MINUTES,
        DEFAULT_NEWS_CURRENCIES,
        DEFAULT_NEWS_IMPACTS,
    )
    if INITIAL_TRADE is not None
    else pd.DataFrame()
)
INITIAL_NEARBY_TITLE = f"{len(INITIAL_NEARBY_NEWS):,} news events in selected trade window"
if INITIAL_NEARBY_NEWS.empty:
    INITIAL_NEARBY_TITLE = "No news in selected trade window with current filters"

app = Dash(__name__)
server = app.server

app.layout = html.Div(
    className="page-shell",
    children=[
        dcc.Store(id="selected-trade-index", data=DEFAULT_TRADE_INDEX),
        html.Header(
            className="topbar",
            children=[
                html.Div(
                    [
                        html.H1("MT5 Trade News Review"),
                        html.Div(
                            f"{len(TRADES):,} trades | {len(NEWS):,} events | {initial_start} to {initial_end} | {DATA_SOURCE_NAME}",
                            className="subtitle",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div("Dash + AG Grid", className="stack-label"),
                        html.Div("Week 3 prototype", className="stack-subtitle"),
                    ],
                    className="stack-badge",
                ),
            ],
        ),
        html.Section(
            className="news-window-shell",
            children=[
                html.Div(
                    className="workbench-header",
                    children=[
                        html.Div(
                            [
                                html.H2("News Window Around Selected Trade"),
                                html.Div(
                                    "Select a trade, inspect nearby MT5 news, then test whether filtering trades around those news events improves the strategy stats.",
                                    className="subtitle",
                                ),
                            ]
                        ),
                    ],
                ),
                html.Div(
                    className="trade-controls",
                    children=[
                        html.Div(
                            [
                                html.Label("News currency"),
                                dcc.Dropdown(
                                    id="news-window-currency-filter",
                                    options=currency_options,
                                    multi=True,
                                    value=DEFAULT_NEWS_CURRENCIES,
                                    placeholder="All currencies",
                                    className="control",
                                ),
                            ],
                            className="filter-control",
                        ),
                        html.Div(
                            [
                                html.Label("News impact"),
                                dcc.Dropdown(
                                    id="news-window-impact-filter",
                                    options=impact_options,
                                    multi=True,
                                    value=DEFAULT_NEWS_IMPACTS,
                                    className="control",
                                ),
                            ],
                            className="filter-control",
                        ),
                        html.Div(
                            [
                                html.Label("Minutes before entry"),
                                dcc.Input(id="filter-before-minutes", type="number", value=DEFAULT_BEFORE_MINUTES, min=0, step=5, className="text-input"),
                            ],
                            className="filter-control",
                        ),
                        html.Div(
                            [
                                html.Label("Minutes after exit"),
                                dcc.Input(id="filter-after-minutes", type="number", value=DEFAULT_AFTER_MINUTES, min=0, step=5, className="text-input"),
                            ],
                            className="filter-control",
                        ),
                    ],
                ),
                html.Div(
                    className="strategy-table-panel mfe-mae-panel",
                    children=[
                        html.Div("MAE/MFE trade map", className="panel-title"),
                        dcc.Graph(
                            id="mfe-mae-scatter",
                            figure=mfe_mae_scatter_figure(DEFAULT_TRADE_INDEX),
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
                html.Div(
                    className="news-window-grid",
                    children=[
                        html.Section(
                            className="strategy-table-panel trade-selector-panel",
                            children=[
                                html.Div("Validation trades", className="panel-title"),
                                dag.AgGrid(
                                    id="trade-selector-grid",
                                    rowData=trade_grid_records(TRADES),
                                    columnDefs=[
                                        {
                                            "field": "trade_index",
                                            "headerName": "#",
                                            "width": 76,
                                            "checkboxSelection": True,
                                            "headerCheckboxSelection": False,
                                        },
                                        {"field": "direction", "headerName": "Side", "width": 82},
                                        {"field": "open_time", "headerName": "Open", "width": 154},
                                        {"field": "duration_min", "headerName": "Min", "width": 78},
                                        {"field": "pnl", "headerName": "PnL", "width": 86},
                                        {"field": "reason", "headerName": "Exit", "width": 78},
                                        {"field": "mfe_pts", "headerName": "MFE", "width": 86},
                                        {"field": "mae_pts", "headerName": "MAE", "width": 86},
                                    ],
                                    defaultColDef={"sortable": True, "filter": True, "resizable": True},
                                    dashGridOptions={
                                        "rowSelection": "single",
                                        "suppressCellFocus": True,
                                        "animateRows": False,
                                    },
                                    selectedRows=[{"trade_index": DEFAULT_TRADE_INDEX}],
                                    className="ag-theme-quartz compact-grid",
                                    style={"height": "420px", "width": "100%"},
                                ),
                            ],
                        ),
                        html.Section(
                            className="news-window-main",
                            children=[
                                html.Div(id="selected-trade-card", children=selected_trade_summary(INITIAL_TRADE)),
                                html.Div(
                                    className="strategy-table-panel trade-price-panel",
                                    children=[
                                        html.Div("Selected trade price chart (XAUUSD M1)", className="panel-title"),
                                        dcc.Graph(
                                            id="selected-trade-price-chart",
                                            figure=trade_chart_figure(
                                                INITIAL_TRADE,
                                                DEFAULT_CHART_WINDOW_MINUTES,
                                                DEFAULT_NEWS_CURRENCIES,
                                                DEFAULT_NEWS_IMPACTS,
                                            ),
                                            config={"displayModeBar": False},
                                        ),
                                    ],
                                ),
                                html.Div(id="news-window-kpis", children=news_window_kpis(news_window_summary(INITIAL_NEARBY_NEWS))),
                                dcc.Graph(
                                    id="news-window-timeline",
                                    figure=trade_news_timeline_figure(
                                        INITIAL_TRADE,
                                        INITIAL_NEARBY_NEWS,
                                        DEFAULT_BEFORE_MINUTES,
                                        DEFAULT_AFTER_MINUTES,
                                    ),
                                    config={"displayModeBar": False},
                                ),
                                html.Div(
                                    className="strategy-table-panel nearby-news-panel",
                                    children=[
                                        html.Div(id="nearby-news-title", className="panel-title", children=INITIAL_NEARBY_TITLE),
                                        dag.AgGrid(
                                            id="nearby-news-grid",
                                            rowData=nearby_news_records(INITIAL_NEARBY_NEWS),
                                            columnDefs=[
                                                {"field": "display_time", "headerName": "Time", "width": 154},
                                                {"field": "event", "headerName": "Event", "flex": 1.8, "minWidth": 260},
                                                {"field": "currency", "headerName": "Currency", "width": 100},
                                                {"field": "impact", "headerName": "Impact", "width": 104},
                                                {"field": "forecast_display", "headerName": "Forecast", "width": 108},
                                                {"field": "result_display", "headerName": "Result", "width": 108},
                                                {"field": "minutes_from_entry", "headerName": "Min Entry", "width": 104},
                                                {"field": "minutes_from_exit", "headerName": "Min Exit", "width": 96},
                                            ],
                                            defaultColDef={"sortable": True, "filter": True, "resizable": True},
                                            dashGridOptions={"domLayout": "autoHeight", "suppressCellFocus": True},
                                            className="ag-theme-quartz compact-grid",
                                            style={"width": "100%"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="filter-effect-grid",
                    children=[
                        html.Div(
                            className="strategy-table-panel",
                            children=[
                                html.Div("Would filtering this news window improve stats?", className="panel-title"),
                                html.Div(id="filter-decision", children=filter_decision(INITIAL_FILTER_STATS)),
                                dag.AgGrid(
                                    id="filter-stats-grid",
                                    rowData=INITIAL_FILTER_STATS.to_dict("records"),
                                    columnDefs=[
                                        {"field": "set", "headerName": "Set", "flex": 1.4, "minWidth": 190},
                                        {"field": "trades", "headerName": "Trades", "width": 92},
                                        {"field": "pnl", "headerName": "PnL", "width": 88},
                                        {"field": "avg_pnl", "headerName": "Avg", "width": 86},
                                        {"field": "win_rate", "headerName": "Win %", "width": 86},
                                        {"field": "avg_mfe_pts", "headerName": "MFE", "width": 86},
                                        {"field": "avg_mae_pts", "headerName": "MAE", "width": 86},
                                    ],
                                    defaultColDef={"sortable": True, "resizable": True},
                                    dashGridOptions={"domLayout": "autoHeight", "suppressCellFocus": True},
                                    className="ag-theme-quartz compact-grid",
                                    style={"width": "100%"},
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="equity-curve-chart",
                            figure=equity_curve_figure_for_trade(
                                DEFAULT_TRADE_INDEX,
                                DEFAULT_NEWS_CURRENCIES,
                                DEFAULT_NEWS_IMPACTS,
                                DEFAULT_BEFORE_MINUTES,
                                DEFAULT_AFTER_MINUTES,
                            ),
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
                html.Div(
                    className="filter-effect-grid global-filter-effect-grid",
                    children=[
                        html.Div(
                            className="strategy-table-panel",
                            children=[
                                html.Div("Global test: current news filter across all trades", className="panel-title"),
                                html.Div(id="global-filter-decision", children=filter_decision(INITIAL_GLOBAL_FILTER_STATS)),
                                dag.AgGrid(
                                    id="global-filter-stats-grid",
                                    rowData=INITIAL_GLOBAL_FILTER_STATS.to_dict("records"),
                                    columnDefs=[
                                        {"field": "set", "headerName": "Set", "flex": 1.4, "minWidth": 190},
                                        {"field": "trades", "headerName": "Trades", "width": 92},
                                        {"field": "pnl", "headerName": "PnL", "width": 88},
                                        {"field": "avg_pnl", "headerName": "Avg", "width": 86},
                                        {"field": "win_rate", "headerName": "Win %", "width": 86},
                                        {"field": "avg_mfe_pts", "headerName": "MFE", "width": 86},
                                        {"field": "avg_mae_pts", "headerName": "MAE", "width": 86},
                                    ],
                                    defaultColDef={"sortable": True, "resizable": True},
                                    dashGridOptions={"domLayout": "autoHeight", "suppressCellFocus": True},
                                    className="ag-theme-quartz compact-grid",
                                    style={"width": "100%"},
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="global-equity-curve-chart",
                            figure=equity_curve_figure(
                                DEFAULT_NEWS_CURRENCIES,
                                DEFAULT_NEWS_IMPACTS,
                                DEFAULT_BEFORE_MINUTES,
                                DEFAULT_AFTER_MINUTES,
                            ),
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("selected-trade-index", "data"),
    Input("trade-selector-grid", "selectedRows"),
    Input("mfe-mae-scatter", "clickData"),
    State("selected-trade-index", "data"),
)
def update_selected_trade_index(selected_rows, scatter_click, current_trade_index):
    if ctx.triggered_id == "mfe-mae-scatter":
        trade = selected_trade_from_click(scatter_click)
        if trade is not None:
            return int(trade["trade_index"])
    if selected_rows:
        return int(selected_rows[0]["trade_index"])
    return current_trade_index or (1 if not TRADES.empty else None)


@app.callback(
    Output("selected-trade-card", "children"),
    Output("mfe-mae-scatter", "figure"),
    Output("selected-trade-price-chart", "figure"),
    Output("news-window-kpis", "children"),
    Output("news-window-timeline", "figure"),
    Output("nearby-news-grid", "rowData"),
    Output("nearby-news-title", "children"),
    Input("selected-trade-index", "data"),
    Input("filter-before-minutes", "value"),
    Input("filter-after-minutes", "value"),
    Input("news-window-currency-filter", "value"),
    Input("news-window-impact-filter", "value"),
)
def update_selected_trade(selected_trade_index, before_minutes, after_minutes, currencies, impacts):
    trade = selected_trade_from_index(selected_trade_index)
    card = selected_trade_summary(trade)
    if trade is None:
        return (
            card,
            mfe_mae_scatter_figure(None),
            empty_figure("Selected trade price chart"),
            news_window_kpis(news_window_summary(pd.DataFrame())),
            empty_figure("News window timeline"),
            [],
            "No selected trade",
        )
    nearby = news_rows_for_trade_window(trade, before_minutes, after_minutes, currencies, impacts)
    title = f"{len(nearby):,} news events in selected trade window"
    if nearby.empty:
        title = "No news in selected trade window with current filters"
    summary = news_window_summary(nearby)
    return (
        card,
        mfe_mae_scatter_figure(selected_trade_index),
        trade_chart_figure(trade, DEFAULT_CHART_WINDOW_MINUTES, currencies, impacts),
        news_window_kpis(summary),
        trade_news_timeline_figure(trade, nearby, before_minutes, after_minutes),
        nearby_news_records(nearby),
        title,
    )


@app.callback(
    Output("filter-stats-grid", "rowData"),
    Output("equity-curve-chart", "figure"),
    Output("filter-decision", "children"),
    Input("selected-trade-index", "data"),
    Input("news-window-currency-filter", "value"),
    Input("news-window-impact-filter", "value"),
    Input("filter-before-minutes", "value"),
    Input("filter-after-minutes", "value"),
)
def update_filter_stats(selected_trade_index, currencies, impacts, before_minutes, after_minutes):
    stats = filter_simulation_stats_for_trade(
        selected_trade_index,
        currencies,
        impacts,
        before_minutes,
        after_minutes,
    )
    return (
        stats.to_dict("records"),
        equity_curve_figure_for_trade(
            selected_trade_index,
            currencies,
            impacts,
            before_minutes,
            after_minutes,
        ),
        filter_decision(stats),
    )


@app.callback(
    Output("global-filter-stats-grid", "rowData"),
    Output("global-equity-curve-chart", "figure"),
    Output("global-filter-decision", "children"),
    Input("news-window-currency-filter", "value"),
    Input("news-window-impact-filter", "value"),
    Input("filter-before-minutes", "value"),
    Input("filter-after-minutes", "value"),
)
def update_global_filter_stats(currencies, impacts, before_minutes, after_minutes):
    stats = filter_simulation_stats(currencies, impacts, before_minutes, after_minutes)
    return (
        stats.to_dict("records"),
        equity_curve_figure(currencies, impacts, before_minutes, after_minutes),
        filter_decision(stats),
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
