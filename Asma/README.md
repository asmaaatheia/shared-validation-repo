# MT5 News Grid Prototype

Dash prototype for John’s week 3 news grid/chart task.

## What it does

- Loads the May 25-30, 2026 economic-calendar CSV from `data/news_events.csv`.
- Uses the same observation window as the MA/vectorbt validation week.
- Displays the news in Dash AG Grid with sorting, filtering, floating filters, pagination, and single-row selection.
- Updates Plotly charts when a news row is selected:
  - events by day and impact
  - event heatmap by currency and hour
  - forecast/result surprise scatter
- Includes top-level filters for currency, impact, date range, and event search.

## MT5 Data Source

John asked for MT5 news. The app now prefers `data/mt5_news_events.csv` when that file exists.

To generate it from MT5:

1. Copy `mt5/ExportCalendarNews_20260525_20260530.mq5` into MT5 `MQL5/Scripts`.
2. Compile it in MetaEditor.
3. Run it in MT5.
4. Copy the generated `MQL5/Files/mt5_news_events.csv` into this app's `data` folder.
5. Restart the Dash app.

## Financial Juice Scraper

John also shared a Financial Juice scraper source. That source is different from MT5 Economic Calendar news:

- MT5 news is structured economic-calendar data: time, event, currency, impact, forecast, result.
- Financial Juice is live headline/news-feed data: title, text, URL, published date, labels/source, and active/red flag.

This project includes a cleaned standalone version at:

```text
financial_juice_scraper.py
```

It does not include hardcoded credentials. If login is needed, provide credentials through environment variables or CLI flags.

Install scraper dependencies:

```powershell
..\..\work\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Quick scrape using an existing Chrome profile/session:

```powershell
..\..\work\.venv\Scripts\python.exe financial_juice_scraper.py --quick --headless --session "C:\Users\HP\AppData\Local\Google\Chrome\User Data" --save-to data\financial_juice_news.csv
```

History scrape:

```powershell
..\..\work\.venv\Scripts\python.exe financial_juice_scraper.py --history --headless --session "C:\Users\HP\AppData\Local\Google\Chrome\User Data" --end-date 2026-05-25 --save-to data\financial_juice_news.csv
```

Login with environment variables:

```powershell
$env:FJ_EMAIL="your-email"
$env:FJ_PASSWORD="your-password"
..\..\work\.venv\Scripts\python.exe financial_juice_scraper.py --login --quick --headless --session "C:\Users\HP\AppData\Local\Google\Chrome\User Data" --save-to data\financial_juice_news.csv
```

## Fallback Data Note

The current bundled rows are a temporary non-MT5 fallback based on the Forex Factory May 2026 calendar page for May 25-30. Replace them with the MT5 export before treating the dashboard as the final John version.

## Run

From this folder:

```powershell
..\..\work\.venv\Scripts\python.exe app.py
```

Then open:

```text
http://127.0.0.1:8050
```
