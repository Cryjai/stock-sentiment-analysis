import html
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
from finvizfinance.quote import finvizfinance


OUTPUT_DIR = Path(".")
BAR_CHART = OUTPUT_DIR / "sentiment_analysis.png"
PIE_CHART = OUTPUT_DIR / "sentiment_pie.png"
HTML_REPORT = OUTPUT_DIR / "index.html"


def fetch_news(ticker: str) -> pd.DataFrame:
    stock = finvizfinance(ticker)
    news_df = stock.ticker_news()

    if news_df is None or news_df.empty:
        raise ValueError("冇新聞，你係咪入錯咗 ticker，定隻股票靜到似失蹤人口？")

    required_cols = [col for col in ["Title", "Date"] if col in news_df.columns]
    df = news_df[required_cols].copy()

    if "Title" not in df.columns:
        raise ValueError("新聞資料冇 Title 欄，個資料源玩你。")

    df = df.dropna(subset=["Title"])
    df["Title"] = df["Title"].astype(str).str.strip()
    df = df[df["Title"] != ""]

    if df.empty:
        raise ValueError("有資料但冇有效新聞標題。")

    return df.reset_index(drop=True)


def analyze_sentiment(text: str) -> tuple[float, float]:
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def classify_sentiment(score: float) -> str:
    if score > 0.1:
        return "Positive"
    if score < -0.1:
        return "Negative"
    return "Neutral"


def build_analysis(news_df: pd.DataFrame) -> pd.DataFrame:
    polarities = []
    subjectivities = []
    labels = []

    for title in news_df["Title"]:
        polarity, subjectivity = analyze_sentiment(title)
        polarities.append(polarity)
        subjectivities.append(subjectivity)
        labels.append(classify_sentiment(polarity))

    analyzed = news_df.copy()
    analyzed["Polarity"] = polarities
    analyzed["Subjectivity"] = subjectivities
    analyzed["Label"] = labels
    return analyzed


def build_summary(analyzed_df: pd.DataFrame) -> dict:
    total = len(analyzed_df)
    avg_sentiment = analyzed_df["Polarity"].mean()
    avg_subjectivity = analyzed_df["Subjectivity"].mean()

    label_counts = analyzed_df["Label"].value_counts()
    positive = int(label_counts.get("Positive", 0))
    neutral = int(label_counts.get("Neutral", 0))
    negative = int(label_counts.get("Negative", 0))

    most_positive = analyzed_df.sort_values("Polarity", ascending=False).iloc[0]
    most_negative = analyzed_df.sort_values("Polarity", ascending=True).iloc[0]

    return {
        "total": total,
        "avg_sentiment": avg_sentiment,
        "avg_subjectivity": avg_subjectivity,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "positive_ratio": positive / total if total else 0,
        "negative_ratio": negative / total if total else 0,
        "most_positive_title": str(most_positive["Title"]),
        "most_positive_score": float(most_positive["Polarity"]),
        "most_negative_title": str(most_negative["Title"]),
        "most_negative_score": float(most_negative["Polarity"]),
    }


def plot_bar_chart(analyzed_df: pd.DataFrame, ticker: str) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = analyzed_df["Polarity"].apply(
        lambda x: "#22c55e" if x > 0.1 else "#ef4444" if x < -0.1 else "#94a3b8"
    )

    ax.bar(range(len(analyzed_df)), analyzed_df["Polarity"], color=colors)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{ticker} News Sentiment by Headline", fontsize=16, weight="bold")
    ax.set_xlabel("Headline Index")
    ax.set_ylabel("Polarity Score")
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(BAR_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pie_chart(summary: dict, ticker: str) -> None:
    labels = ["Positive", "Neutral", "Negative"]
    sizes = [summary["positive"], summary["neutral"], summary["negative"]]
    colors = ["#22c55e", "#94a3b8", "#ef4444"]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    ax.set_title(f"{ticker} Sentiment Distribution", fontsize=14, weight="bold")

    plt.tight_layout()
    plt.savefig(PIE_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_table_rows(analyzed_df: pd.DataFrame) -> str:
    rows = []
    ranked = analyzed_df.sort_values("Polarity", ascending=False).reset_index(drop=True)

    for _, row in ranked.iterrows():
        title = html.escape(str(row["Title"]))
        date_val = html.escape(str(row["Date"])) if "Date" in ranked.columns else "-"
        label = row["Label"]
        polarity = row["Polarity"]
        subjectivity = row["Subjectivity"]

        badge_class = {
            "Positive": "badge positive",
            "Neutral": "badge neutral",
            "Negative": "badge negative",
        }[label]

        rows.append(
            f"""
            <tr>
                <td>{date_val}</td>
                <td>{title}</td>
                <td><span class="{badge_class}">{label}</span></td>
                <td>{polarity:.3f}</td>
                <td>{subjectivity:.3f}</td>
            </tr>
            """
        )

    return "\n".join(rows)


def build_html_report(ticker: str, summary: dict, analyzed_df: pd.DataFrame) -> str:
    table_rows = render_table_rows(analyzed_df)

    most_positive_title = html.escape(summary["most_positive_title"])
    most_negative_title = html.escape(summary["most_negative_title"])

    overall_label = (
        "Bullish"
        if summary["avg_sentiment"] > 0.1
        else "Bearish"
        if summary["avg_sentiment"] < -0.1
        else "Mixed / Neutral"
    )

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} Sentiment Dashboard | AhCry Finance</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            font-family: Inter, Arial, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
        }}
        .wrapper {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 32px 20px 60px;
        }}
        .hero {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 24px;
            flex-wrap: wrap;
            margin-bottom: 28px;
        }}
        .brand {{
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #94a3b8;
            margin-bottom: 12px;
        }}
        h1 {{
            font-size: 42px;
            margin: 0 0 10px;
        }}
        .subtitle {{
            color: #cbd5e1;
            max-width: 760px;
            line-height: 1.6;
        }}
        .signal {{
            background: #111827;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 16px 18px;
            min-width: 220px;
        }}
        .signal-label {{
            color: #94a3b8;
            font-size: 13px;
            margin-bottom: 8px;
        }}
        .signal-value {{
            font-size: 28px;
            font-weight: 800;
            color: #f8fafc;
        }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin: 24px 0 32px;
        }}
        .card {{
            background: #111827;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 18px;
        }}
        .card .label {{
            color: #94a3b8;
            font-size: 13px;
            margin-bottom: 10px;
        }}
        .card .value {{
            font-size: 28px;
            font-weight: 800;
        }}
        .charts {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 32px;
        }}
        .panel {{
            background: #111827;
            border: 1px solid #334155;
            border-radius: 18px;
            padding: 18px;
        }}
        .panel h2 {{
            margin-top: 0;
            font-size: 20px;
        }}
        .panel img {{
            width: 100%;
            border-radius: 12px;
            margin-top: 10px;
        }}
        .insight-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 32px;
        }}
        .headline-box {{
            background: #111827;
            border: 1px solid #334155;
            border-radius: 18px;
            padding: 18px;
        }}
        .headline-box h3 {{
            margin-top: 0;
            font-size: 18px;
        }}
        .headline-box p {{
            color: #cbd5e1;
            line-height: 1.6;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
        }}
        .positive {{ background: rgba(34, 197, 94, 0.18); color: #86efac; }}
        .neutral {{ background: rgba(148, 163, 184, 0.18); color: #cbd5e1; }}
        .negative {{ background: rgba(239, 68, 68, 0.18); color: #fca5a5; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            overflow: hidden;
            background: #111827;
            border: 1px solid #334155;
            border-radius: 16px;
        }}
        th, td {{
            padding: 14px 12px;
            border-bottom: 1px solid #1e293b;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            color: #93c5fd;
            background: #0b1220;
            font-size: 14px;
        }}
        tr:hover {{
            background: rgba(148, 163, 184, 0.06);
        }}
        .footer {{
            color: #94a3b8;
            font-size: 13px;
            margin-top: 22px;
            line-height: 1.7;
        }}
        @media (max-width: 900px) {{
            .charts,
            .insight-grid {{
                grid-template-columns: 1fr;
            }}
            h1 {{
                font-size: 32px;
            }}
        }}
    </style>
</head>
<body>
    <div class="wrapper">
        <div class="hero">
            <div>
                <div class="brand">AhCry Finance · Sentiment Intelligence</div>
                <h1>{ticker} News Sentiment Dashboard</h1>
                <div class="subtitle">
                    AI-powered headline sentiment analysis using Finviz news data and TextBlob baseline NLP scoring.
                    This report helps quickly gauge whether recent media tone is broadly bullish, bearish, or mixed.
                </div>
            </div>
            <div class="signal">
                <div class="signal-label">Overall Signal</div>
                <div class="signal-value">{overall_label}</div>
            </div>
        </div>

        <div class="cards">
            <div class="card">
                <div class="label">Total Headlines</div>
                <div class="value">{summary["total"]}</div>
            </div>
            <div class="card">
                <div class="label">Average Polarity</div>
                <div class="value">{summary["avg_sentiment"]:.3f}</div>
            </div>
            <div class="card">
                <div class="label">Average Subjectivity</div>
                <div class="value">{summary["avg_subjectivity"]:.3f}</div>
            </div>
            <div class="card">
                <div class="label">Positive Ratio</div>
                <div class="value">{summary["positive_ratio"] * 100:.1f}%</div>
            </div>
            <div class="card">
                <div class="label">Negative Ratio</div>
                <div class="value">{summary["negative_ratio"] * 100:.1f}%</div>
            </div>
        </div>

        <div class="charts">
            <div class="panel">
                <h2>Headline-Level Sentiment</h2>
                <img src="sentiment_analysis.png" alt="Bar chart of sentiment polarity for each headline">
            </div>
            <div class="panel">
                <h2>Distribution</h2>
                <img src="sentiment_pie.png" alt="Pie chart showing positive neutral and negative headline distribution">
            </div>
        </div>

        <div class="insight-grid">
            <div class="headline-box">
                <h3>Most Positive Headline</h3>
                <p>{most_positive_title}</p>
                <span class="badge positive">{summary["most_positive_score"]:.3f}</span>
            </div>
            <div class="headline-box">
                <h3>Most Negative Headline</h3>
                <p>{most_negative_title}</p>
                <span class="badge negative">{summary["most_negative_score"]:.3f}</span>
            </div>
        </div>

        <div class="panel">
            <h2>Ranked Headlines Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Headline</th>
                        <th>Label</th>
                        <th>Polarity</th>
                        <th>Subjectivity</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>

        <div class="footer">
            Baseline model: TextBlob polarity and subjectivity scoring. Source: Finviz news feed.<br>
            For research/demo use only — this is not investment advice, just a smarter version of financial gossip.
        </div>
    </div>
</body>
</html>
"""


def save_html_report(content: str) -> None:
    HTML_REPORT.write_text(content, encoding="utf-8")


def main():
    ticker = input("輸入股票代號 (e.g. AAPL): ").strip().upper()

    if not ticker:
        print("你連 ticker 都唔入，分析空氣咩。")
        return

    try:
        news_df = fetch_news(ticker)
        analyzed_df = build_analysis(news_df)
        summary = build_summary(analyzed_df)

        plot_bar_chart(analyzed_df, ticker)
        plot_pie_chart(summary, ticker)

        html_report = build_html_report(ticker, summary, analyzed_df)
        save_html_report(html_report)

        print(f"成功生成 {ticker} 情緒分析報告。")
        print(f"總新聞數: {summary['total']}")
        print(f"平均情緒分數: {summary['avg_sentiment']:.3f}")
        print(f"正面 / 中立 / 負面: {summary['positive']} / {summary['neutral']} / {summary['negative']}")
        print("輸出檔案：sentiment_analysis.png, sentiment_pie.png, index.html")

    except Exception as e:
        print(f"出事啦：{e}")


if __name__ == "__main__":
    main()
