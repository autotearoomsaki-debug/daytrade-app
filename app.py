import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
import requests

warnings.filterwarnings("ignore")

# ============================================================
# Page Config
# ============================================================
st.set_page_config(page_title="デイトレ振り返り", layout="wide", page_icon="📊")

# ============================================================
# Custom CSS
# ============================================================
st.markdown(
    """
<style>
/* Hide default Streamlit padding for tighter layout */
.block-container { padding-top: 2.5rem; padding-bottom: 0rem; }

/* Metric card */
.mc {
    background: #1a1a2e;
    border: 1px solid #2a2a40;
    border-radius: 10px;
    padding: 18px 20px;
    height: 100%;
}
.mc .label { color: #888; font-size: 13px; margin-bottom: 2px; }
.mc .value { font-size: 30px; font-weight: 700; line-height: 1.2; }
.mc .sub   { color: #666; font-size: 11px; margin-top: 4px; }

/* Badge pills */
.badge-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
.badge {
    background: #1a1a2e;
    border: 1px solid #2a2a40;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 13px;
    color: #ccc;
    white-space: nowrap;
}
.badge .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.dot-green { background: #4caf50; }
.dot-red   { background: #ef5350; }
.dot-gray  { background: #888; }

/* Donut center text */
.donut-center {
    text-align: center;
    margin-top: -8px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Plotly theme helper
# ============================================================
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ccc", size=12),
    margin=dict(l=50, r=20, t=36, b=50),
)

GREEN = "#4caf50"
RED = "#ef5350"
LIGHT_GREEN = "#81c784"
LIGHT_RED = "#e57373"

# ローソク足チャート用インタラクション設定（TradingView風）
CHART_CONFIG = {
    "scrollZoom": True,          # マウスホイールでズーム
    "displayModeBar": True,      # ツールバー表示
    "modeBarButtonsToRemove": [  # 不要なボタンを非表示
        "select2d", "lasso2d", "autoScale2d", "toggleSpikelines",
        "hoverClosestCartesian", "hoverCompareCartesian",
    ],
    "modeBarButtonsToAdd": ["pan2d", "zoom2d", "zoomIn2d", "zoomOut2d", "resetScale2d"],
    "displaylogo": False,
}


# ============================================================
# Supabase helpers
# ============================================================
def _supabase_headers():
    """Supabase REST API用のヘッダーを返す。"""
    key = st.secrets.get("SUPABASE_KEY", "")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _supabase_url():
    return st.secrets.get("SUPABASE_URL", "")


def supabase_available():
    """Supabase設定が存在するか。"""
    return bool(_supabase_url()) and bool(st.secrets.get("SUPABASE_KEY", ""))


def save_session(trade_date, ticker_code, ticker_name, metrics, trades_df, candle_df=None):
    """トレードセッションをSupabaseに保存する。"""
    url = f"{_supabase_url()}/rest/v1/trade_sessions"
    trades_list = trades_df.copy()
    trades_list["entry_time"] = trades_list["entry_time"].astype(str)
    trades_list["exit_time"] = trades_list["exit_time"].astype(str)
    payload = {
        "trade_date": str(trade_date),
        "ticker_code": ticker_code,
        "ticker_name": ticker_name,
        "total_pnl": float(metrics["total_pnl"]),
        "win_rate": float(metrics["win_rate"]),
        "profit_factor": float(metrics["profit_factor"]),
        "trade_count": int(metrics["n_trades"]),
        "trades_json": trades_list.to_dict(orient="records"),
    }
    # ローソク足データも保存
    if candle_df is not None and not candle_df.empty:
        candle_save = candle_df.copy()
        candle_save.index = candle_save.index.astype(str)
        payload["candles_json"] = candle_save.reset_index().to_dict(orient="records")
    resp = requests.post(url, headers=_supabase_headers(), json=payload)
    return resp.status_code in (200, 201)


def load_sessions():
    """保存済みセッション一覧を取得。"""
    url = f"{_supabase_url()}/rest/v1/trade_sessions?select=id,trade_date,ticker_code,ticker_name,total_pnl,win_rate,profit_factor,trade_count,created_at&order=trade_date.desc,created_at.desc"
    resp = requests.get(url, headers=_supabase_headers())
    if resp.status_code == 200:
        return resp.json()
    return []


def load_session_detail(session_id):
    """指定セッションの詳細（trades_json含む）を取得。"""
    url = f"{_supabase_url()}/rest/v1/trade_sessions?id=eq.{session_id}&select=*"
    resp = requests.get(url, headers=_supabase_headers())
    if resp.status_code == 200 and resp.json():
        return resp.json()[0]
    return None


def delete_session(session_id):
    """セッションを削除。"""
    url = f"{_supabase_url()}/rest/v1/trade_sessions?id=eq.{session_id}"
    resp = requests.delete(url, headers=_supabase_headers())
    return resp.status_code in (200, 204)


# ============================================================
# Helper: SBI raw text parser
# ============================================================
def parse_sbi_raw_text(text):
    """SBI証券の注文一覧テキストをパースしてDataFrameに変換。
    取消完了の注文はスキップ。複数約定は加重平均単価で1行に集約。"""
    import re

    lines = text.strip().split("\n")
    orders = []
    current_order = []

    # 注文番号の行で区切る（数字のみの行 or 数字で始まる行）
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # 新しい注文ブロックの開始判定：4桁以上の数字で始まる
        if re.match(r'^\d{4,}', stripped) and current_order:
            orders.append(current_order)
            current_order = [stripped]
        else:
            current_order.append(stripped)
    if current_order:
        orders.append(current_order)

    rows = []
    for order_lines in orders:
        block = "\n".join(order_lines)

        # 取消完了はスキップ（ただし一部約定は約定分を含める）
        if "取消完了" in block and "一部約定" not in block:
            continue

        # 区分を抽出: 信新売, 信返買, 信新買, 信返売
        cat_match = re.search(r'(信新売|信返買|信新買|信返売|現物買|現物売)', block)
        if not cat_match:
            continue
        cat = cat_match.group(1)

        # 約定行を抽出: "約定\t市場\tMM/DD\nHH:MM:SS\t株数\t単価" パターン
        # または "約定\t東証\t04/16\n09:00:00\t500\t33,000"
        fill_pattern = re.findall(
            r'約定\s+(?:東証|PTS(?:（[^）]+）)?|SOR)\s+(\d{2}/\d{2})\s*\n?\s*(\d{2}:\d{2}:\d{2})\s+(\d[\d,]*)\s+([\d,]+(?:\.\d+)?)',
            block
        )
        if not fill_pattern:
            # 別パターン: 同じ行に全部ある場合
            fill_pattern = re.findall(
                r'約定\s+(?:東証|PTS(?:（[^）]+）)?|SOR)\s+(\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})\s+(\d[\d,]*)\s+([\d,]+(?:\.\d+)?)',
                block
            )

        if not fill_pattern:
            continue

        # 約定データを集計（加重平均単価）
        total_shares = 0
        weighted_price = 0.0
        first_datetime = None

        for date_str, time_str, shares_str, price_str in fill_pattern:
            shares = int(shares_str.replace(",", ""))
            price = float(price_str.replace(",", ""))
            total_shares += shares
            weighted_price += price * shares
            if first_datetime is None:
                first_datetime = f"{date_str} {time_str}"

        if total_shares > 0:
            avg_price = weighted_price / total_shares
            rows.append({
                "区分": cat,
                "注文株数": total_shares,
                "約定単価": avg_price,
                "約定日時": first_datetime,
            })

    if not rows:
        return None
    return pd.DataFrame(rows)


# ============================================================
# Helper: parse datetime
# ============================================================
def parse_datetimes(series, year=None, fallback_date=None):
    """約定日時を解析。日付省略行は直前の日付を引き継ぐ。
    全行が時刻のみの場合は fallback_date を使用。"""
    if year is None:
        year = datetime.now().year
    # fallback_date: "YYYY-MM-DD" 文字列 or date オブジェクト
    if fallback_date is not None:
        last_date = str(fallback_date)
    else:
        last_date = None
    parsed = []
    for v in series:
        s = str(v).strip()
        if "/" in s:
            parts = s.split(" ", 1)
            date_part = parts[0]
            time_part = parts[1] if len(parts) > 1 else "00:00:00"
            month, day = date_part.split("/")
            last_date = f"{year}-{int(month):02d}-{int(day):02d}"
            parsed.append(pd.Timestamp(f"{last_date} {time_part}"))
        else:
            if last_date:
                parsed.append(pd.Timestamp(f"{last_date} {s}"))
            else:
                parsed.append(pd.NaT)
    return parsed


# ============================================================
# Helper: FIFO trade pairing
# ============================================================
def pair_trades_fifo(df):
    """
    信新売↔信返買、信新買↔信返売 を FIFO でペアリング（部分約定対応）。
    エントリー100株×3 → 決済300株×1 のようなケースも正しくマッチ。
    戻り値: (paired DataFrame, unmatched list)
    """
    df = df.sort_values("dt").reset_index(drop=True)

    # キュー: [(row, remaining_shares), ...]
    queues = {"short": [], "long": [], "cash": []}
    trades = []

    # エントリー/エグジットの区分マッピング
    entry_map = {"信新売": "short", "信新買": "long", "現物買": "cash"}
    exit_map = {"信返買": "short", "信返売": "long", "現物売": "cash"}

    for _, row in df.iterrows():
        cat = row["区分"]

        if cat in entry_map:
            side = entry_map[cat]
            queues[side].append({"row": row, "remaining": row["注文株数"]})

        elif cat in exit_map:
            side = exit_map[cat]
            queue = queues[side]
            exit_remaining = row["注文株数"]

            while exit_remaining > 0 and queue:
                entry_item = queue[0]
                entry_row = entry_item["row"]
                match_shares = min(entry_item["remaining"], exit_remaining)

                # PnL計算
                if side == "short":
                    pnl = (entry_row["約定単価"] - row["約定単価"]) * match_shares
                else:
                    pnl = (row["約定単価"] - entry_row["約定単価"]) * match_shares

                trade_side = "short" if side == "short" else "long"
                trades.append(_make_trade(entry_row, row, trade_side, pnl, match_shares))

                entry_item["remaining"] -= match_shares
                exit_remaining -= match_shares

                if entry_item["remaining"] <= 0:
                    queue.pop(0)

    unmatched = []
    for q in queues.values():
        for item in q:
            if item["remaining"] > 0:
                unmatched.append({"row": item["row"], "remaining": item["remaining"]})

    return pd.DataFrame(trades), unmatched


def _make_trade(entry, exit_, side, pnl, shares=None):
    if shares is None:
        shares = entry["注文株数"]
    hold_minutes = (exit_["dt"] - entry["dt"]).total_seconds() / 60
    entry_value = entry["約定単価"] * shares
    return {
        "entry_time": entry["dt"],
        "exit_time": exit_["dt"],
        "entry_price": entry["約定単価"],
        "exit_price": exit_["約定単価"],
        "shares": shares,
        "side": side,
        "pnl": pnl,
        "hold_min": round(hold_minutes, 1),
        "pnl_rate": pnl / entry_value if entry_value else 0,
        "entry_order": entry.get("注文番号", ""),
        "exit_order": exit_.get("注文番号", ""),
    }


# ============================================================
# Helper: metrics calculation
# ============================================================
def calculate_metrics(trades_df):
    if trades_df.empty:
        return {}

    pnl = trades_df["pnl"]
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    n = len(trades_df)
    n_win = len(wins)
    n_loss = len(losses)

    gross_profit = wins["pnl"].sum() if not wins.empty else 0
    gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 0
    avg_win = wins["pnl"].mean() if not wins.empty else 0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0

    # Max drawdown
    cum = pnl.cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = dd.min()
    if max_dd < 0:
        dd_end_idx = dd.idxmin()
        dd_start_idx = cum.loc[:dd_end_idx].idxmax()
        dd_start_time = trades_df.loc[dd_start_idx, "entry_time"]
        dd_end_time = trades_df.loc[dd_end_idx, "exit_time"]
    else:
        dd_start_time = dd_end_time = None

    # Streaks
    max_win_streak = max_loss_streak = cur_w = cur_l = 0
    for p in pnl:
        if p > 0:
            cur_w += 1
            cur_l = 0
        elif p < 0:
            cur_l += 1
            cur_w = 0
        else:
            cur_w = cur_l = 0
        max_win_streak = max(max_win_streak, cur_w)
        max_loss_streak = max(max_loss_streak, cur_l)

    total_in = (trades_df["entry_price"] * trades_df["shares"]).sum()
    total_out = total_in + pnl.sum()

    return {
        "total_pnl": pnl.sum(),
        "n_trades": n,
        "n_win": n_win,
        "n_loss": n_loss,
        "win_rate": n_win / n * 100 if n else 0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_win": pnl.max(),
        "max_loss": pnl.min(),
        "payoff_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else float("inf"),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "max_drawdown": max_dd,
        "dd_start": dd_start_time,
        "dd_end": dd_end_time,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "avg_hold_min": trades_df["hold_min"].mean(),
        "total_in": total_in,
        "total_out": total_out,
        "pnl_rate": pnl.sum() / total_in * 100 if total_in else 0,
        "max_pnl_rate": trades_df["pnl_rate"].max() * 100,
        "min_pnl_rate": trades_df["pnl_rate"].min() * 100,
        "avg_pnl": pnl.mean(),
    }


# ============================================================
# HTML helpers
# ============================================================
def metric_card(label, value, subtitle="", color="#ffffff"):
    sub_html = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="mc">
        <div class="label">{label}</div>
        <div class="value" style="color:{color}">{value}</div>
        {sub_html}
    </div>"""


def badge(text, dot_class="dot-gray"):
    return f'<span class="badge"><span class="dot {dot_class}"></span>{text}</span>'


# ============================================================
# Chart: PnL Distribution Histogram
# ============================================================
def chart_pnl_histogram(trades_df):
    wins = trades_df[trades_df["pnl"] > 0]["pnl"]
    losses = trades_df[trades_df["pnl"] <= 0]["pnl"]
    fig = go.Figure()
    # Determine common bin edges
    all_pnl = trades_df["pnl"]
    bin_size = max(abs(all_pnl.max() - all_pnl.min()) / 20, 100)
    fig.add_trace(go.Histogram(x=wins, name="利益", marker_color=GREEN, xbins=dict(size=bin_size)))
    fig.add_trace(go.Histogram(x=losses, name="損失", marker_color=RED, xbins=dict(size=bin_size)))
    fig.update_layout(
        **CHART_LAYOUT,
        title="損益分布",
        xaxis_title="損益（円）",
        yaxis_title="回数",
        barmode="overlay",
        showlegend=False,
        height=350,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#666", line_width=1)
    return fig


# ============================================================
# Chart: Time-based Performance
# ============================================================
def chart_time_performance(trades_df):
    df = trades_df.copy()
    df["time_slot"] = df["entry_time"].dt.floor("30min").dt.strftime("%H:%M")

    grouped = df.groupby("time_slot").agg(
        pnl_sum=("pnl", "sum"),
        count=("pnl", "count"),
    ).reset_index()

    colors = [GREEN if v >= 0 else RED for v in grouped["pnl_sum"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["time_slot"],
            y=grouped["pnl_sum"],
            marker_color=colors,
            text=[f"{c}回" for c in grouped["count"]],
            textposition="outside",
            textfont=dict(size=11, color="#aaa"),
        )
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title="時間帯別パフォーマンス",
        xaxis_title="時間帯",
        yaxis_title="損益（円）",
        showlegend=False,
        height=350,
    )
    return fig


# ============================================================
# Chart: Holding Time vs PnL Scatter
# ============================================================
def chart_holding_scatter(trades_df):
    df = trades_df.copy()
    df["color"] = df["pnl"].apply(lambda x: GREEN if x > 0 else RED)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["hold_min"],
            y=df["pnl"],
            mode="markers",
            marker=dict(
                color=df["color"],
                size=8,
                opacity=0.7,
                line=dict(width=0),
            ),
            hovertemplate="保有: %{x}分<br>損益: ¥%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#666", line_width=1)
    fig.update_layout(
        **CHART_LAYOUT,
        title="保有時間 VS 損益",
        xaxis_title="保有時間（分）",
        yaxis_title="損益（円）",
        showlegend=False,
        height=380,
    )
    return fig


# ============================================================
# Chart: Win Rate Donut
# ============================================================
def chart_donut(n_win, n_loss):
    fig = go.Figure(
        go.Pie(
            values=[n_win, n_loss],
            labels=["勝", "敗"],
            hole=0.65,
            marker=dict(colors=[GREEN, RED]),
            textinfo="none",
            hovertemplate="%{label}: %{value}回<extra></extra>",
        )
    )
    donut_layout = {k: v for k, v in CHART_LAYOUT.items() if k != "margin"}
    fig.update_layout(
        **donut_layout,
        showlegend=False,
        height=180,
        width=180,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ============================================================
# Helper: Add SMA lines to candlestick chart
# ============================================================
SMA_CONFIGS = [
    (5,  "#ffeb3b", 1.0),   # 5期間  — 黄色
    (25, "#e040fb", 1.0),   # 25期間 — マゼンタ
    (75, "#29b6f6", 1.0),   # 75期間 — ライトブルー
]


def _add_sma_traces(fig, candle_df, row=1, col=1):
    """ローソク足チャートにSMA(5,25,75)を追加"""
    for period, color, width in SMA_CONFIGS:
        sma = candle_df["Close"].rolling(window=period, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=candle_df["time_label"],
                y=sma,
                mode="lines",
                line=dict(color=color, width=width),
                name=f"SMA{period}",
                hovertemplate=f"SMA{period}: ¥%{{y:,.0f}}<extra></extra>",
            ),
            row=row, col=col,
        )


def _add_vwap_trace(fig, candle_df, row=1, col=1):
    """ローソク足チャートにVWAPを追加"""
    typical = (candle_df["High"] + candle_df["Low"] + candle_df["Close"]) / 3
    vol = candle_df["Volume"]
    vwap = (typical * vol).cumsum() / vol.cumsum()
    fig.add_trace(
        go.Scatter(
            x=candle_df["time_label"],
            y=vwap,
            mode="lines",
            line=dict(color="#ff6b6b", width=1.5, dash="dot"),
            name="VWAP",
            hovertemplate="VWAP: ¥%{y:,.0f}<extra></extra>",
        ),
        row=row, col=col,
    )


# ============================================================
# Chart: Candlestick with trade markers
# ============================================================
def chart_candlestick(candle_df, trades_df, interval_label="1分足"):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.8, 0.2],
    )

    # Format x-axis labels as time only (HH:MM)
    candle_df = candle_df.copy()
    candle_df["time_label"] = candle_df.index.strftime("%H:%M")

    # Map trade times to nearest candle time_label for annotations
    def _nearest_label(dt):
        idx = candle_df.index.get_indexer([dt], method="nearest")[0]
        if 0 <= idx < len(candle_df):
            return candle_df["time_label"].iloc[idx]
        return candle_df["time_label"].iloc[0]

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=candle_df["time_label"],
            open=candle_df["Open"],
            high=candle_df["High"],
            low=candle_df["Low"],
            close=candle_df["Close"],
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            increasing_fillcolor=GREEN,
            decreasing_fillcolor=RED,
            name="",
        ),
        row=1,
        col=1,
    )

    # Volume
    vol_colors = [
        GREEN if c >= o else RED
        for c, o in zip(candle_df["Close"], candle_df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=candle_df["time_label"],
            y=candle_df["Volume"],
            marker_color=vol_colors,
            opacity=0.5,
            name="出来高",
        ),
        row=2,
        col=1,
    )

    # SMA + VWAP
    _add_sma_traces(fig, candle_df, row=1, col=1)
    _add_vwap_trace(fig, candle_df, row=1, col=1)

    # Entry/Exit markers — 売り=↓オレンジ 買い=↑シアン
    SELL_CLR = "#ff9800"  # オレンジ
    BUY_CLR  = "#00e5ff"  # シアン

    for _, t in trades_df.iterrows():
        entry_label = _nearest_label(t["entry_time"])
        exit_label = _nearest_label(t["exit_time"])
        is_sell_entry = t["side"] == "short"

        # Entry
        in_color = SELL_CLR if is_sell_entry else BUY_CLR
        fig.add_annotation(
            x=entry_label,
            y=t["entry_price"],
            text=f"IN {int(t['shares'])}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor=in_color,
            font=dict(size=10, color=in_color),
            bgcolor="rgba(0,0,0,0.7)",
            borderpad=2,
            ax=0,
            ay=-30 if is_sell_entry else 30,
            row=1, col=1,
        )
        # Exit
        out_color = BUY_CLR if is_sell_entry else SELL_CLR
        pnl_sign = "+" if t["pnl"] >= 0 else ""
        fig.add_annotation(
            x=exit_label,
            y=t["exit_price"],
            text=f"{pnl_sign}{t['pnl']:,.0f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor=out_color,
            font=dict(size=10, color=out_color),
            bgcolor="rgba(0,0,0,0.7)",
            borderpad=2,
            ax=0,
            ay=30 if is_sell_entry else -30,
            row=1, col=1,
        )

    fig.update_layout(
        **CHART_LAYOUT,
        title=f"チャート（売買ポイント付き） - {interval_label}",
        xaxis_rangeslider_visible=False,
        xaxis2_title="時刻",
        yaxis_title="株価",
        yaxis2_title="出来高",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        height=600,
        dragmode="pan",
    )
    fig.update_xaxes(type="category", nticks=15, row=1, col=1)
    fig.update_xaxes(type="category", nticks=15, row=2, col=1)
    fig.update_yaxes(fixedrange=False)
    return fig


# ============================================================
# Chart: 1min Full-session (IN/OUT only, wide)
# ============================================================
def chart_candlestick_1m_full(candle_df, trades_df):
    """1分足全場チャート — IN/OUTマーカーのみ、横長で見やすく"""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.85, 0.15],
    )

    candle_df = candle_df.copy()
    candle_df["time_label"] = candle_df.index.strftime("%H:%M")

    def _nearest_label(dt):
        idx = candle_df.index.get_indexer([dt], method="nearest")[0]
        if 0 <= idx < len(candle_df):
            return candle_df["time_label"].iloc[idx]
        return candle_df["time_label"].iloc[0]

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=candle_df["time_label"],
            open=candle_df["Open"],
            high=candle_df["High"],
            low=candle_df["Low"],
            close=candle_df["Close"],
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            increasing_fillcolor=GREEN,
            decreasing_fillcolor=RED,
            name="",
        ),
        row=1, col=1,
    )

    # Volume
    vol_colors = [GREEN if c >= o else RED for c, o in zip(candle_df["Close"], candle_df["Open"])]
    fig.add_trace(
        go.Bar(x=candle_df["time_label"], y=candle_df["Volume"],
               marker_color=vol_colors, opacity=0.4, name=""),
        row=2, col=1,
    )

    # SMA + VWAP
    _add_sma_traces(fig, candle_df, row=1, col=1)
    _add_vwap_trace(fig, candle_df, row=1, col=1)

    # 高視認性カラー（チャートの赤緑と被らない）
    SELL_COLOR = "#ff9800"   # オレンジ — 売り
    BUY_COLOR = "#00e5ff"    # シアン  — 買い

    # IN/OUT scatter markers — 売り=▼(↓) 買い=▲(↑)
    for _, t in trades_df.iterrows():
        entry_label = _nearest_label(t["entry_time"])
        exit_label = _nearest_label(t["exit_time"])
        is_sell_entry = t["side"] == "short"
        price_str_in = f"¥{t['entry_price']:,.0f}"
        price_str_out = f"¥{t['exit_price']:,.0f}"

        if is_sell_entry:
            # ---- 売りエントリー: ▼ オレンジ, 単価は矢印の上 ----
            in_color, in_symbol = SELL_COLOR, "triangle-down"
            in_text = price_str_in
            in_textpos = "top center"
            # ---- 買い決済: ▲ シアン, 単価は矢印の下 ----
            out_color, out_symbol = BUY_COLOR, "triangle-up"
            out_text = price_str_out
            out_textpos = "bottom center"
        else:
            # ---- 買いエントリー: ▲ シアン, 単価は矢印の下 ----
            in_color, in_symbol = BUY_COLOR, "triangle-up"
            in_text = price_str_in
            in_textpos = "bottom center"
            # ---- 売り決済: ▼ オレンジ, 単価は矢印の上 ----
            out_color, out_symbol = SELL_COLOR, "triangle-down"
            out_text = price_str_out
            out_textpos = "top center"

        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[entry_label], y=[t["entry_price"]],
                mode="markers+text",
                marker=dict(symbol=in_symbol, size=13, color=in_color,
                            line=dict(width=1.5, color="#fff")),
                text=[in_text], textposition=in_textpos,
                textfont=dict(size=9, color=in_color, family="monospace"),
                hovertemplate=f"IN {int(t['shares'])}株 @ ¥{t['entry_price']:,.0f}<br>{t['entry_time'].strftime('%H:%M:%S')}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1,
        )

        # Exit marker
        fig.add_trace(
            go.Scatter(
                x=[exit_label], y=[t["exit_price"]],
                mode="markers+text",
                marker=dict(symbol=out_symbol, size=13, color=out_color,
                            line=dict(width=1.5, color="#fff")),
                text=[out_text], textposition=out_textpos,
                textfont=dict(size=9, color=out_color, family="monospace"),
                hovertemplate=f"OUT {int(t['shares'])}株 @ ¥{t['exit_price']:,.0f}<br>{t['exit_time'].strftime('%H:%M:%S')}<br>損益: ¥{t['pnl']:+,.0f}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=1,
        )

    fig.update_layout(
        **CHART_LAYOUT,
        title="1分足チャート（全場） — IN / OUT",
        xaxis_rangeslider_visible=False,
        xaxis2_title="時刻",
        yaxis_title="株価",
        yaxis2_title="出来高",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        height=700,
        dragmode="pan",
    )
    fig.update_xaxes(type="category", nticks=30, row=1, col=1)
    fig.update_xaxes(type="category", nticks=30, row=2, col=1)
    fig.update_yaxes(fixedrange=False)
    return fig


# ============================================================
# Chart: Cumulative PnL
# ============================================================
def chart_cumulative_pnl(trades_df):
    cum = trades_df["pnl"].cumsum()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cum) + 1)),
            y=cum,
            mode="lines+markers",
            line=dict(color=GREEN if cum.iloc[-1] >= 0 else RED, width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(76,175,80,0.1)" if cum.iloc[-1] >= 0 else "rgba(239,83,80,0.1)",
            hovertemplate="トレード #%{x}<br>累積損益: ¥%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#666", line_width=1)
    fig.update_layout(
        **CHART_LAYOUT,
        title="累積損益推移",
        xaxis_title="トレード番号",
        yaxis_title="累積損益（円）",
        showlegend=False,
        height=350,
    )
    return fig


# ============================================================
# Fetch candle data from yfinance
# ============================================================
@st.cache_data(ttl=300)
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stock_name(ticker_code):
    """kabutan から日本語銘柄名を取得（1日キャッシュ）。失敗時は yfinance 英語名にフォールバック。"""
    import re
    from bs4 import BeautifulSoup

    # kabutan から日本語名を取得（優先）
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        r = requests.get(f"https://kabutan.jp/stock/?code={ticker_code}", headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        h1 = soup.find("h1")
        if h1:
            name = re.sub(r"\s*\(.*?\).*$", "", h1.text.strip())
            if name:
                return name
    except Exception:
        pass

    # フォールバック: yf.Search（英語名）
    try:
        results = yf.Search(ticker_code, max_results=3).quotes
        for q in results:
            sym = q.get("symbol", "")
            if sym == f"{ticker_code}.T" or sym == ticker_code:
                name = q.get("shortname") or q.get("longname") or ""
                if name:
                    return name
    except Exception:
        pass

    return ""


def fetch_candles(ticker_code, trade_date, interval="1m"):
    """yfinance から日本株のローソク足データを取得"""
    symbol = f"{ticker_code}.T"
    start = trade_date
    end = trade_date + timedelta(days=1)
    try:
        tk = yf.Ticker(symbol)
        data = tk.history(start=start, end=end, interval=interval)
        if data.empty:
            return None
        # Timezone handling — convert to JST if timezone-aware
        if data.index.tz is not None:
            data.index = data.index.tz_convert("Asia/Tokyo")
        data.index = data.index.tz_localize(None)
        return data
    except Exception:
        return None


# ============================================================
# Main App
# ============================================================
def main():
    # --- Sidebar ---
    with st.sidebar:
        st.header("📊 データ入力")
        input_method = st.radio("入力方法", ["📁 ファイル", "📋 テキスト貼り付け"], horizontal=True)
        if input_method == "📁 ファイル":
            uploaded_file = st.file_uploader("約定履歴ファイル", type=["xlsx", "xls", "csv"])
            # query params でプリセット可能 (?path=...&ticker=...&name=...)
            qp = st.query_params
            file_path = st.text_input("またはファイルパス", value=qp.get("path", ""), placeholder="例: /path/to/trades.xlsx")
            raw_text = None
        else:
            uploaded_file = None
            file_path = None
            qp = st.query_params
            raw_text = st.text_area(
                "SBI証券の注文一覧を貼り付け",
                height=200,
                placeholder="SBI証券の注文一覧ページからコピー＆ペーストしてください",
            )
        st.markdown("---")
        ticker_code = st.text_input("銘柄コード", value=qp.get("ticker", ""), placeholder="例: 285A")
        # 銘柄コードが入力されたら自動で銘柄名を取得
        auto_name = ""
        if ticker_code:
            with st.spinner("銘柄名を検索中..."):
                auto_name = fetch_stock_name(ticker_code)
            if auto_name:
                st.caption(f"✅ {auto_name}")
            else:
                st.caption("銘柄名が見つかりませんでした")
        # 手動上書き用（空欄なら自動取得名を使用）
        stock_name_override = st.text_input(
            "銘柄名（上書き）",
            value=qp.get("name", ""),
            placeholder="空欄なら自動取得",
        )
        stock_name = stock_name_override if stock_name_override else auto_name
        trade_date_input = st.date_input("取引日（日付が無いデータ用）", value=datetime.now().date())
        st.markdown("---")
        st.subheader("ローソク足設定")
        candle_interval = st.radio("足種", ["5m", "1m"], horizontal=True, format_func=lambda x: "1分足" if x == "1m" else "5分足")

    # --- Mode tabs ---
    if supabase_available():
        mode = st.sidebar.radio("モード", ["📊 新規分析", "📁 過去データ", "📈 パターン分析"], horizontal=True)
    else:
        mode = st.sidebar.radio("モード", ["📊 新規分析", "📈 パターン分析"], horizontal=True)

    # --- Pattern analysis view ---
    if mode == "📈 パターン分析":
        render_pattern_analysis()
        return

    # --- Past sessions view ---
    if mode == "📁 過去データ":
        st.markdown("## 📁 過去のトレード記録")
        sessions = load_sessions()
        if not sessions:
            st.info("保存済みのデータはありません。")
            return

        for s in sessions:
            pnl = s["total_pnl"] or 0
            pnl_color = GREEN if pnl >= 0 else RED
            col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 0.5])
            with col1:
                st.markdown(f"**{s['trade_date']}** — {s.get('ticker_name', '')} ({s.get('ticker_code', '')})")
            with col2:
                st.markdown(f"<span style='color:{pnl_color};font-weight:700'>¥{pnl:+,.0f}</span>", unsafe_allow_html=True)
            with col3:
                st.caption(f"勝率 {s.get('win_rate', 0):.1f}% / PF {s.get('profit_factor', 0):.2f}")
            with col4:
                st.caption(f"{s.get('trade_count', 0)}トレード")
            with col5:
                if st.button("🗑️", key=f"del_{s['id']}", help="削除"):
                    delete_session(s["id"])
                    st.rerun()

        st.markdown("---")

        # Session detail selector
        session_options = {
            f"{s['trade_date']} {s.get('ticker_name', '')} ({s.get('ticker_code', '')})": s["id"]
            for s in sessions
        }
        selected_label = st.selectbox("詳細を表示", list(session_options.keys()))
        if selected_label:
            detail = load_session_detail(session_options[selected_label])
            if detail and detail.get("trades_json"):
                detail_df = pd.DataFrame(detail["trades_json"])
                detail_df["entry_time"] = pd.to_datetime(detail_df["entry_time"])
                detail_df["exit_time"] = pd.to_datetime(detail_df["exit_time"])
                detail_df = detail_df.sort_values("entry_time").reset_index(drop=True)

                dm = calculate_metrics(detail_df)
                title_d = detail.get("ticker_name", "") or "不明"
                code_d = detail.get("ticker_code", "") or ""

                # --- Header ---
                st.markdown(f"## {title_d}")
                if code_d:
                    st.caption(code_d)

                # --- Badge row ---
                badges_html = '<div class="badge-row">'
                badges_html += badge(f"最大勝ち ¥{dm['max_win']:+,.0f}", "dot-green")
                badges_html += badge(f"最大負け ¥{dm['max_loss']:+,.0f}", "dot-red")
                badges_html += badge(f"平均勝ち ¥{dm['avg_win']:+,.0f}", "dot-green")
                badges_html += badge(f"平均負け ¥{dm['avg_loss']:+,.0f}", "dot-red")
                badges_html += badge(f"平均保有 {dm['avg_hold_min']:.1f}分", "dot-gray")
                badges_html += "</div>"
                st.markdown(badges_html, unsafe_allow_html=True)

                # --- KPI Row 1 ---
                c1, c2, c3, c4 = st.columns(4)
                dpnl_color = GREEN if dm["total_pnl"] >= 0 else RED
                with c1:
                    st.markdown(metric_card("総損益", f"¥{dm['total_pnl']:+,.0f}", f"平均 ¥{dm['avg_pnl']:+,.0f}/トレード", dpnl_color), unsafe_allow_html=True)
                with c2:
                    st.markdown(metric_card("勝率", f"{dm['win_rate']:.1f}%", f"{dm['n_win']}勝 {dm['n_loss']}敗 {dm['n_trades'] - dm['n_win'] - dm['n_loss']}分"), unsafe_allow_html=True)
                with c3:
                    st.markdown(metric_card("ペイオフレシオ", f"{dm['payoff_ratio']:.2f}", "平均利益÷平均損失"), unsafe_allow_html=True)
                with c4:
                    st.markdown(metric_card("トレード数", f"{dm['n_trades']}", f"保有{dm['avg_hold_min']:.1f}分"), unsafe_allow_html=True)

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                # --- KPI Row 2: Donut + PF + DD + Streaks ---
                c1, c2, c3, c4 = st.columns([1.2, 1, 1.3, 1])
                with c1:
                    donut_fig = chart_donut(dm["n_win"], dm["n_loss"])
                    st.plotly_chart(donut_fig, use_container_width=False, config={"displayModeBar": False})
                    st.markdown(
                        f'<div class="donut-center" style="margin-top:-20px;">'
                        f'<span style="font-size:22px;font-weight:700;color:#fff">{dm["win_rate"]:.1f}%</span><br>'
                        f'<span style="color:#888;font-size:12px">勝率</span><br>'
                        f'<span style="color:{GREEN};font-size:12px">■ {dm["n_win"]}勝</span> '
                        f'<span style="color:{RED};font-size:12px">■ {dm["n_loss"]}敗</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    pf_color = GREEN if dm["profit_factor"] >= 1 else RED
                    st.markdown(
                        metric_card(
                            "プロフィットファクター", f"{dm['profit_factor']:.2f}",
                            f'<span style="color:{GREEN}">利益 ¥{dm["gross_profit"]:,.0f}</span><br>'
                            f'<span style="color:{RED}">損失 ¥{dm["gross_loss"]:,.0f}</span>',
                            pf_color,
                        ), unsafe_allow_html=True,
                    )
                with c3:
                    dd_text = f"¥{dm['max_drawdown']:,.0f}"
                    dd_sub = ""
                    if dm["dd_start"] is not None:
                        dd_sub = f"{dm['dd_start'].strftime('%H:%M')} → {dm['dd_end'].strftime('%H:%M')}"
                    st.markdown(metric_card("最大ドローダウン", dd_text, dd_sub, RED), unsafe_allow_html=True)
                with c4:
                    st.markdown(
                        metric_card(
                            "連勝 / 連敗",
                            f'<span style="color:{GREEN}">{dm["max_win_streak"]}</span>'
                            f'&nbsp;&nbsp;<span style="color:{RED}">{dm["max_loss_streak"]}</span>',
                            "最大連勝　最大連敗",
                        ), unsafe_allow_html=True,
                    )

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                # --- KPI Row 3: PnL rates ---
                c1, c2, c3 = st.columns(3)
                rate_color = GREEN if dm["pnl_rate"] >= 0 else RED
                with c1:
                    st.markdown(metric_card("損益率", f"{dm['pnl_rate']:+.2f}%", f"IN ¥{dm['total_in']:,.0f} / OUT ¥{dm['total_out']:,.0f}", rate_color), unsafe_allow_html=True)
                with c2:
                    st.markdown(metric_card("最大利益率", f"{dm['max_pnl_rate']:+.1f}%", "1トレード最大", GREEN), unsafe_allow_html=True)
                with c3:
                    st.markdown(metric_card("最大損失率", f"{dm['min_pnl_rate']:+.1f}%", "1トレード最大", RED), unsafe_allow_html=True)

                st.markdown("---")

                # --- Candlestick Charts (from saved data) ---
                if detail.get("candles_json"):
                    saved_candles = pd.DataFrame(detail["candles_json"])
                    # Restore index
                    if "index" in saved_candles.columns:
                        saved_candles.index = pd.to_datetime(saved_candles["index"])
                        saved_candles = saved_candles.drop(columns=["index"])
                    elif "Datetime" in saved_candles.columns:
                        saved_candles.index = pd.to_datetime(saved_candles["Datetime"])
                        saved_candles = saved_candles.drop(columns=["Datetime"])
                    else:
                        saved_candles.index = pd.to_datetime(saved_candles.iloc[:, 0])
                        saved_candles = saved_candles.iloc[:, 1:]

                    # Ensure column names match expected format
                    col_map = {}
                    for c in saved_candles.columns:
                        cl = c.lower()
                        if cl == "open":
                            col_map[c] = "Open"
                        elif cl == "high":
                            col_map[c] = "High"
                        elif cl == "low":
                            col_map[c] = "Low"
                        elif cl == "close":
                            col_map[c] = "Close"
                        elif cl == "volume":
                            col_map[c] = "Volume"
                    if col_map:
                        saved_candles = saved_candles.rename(columns=col_map)

                    fig_candle_hist = chart_candlestick(saved_candles, detail_df, "1分足")
                    st.plotly_chart(fig_candle_hist, use_container_width=True, config=CHART_CONFIG)

                    fig_1m_hist = chart_candlestick_1m_full(saved_candles, detail_df)
                    st.plotly_chart(fig_1m_hist, use_container_width=True, config=CHART_CONFIG)

                st.markdown("---")

                # --- Analytics Charts ---
                col_left, col_right = st.columns(2)
                with col_left:
                    st.plotly_chart(chart_pnl_histogram(detail_df), use_container_width=True)
                with col_right:
                    st.plotly_chart(chart_time_performance(detail_df), use_container_width=True)

                col_left2, col_right2 = st.columns(2)
                with col_left2:
                    st.plotly_chart(chart_holding_scatter(detail_df), use_container_width=True)
                with col_right2:
                    st.plotly_chart(chart_cumulative_pnl(detail_df), use_container_width=True)

                st.markdown("---")

                # --- Trade table ---
                st.subheader("トレード一覧")
                show_df = detail_df[["entry_time", "exit_time", "side", "shares", "entry_price", "exit_price", "pnl", "pnl_rate", "hold_min"]].copy()
                show_df.columns = ["エントリー", "エグジット", "売買", "株数", "IN単価", "OUT単価", "損益", "損益率", "保有(分)"]
                show_df["売買"] = show_df["売買"].map({"short": "空売", "long": "買い"})
                show_df["損益率"] = show_df["損益率"].apply(lambda x: f"{x*100:+.2f}%")
                st.dataframe(
                    show_df.style.map(
                        lambda v: f"color: {GREEN}" if isinstance(v, (int, float)) and v > 0 else (f"color: {RED}" if isinstance(v, (int, float)) and v < 0 else ""),
                        subset=["損益"],
                    ).format({"損益": "¥{:+,.0f}", "IN単価": "¥{:,.1f}", "OUT単価": "¥{:,.1f}"}),
                    use_container_width=True,
                    height=min(len(show_df) * 38 + 40, 600),
                )
        return

    if not raw_text and uploaded_file is None and not file_path:
        st.markdown("## 📊 デイトレ振り返りダッシュボード")
        st.info("サイドバーから約定履歴ファイル（xlsx / csv）をアップロードするか、テキスト貼り付けを使用してください。")
        return

    # --- Load Data ---
    try:
        if raw_text:
            raw = parse_sbi_raw_text(raw_text)
            if raw is None:
                st.error("テキストから約定データをパースできませんでした。SBI証券の注文一覧をそのままコピー＆ペーストしてください。")
                return
        elif uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                try:
                    raw = pd.read_csv(uploaded_file, encoding="utf-8")
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    raw = pd.read_csv(uploaded_file, encoding="shift_jis")
            else:
                raw = pd.read_excel(uploaded_file)
        else:
            if file_path.endswith(".csv"):
                try:
                    raw = pd.read_csv(file_path, encoding="utf-8")
                except UnicodeDecodeError:
                    raw = pd.read_csv(file_path, encoding="shift_jis")
            else:
                raw = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return

    # --- Parse datetimes ---
    raw["dt"] = parse_datetimes(raw["約定日時"], fallback_date=trade_date_input)
    raw = raw.dropna(subset=["dt"])

    # --- Pair trades ---
    trades_df, unmatched = pair_trades_fifo(raw)

    if trades_df.empty:
        st.warning("ペアリング可能なトレードが見つかりませんでした。")
        return

    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    m = calculate_metrics(trades_df)

    # --- Header ---
    title = stock_name if stock_name else "不明"
    code = ticker_code if ticker_code else ""
    st.markdown(f"## {title}")
    if code:
        st.caption(code)

    # --- Badge row ---
    badges_html = '<div class="badge-row">'
    badges_html += badge(f"最大勝ち ¥{m['max_win']:+,.0f}", "dot-green")
    badges_html += badge(f"最大負け ¥{m['max_loss']:+,.0f}", "dot-red")
    badges_html += badge(f"平均勝ち ¥{m['avg_win']:+,.0f}", "dot-green")
    badges_html += badge(f"平均負け ¥{m['avg_loss']:+,.0f}", "dot-red")
    um_shares = sum(item["remaining"] for item in unmatched) if unmatched else 0
    badges_html += badge(f"未決済 {um_shares}株 / 保有 {m['avg_hold_min']:.1f}分", "dot-gray")
    badges_html += "</div>"
    st.markdown(badges_html, unsafe_allow_html=True)

    # --- KPI Row 1 ---
    c1, c2, c3, c4 = st.columns(4)
    pnl_color = GREEN if m["total_pnl"] >= 0 else RED
    with c1:
        st.markdown(
            metric_card(
                "総損益",
                f"¥{m['total_pnl']:+,.0f}",
                f"平均 ¥{m['avg_pnl']:+,.0f}/トレード",
                pnl_color,
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card(
                "勝率",
                f"{m['win_rate']:.1f}%",
                f"{m['n_win']}勝 {m['n_loss']}敗 {m['n_trades'] - m['n_win'] - m['n_loss']}分",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card(
                "ペイオフレシオ",
                f"{m['payoff_ratio']:.2f}",
                "平均利益÷平均損失",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            metric_card(
                "トレード数",
                f"{m['n_trades']}",
                f"未決済{um_shares}株 / 保有{m['avg_hold_min']:.1f}分",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # --- KPI Row 2: Donut + PF + DD + Streaks ---
    c1, c2, c3, c4 = st.columns([1.2, 1, 1.3, 1])
    with c1:
        donut_fig = chart_donut(m["n_win"], m["n_loss"])
        st.plotly_chart(donut_fig, use_container_width=False, config={"displayModeBar": False})
        st.markdown(
            f'<div class="donut-center" style="margin-top:-20px;">'
            f'<span style="font-size:22px;font-weight:700;color:#fff">{m["win_rate"]:.1f}%</span><br>'
            f'<span style="color:#888;font-size:12px">勝率</span><br>'
            f'<span style="color:{GREEN};font-size:12px">■ {m["n_win"]}勝</span> '
            f'<span style="color:{RED};font-size:12px">■ {m["n_loss"]}敗</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        pf_color = GREEN if m["profit_factor"] >= 1 else RED
        st.markdown(
            metric_card(
                "プロフィットファクター",
                f"{m['profit_factor']:.2f}",
                f'<span style="color:{GREEN}">利益 ¥{m["gross_profit"]:,.0f}</span><br>'
                f'<span style="color:{RED}">損失 ¥{m["gross_loss"]:,.0f}</span>',
                pf_color,
            ),
            unsafe_allow_html=True,
        )
    with c3:
        dd_text = f"¥{m['max_drawdown']:,.0f}"
        dd_sub = ""
        if m["dd_start"] is not None:
            dd_sub = f"{m['dd_start'].strftime('%H:%M')} → {m['dd_end'].strftime('%H:%M')}"
        st.markdown(
            metric_card("最大ドローダウン", dd_text, dd_sub, RED),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            metric_card(
                "連勝 / 連敗",
                f'<span style="color:{GREEN}">{m["max_win_streak"]}</span>'
                f'&nbsp;&nbsp;<span style="color:{RED}">{m["max_loss_streak"]}</span>',
                "最大連勝　最大連敗",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # --- KPI Row 3: PnL rates ---
    c1, c2, c3 = st.columns(3)
    rate_color = GREEN if m["pnl_rate"] >= 0 else RED
    with c1:
        st.markdown(
            metric_card(
                "損益率",
                f"{m['pnl_rate']:+.2f}%",
                f"IN ¥{m['total_in']:,.0f} / OUT ¥{m['total_out']:,.0f}",
                rate_color,
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card("最大利益率", f"{m['max_pnl_rate']:+.1f}%", "1トレード最大", GREEN),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card("最大損失率", f"{m['min_pnl_rate']:+.1f}%", "1トレード最大", RED),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- Candlestick Charts ---
    candle_1m = None
    if ticker_code:
        trade_date = trades_df["entry_time"].iloc[0].normalize()

        # 選択足種チャート（5分足 or 1分足 — アノテーション付き）
        candle_data = fetch_candles(ticker_code, trade_date, interval=candle_interval)
        if candle_data is not None and not candle_data.empty:
            interval_label = "1分足" if candle_interval == "1m" else "5分足"
            fig_candle = chart_candlestick(candle_data, trades_df, interval_label)
            st.plotly_chart(fig_candle, use_container_width=True, config=CHART_CONFIG)
        else:
            st.warning(
                f"ローソク足データを取得できませんでした（{ticker_code}.T / {candle_interval}）。"
                "yfinanceの1分足は直近7日分のみ取得可能です。"
            )

        # 1分足全場チャート（IN/OUTのみ、横長）
        candle_1m = fetch_candles(ticker_code, trade_date, interval="1m")
        if candle_1m is not None and not candle_1m.empty:
            fig_1m = chart_candlestick_1m_full(candle_1m, trades_df)
            st.plotly_chart(fig_1m, use_container_width=True, config=CHART_CONFIG)

    st.markdown("---")

    # --- Analytics Charts ---
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(chart_pnl_histogram(trades_df), use_container_width=True)
    with col_right:
        st.plotly_chart(chart_time_performance(trades_df), use_container_width=True)

    col_left2, col_right2 = st.columns(2)
    with col_left2:
        st.plotly_chart(chart_holding_scatter(trades_df), use_container_width=True)
    with col_right2:
        st.plotly_chart(chart_cumulative_pnl(trades_df), use_container_width=True)

    st.markdown("---")

    # --- Trade Detail Table ---
    st.subheader("トレード一覧")
    display_df = trades_df[
        ["entry_time", "exit_time", "side", "shares", "entry_price", "exit_price", "pnl", "pnl_rate", "hold_min"]
    ].copy()
    display_df.columns = ["エントリー", "エグジット", "売買", "株数", "IN単価", "OUT単価", "損益", "損益率", "保有(分)"]
    display_df["売買"] = display_df["売買"].map({"short": "空売", "long": "買い"})
    display_df["損益率"] = display_df["損益率"].apply(lambda x: f"{x*100:+.2f}%")

    st.dataframe(
        display_df.style.map(
            lambda v: f"color: {GREEN}" if isinstance(v, (int, float)) and v > 0 else (f"color: {RED}" if isinstance(v, (int, float)) and v < 0 else ""),
            subset=["損益"],
        ).format({"損益": "¥{:+,.0f}", "IN単価": "¥{:,.1f}", "OUT単価": "¥{:,.1f}"}),
        use_container_width=True,
        height=min(len(display_df) * 38 + 40, 600),
    )

    # --- Save to Supabase ---
    if supabase_available():
        st.markdown("---")
        col_save, col_msg = st.columns([1, 3])
        with col_save:
            if st.button("💾 この結果を保存", type="primary", use_container_width=True):
                td = trades_df["entry_time"].iloc[0].date()
                ok = save_session(td, ticker_code or code, stock_name or title, m, trades_df, candle_1m)
                if ok:
                    st.success("✅ 保存しました！「過去データ」モードで確認できます。")
                else:
                    st.error("保存に失敗しました。Supabase設定を確認してください。")

    # --- Unmatched trades ---
    if unmatched:
        st.markdown("---")
        total_unmatched = sum(item["remaining"] for item in unmatched)
        st.subheader(f"未決済ポジション ({total_unmatched}株 / {len(unmatched)}件)")
        um_data = []
        for item in unmatched:
            row = item["row"]
            um_data.append({
                "区分": row["区分"],
                "約定日時": row["dt"],
                "約定単価": row["約定単価"],
                "元株数": row["注文株数"],
                "残株数": item["remaining"],
            })
        st.dataframe(pd.DataFrame(um_data), use_container_width=True)


# ============================================================
# Pattern Analysis: decline → sideways → reversal vs continuation
# ============================================================
def _calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _calc_vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()


def detect_patterns(df, decline_pct=1.5, sideways_bars=5, sideways_range_pct=0.8, outcome_bars=10, outcome_pct=1.0):
    """
    下落→ヨコヨコ→リバ/続落のパターンを検出する。
    decline_pct: 高値からの下落率(%)で「下落」と見なす閾値
    sideways_bars: ヨコヨコと見なす最小バー数
    sideways_range_pct: ヨコヨコ期間の価格レンジ上限(%)
    outcome_bars: ヨコヨコ終了後に見るバー数
    outcome_pct: リバ/続落の判定閾値(%)
    """
    patterns = []
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values
    n = len(df)

    i = 20
    while i < n - outcome_bars - sideways_bars:
        # 直近20バーの高値
        peak_idx = np.argmax(highs[max(0, i-20):i]) + max(0, i-20)
        peak_price = highs[peak_idx]
        current_price = closes[i]

        if peak_price == 0:
            i += 1
            continue

        drop_pct = (peak_price - current_price) / peak_price * 100

        if drop_pct < decline_pct:
            i += 1
            continue

        # ヨコヨコ検出: i から sideways_bars 以上続く低レンジ期間を探す
        sideways_end = None
        for sw_len in range(sideways_bars, min(30, n - i - outcome_bars)):
            sw_slice = df.iloc[i:i+sw_len]
            sw_high = sw_slice["High"].max()
            sw_low = sw_slice["Low"].min()
            if sw_low == 0:
                continue
            sw_range_pct = (sw_high - sw_low) / sw_low * 100
            if sw_range_pct <= sideways_range_pct:
                sideways_end = i + sw_len
            else:
                break

        if sideways_end is None or sideways_end >= n - outcome_bars:
            i += 1
            continue

        # アウトカム判定
        sw_close = closes[sideways_end - 1]
        future = closes[sideways_end:min(sideways_end + outcome_bars, n)]
        if len(future) == 0:
            i += 1
            continue

        max_up = (max(future) - sw_close) / sw_close * 100
        max_down = (sw_close - min(future)) / sw_close * 100

        if max_up >= outcome_pct and max_up > max_down:
            outcome = "reversal"
        elif max_down >= outcome_pct and max_down > max_up:
            outcome = "continuation"
        else:
            i = sideways_end + 1
            continue

        # ヨコヨコ期間のボリューム傾向
        sw_vols = volumes[i:sideways_end]
        vol_trend = "増加" if len(sw_vols) > 2 and sw_vols[-1] > sw_vols[0] else "減少"

        patterns.append({
            "start_idx": i,
            "peak_idx": peak_idx,
            "sideways_start": i,
            "sideways_end": sideways_end,
            "outcome": outcome,
            "drop_pct": drop_pct,
            "sw_range_pct": (df["High"].iloc[i:sideways_end].max() - df["Low"].iloc[i:sideways_end].min()) / df["Low"].iloc[i:sideways_end].min() * 100,
            "vol_trend": vol_trend,
            "max_up_pct": max_up,
            "max_down_pct": max_down,
        })

        i = sideways_end + outcome_bars

    return patterns


def render_pattern_analysis():
    st.markdown("## 📈 パターン分析 — 下落→ヨコヨコ後のリバ vs 続落")

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        ticker_input = st.text_input("銘柄コード (Yahoo Finance)", value="6600.T", placeholder="例: 6600.T")
    with col_cfg2:
        analysis_date = st.date_input("分析日", value=datetime.now().date())
    with col_cfg3:
        interval = st.radio("足種", ["1m", "5m"], horizontal=True, format_func=lambda x: "1分足" if x == "1m" else "5分足")

    with st.expander("検出パラメータ", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        decline_pct = c1.slider("下落判定(%)", 0.5, 5.0, 1.5, 0.1)
        sideways_bars = c2.slider("ヨコヨコ最小バー数", 3, 20, 5)
        sideways_range = c3.slider("ヨコヨコ幅上限(%)", 0.2, 2.0, 0.8, 0.1)
        outcome_bars = c4.slider("判定バー数", 5, 30, 10)

    if not st.button("🔍 分析開始", type="primary"):
        st.info("銘柄コードと日付を選択して「分析開始」を押してください。\n\nキオクシアは `6600.T`、1分足で当日を選択するのがおすすめです。")
        _show_pattern_guide()
        return

    # データ取得
    start_dt = datetime.combine(analysis_date, datetime.min.time())
    end_dt = start_dt + timedelta(days=1)

    with st.spinner("データ取得中..."):
        try:
            tk = yf.Ticker(ticker_input)
            df = tk.history(start=start_dt, end=end_dt, interval=interval)
        except Exception as e:
            st.error(f"データ取得エラー: {e}")
            return

    if df is None or df.empty:
        st.warning("データが取得できませんでした。銘柄コードや日付を確認してください。")
        return

    # タイムゾーン正規化
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Tokyo")
    df = df.between_time("09:00", "15:30")

    if df.empty:
        st.warning("取引時間内のデータがありません。")
        return

    df["RSI"] = _calc_rsi(df["Close"])
    df["VWAP"] = _calc_vwap(df)
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA25"] = df["Close"].rolling(25).mean()

    # パターン検出
    patterns = detect_patterns(df, decline_pct=decline_pct, sideways_bars=sideways_bars,
                               sideways_range_pct=sideways_range, outcome_bars=outcome_bars)

    reversals = [p for p in patterns if p["outcome"] == "reversal"]
    continuations = [p for p in patterns if p["outcome"] == "continuation"]

    st.markdown(f"**検出結果**: リバ {len(reversals)}件 / 続落 {len(continuations)}件")

    if not patterns:
        st.info("パターンが検出されませんでした。パラメータを緩めてみてください。")
        _show_full_chart(df, [], ticker_input)
        _show_pattern_guide()
        return

    # 全体チャート（パターン位置をハイライト）
    _show_full_chart(df, patterns, ticker_input)

    # 統計比較
    _show_pattern_stats(reversals, continuations)

    # 個別パターン詳細
    st.markdown("### 個別パターン詳細")
    tab_labels = [f"{'🔼 リバ' if p['outcome']=='reversal' else '🔽 続落'} #{i+1}" for i, p in enumerate(patterns)]
    if tab_labels:
        tabs = st.tabs(tab_labels)
        for tab, pattern in zip(tabs, patterns):
            with tab:
                _show_pattern_detail(df, pattern, ticker_input)

    _show_pattern_guide()


def _show_full_chart(df, patterns, ticker):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03)

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        name="価格", showlegend=False,
    ), row=1, col=1)

    # VWAP
    fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
                             line=dict(color="#ff9800", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA5"], name="MA5",
                             line=dict(color="#64b5f6", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA25"], name="MA25",
                             line=dict(color="#ce93d8", width=1)), row=1, col=1)

    # パターン区間をハイライト
    for p in patterns:
        color = "rgba(76,175,80,0.12)" if p["outcome"] == "reversal" else "rgba(239,83,80,0.12)"
        border = GREEN if p["outcome"] == "reversal" else RED
        label = "🔼 リバ" if p["outcome"] == "reversal" else "🔽 続落"
        x0 = df.index[p["sideways_start"]]
        x1 = df.index[min(p["sideways_end"], len(df)-1)]
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, line_color=border,
                      line_width=1, opacity=1, row=1, col=1)
        fig.add_annotation(x=x0, y=df["High"].iloc[p["sideways_start"]:p["sideways_end"]+1].max(),
                           text=label, showarrow=False, font=dict(color=border, size=11), row=1, col=1)

    # 出来高
    colors = [GREEN if c >= o else RED for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors,
                         name="出来高", showlegend=False), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                             line=dict(color="#ffd54f", width=1.5)), row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(76,175,80,0.5)", row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,83,80,0.5)", row=3, col=1)

    fig.update_layout(**CHART_LAYOUT, height=600, title=f"{ticker} 全体チャート",
                      xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)


def _show_pattern_detail(df, pattern, ticker):
    sw_start = pattern["sideways_start"]
    sw_end = pattern["sideways_end"]
    outcome = pattern["outcome"]

    # 表示範囲: ピーク-10バー ～ ヨコヨコ終了+15バー
    view_start = max(0, pattern["peak_idx"] - 10)
    view_end = min(len(df), sw_end + 15)
    sub = df.iloc[view_start:view_end]

    outcome_label = "🔼 リバ（反発）" if outcome == "reversal" else "🔽 続落"
    color = GREEN if outcome == "reversal" else RED

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("判定", outcome_label)
    col_m2.metric("下落幅", f"{pattern['drop_pct']:.1f}%")
    col_m3.metric("ヨコヨコ幅", f"{pattern['sw_range_pct']:.2f}%")
    col_m4.metric("ヨコヨコ中の出来高", pattern["vol_trend"])

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.2],
                        vertical_spacing=0.04)

    fig.add_trace(go.Candlestick(
        x=sub.index, open=sub["Open"], high=sub["High"],
        low=sub["Low"], close=sub["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        name="価格", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub.index, y=sub["VWAP"], name="VWAP",
                             line=dict(color="#ff9800", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sub.index, y=sub["MA5"], name="MA5",
                             line=dict(color="#64b5f6", width=1)), row=1, col=1)

    # ヨコヨコ区間ハイライト
    x0 = df.index[sw_start]
    x1 = df.index[min(sw_end, len(df)-1)]
    fig.add_vrect(x0=x0, x1=x1,
                  fillcolor="rgba(76,175,80,0.1)" if outcome == "reversal" else "rgba(239,83,80,0.1)",
                  line_color=color, line_width=1.5, opacity=1, annotation_text="ヨコヨコ", row=1, col=1)

    # ピーク注釈
    if view_start <= pattern["peak_idx"] < view_end:
        peak_x = df.index[pattern["peak_idx"]]
        fig.add_vline(x=peak_x, line_dash="dash", line_color="#888", row=1, col=1)
        fig.add_annotation(x=peak_x, y=df["High"].iloc[pattern["peak_idx"]],
                           text="天井", showarrow=True, arrowcolor="#888",
                           font=dict(color="#888", size=11), row=1, col=1)

    # 出来高
    colors = [GREEN if c >= o else RED for c, o in zip(sub["Close"], sub["Open"])]
    fig.add_trace(go.Bar(x=sub.index, y=sub["Volume"], marker_color=colors,
                         name="出来高", showlegend=False), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=sub.index, y=sub["RSI"], name="RSI",
                             line=dict(color="#ffd54f", width=1.5)), row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(76,175,80,0.5)", row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,83,80,0.5)", row=3, col=1)

    fig.update_layout(**CHART_LAYOUT, height=500,
                      title=f"{'リバパターン' if outcome=='reversal' else '続落パターン'} — 下落{pattern['drop_pct']:.1f}%",
                      xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)


def _show_pattern_stats(reversals, continuations):
    if not reversals and not continuations:
        return

    st.markdown("### リバ vs 続落 — 統計比較")
    col_r, col_c = st.columns(2)

    def stat_block(patterns, label, color):
        if not patterns:
            st.info(f"{label}のパターンなし")
            return
        drop_avg = np.mean([p["drop_pct"] for p in patterns])
        sw_range_avg = np.mean([p["sw_range_pct"] for p in patterns])
        vol_inc = sum(1 for p in patterns if p["vol_trend"] == "増加")
        move_avg = np.mean([p["max_up_pct"] if p["outcome"] == "reversal" else p["max_down_pct"] for p in patterns])

        st.markdown(f"""
<div class="mc">
  <div class="label">{label}</div>
  <div class="value" style="color:{color}; font-size:22px;">{len(patterns)}件</div>
  <div style="margin-top:10px; font-size:13px; color:#ccc;">
    📉 平均下落幅: <b>{drop_avg:.1f}%</b><br>
    ↔ ヨコヨコ幅: <b>{sw_range_avg:.2f}%</b><br>
    📊 ヨコヨコ中に出来高増加: <b>{vol_inc}/{len(patterns)}件</b><br>
    🎯 平均値動き: <b>{move_avg:.1f}%</b>
  </div>
</div>""", unsafe_allow_html=True)

    with col_r:
        stat_block(reversals, "🔼 リバ（反発）", GREEN)
    with col_c:
        stat_block(continuations, "🔽 続落", RED)



def _show_pattern_guide():
    st.markdown("---")
    st.markdown("### 📖 リバ vs 続落の見分け方")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
<div class="mc">
  <div class="label">🔼 リバサイン</div>
  <div style="margin-top:10px; font-size:13px; color:#ccc; line-height:1.8;">
    ✅ <b>RSI 30以下</b>から反転上昇<br>
    ✅ ヨコヨコ中に<b>出来高が徐々に増加</b><br>
    ✅ 下ヒゲ多め（<b>ハンマー足</b>出現）<br>
    ✅ <b>VWAPが近くで支持</b>として機能<br>
    ✅ ヨコヨコ中の<b>安値が切り上がっている</b><br>
    ✅ 時間帯: 前場10:00前後や後場寄り<br>
    ✅ 板に<b>買い圧力が入りはじめる</b>
  </div>
</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
<div class="mc">
  <div class="label">🔽 続落サイン</div>
  <div style="margin-top:10px; font-size:13px; color:#ccc; line-height:1.8;">
    ❌ RSIが30〜40で<b>戻り切れない</b><br>
    ❌ ヨコヨコ中に<b>出来高が枯れている</b><br>
    ❌ <b>VWAPを一度も回復できない</b><br>
    ❌ <b>高値が切り下がって</b>いる（デッドキャット）<br>
    ❌ 上ヒゲ多め（<b>上値を抑えられている</b>）<br>
    ❌ 時間帯: 後場14:00以降の軟調<br>
    ❌ 板に<b>戻り売りが並ぶ</b>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
---
**💡 ポイント**: キオクシアのような値動きの大きい銘柄では、ヨコヨコ中の **出来高の質（買い方 vs 売り方）** と **VWAPとの位置関係** が最も重要な判断材料になります。
単純な出来高増加だけでなく、**どちらの方向に動いた後に出来高が出るか**（上に動いたときに多い → リバ方向）を確認するのが効果的です。
""")


if __name__ == "__main__":
    main()
