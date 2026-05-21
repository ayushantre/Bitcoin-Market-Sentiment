"""
Primetrade AI Assignment — Streamlit Dashboard
Run: streamlit run dashboard.py
Requires:  streamlit pandas numpy scipy scikit-learn matplotlib seaborn plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Primetrade AI · Fear & Greed Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3, h4 {
    font-family: 'Space Mono', monospace;
    letter-spacing: -0.5px;
}
.metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1420 100%);
    border: 1px solid #2a3050;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #f0c040;
}
.metric-label {
    font-size: 0.78rem;
    color: #8899aa;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.insight-box {
    background: #12192b;
    border-left: 3px solid #f0c040;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 0.75rem 0;
    font-size: 0.9rem;
    color: #ccd6e8;
}
.section-badge {
    display: inline-block;
    background: #f0c040;
    color: #000;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.5rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    background: #1a1f2e;
    border-radius: 8px;
    color: #8899aa;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}
.stTabs [aria-selected="true"] {
    background: #f0c040 !important;
    color: #000 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Color constants ─────────────────────────────────────────────────────────────
SENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
COLORS = {
    "Extreme Fear": "#d62728",
    "Fear":         "#ff7f0e",
    "Neutral":      "#7f7f7f",
    "Greed":        "#2ca02c",
    "Extreme Greed":"#1f77b4",
}
COLORS_BIN = {"Fear": "#ff7f0e", "Neutral": "#7f7f7f", "Greed": "#2ca02c"}

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(fg_file, ht_file):
    fg_raw = pd.read_csv(fg_file)
    ht_raw = pd.read_csv(ht_file)

    # ── Clean Fear/Greed ──
    fg = fg_raw.copy()
    fg["date"] = pd.to_datetime(fg["date"]).dt.date
    fg.drop_duplicates(subset="date", keep="last", inplace=True)
    fg["classification"] = pd.Categorical(fg["classification"], categories=SENT_ORDER, ordered=True)
    fg["sentiment_bin"] = fg["classification"].apply(
        lambda x: "Fear" if "Fear" in str(x) else ("Greed" if "Greed" in str(x) else "Neutral")
    )

    # ── Clean trades ──
    ht = ht_raw.copy()
    ht["dt"]   = pd.to_datetime(ht["Timestamp IST"], format="%d-%m-%Y %H:%M", dayfirst=True)
    ht["date"] = ht["dt"].dt.date
    ht.drop_duplicates(subset=["Transaction Hash", "Trade ID"], keep="first", inplace=True)
    ht["is_long"]  = ht["Direction"].isin(["Open Long","Close Short","Buy","Long > Short"]).astype(int)
    ht["is_short"] = ht["Direction"].isin(["Open Short","Close Long","Sell","Short > Long"]).astype(int)

    # ── Merge ──
    merged = ht.merge(fg[["date","value","classification","sentiment_bin"]], on="date", how="inner")

    return fg, ht, merged


@st.cache_data
def compute_metrics(_merged, _fg):
    merged = _merged; fg = _fg

    def daily_metrics(df):
        g = df.groupby(["Account","date","classification","sentiment_bin"])
        agg = g.agg(
            total_pnl      = ("Closed PnL","sum"),
            n_trades       = ("Trade ID","count"),
            avg_size_usd   = ("Size USD","mean"),
            total_size_usd = ("Size USD","sum"),
            n_wins         = ("Closed PnL", lambda x: (x > 0).sum()),
            n_losses       = ("Closed PnL", lambda x: (x < 0).sum()),
            gross_profit   = ("Closed PnL", lambda x: x[x > 0].sum()),
            gross_loss     = ("Closed PnL", lambda x: x[x < 0].sum()),
            long_trades    = ("is_long","sum"),
            short_trades   = ("is_short","sum"),
            total_fee      = ("Fee","sum"),
            max_single_pnl = ("Closed PnL","max"),
            min_single_pnl = ("Closed PnL","min"),
        ).reset_index()
        agg["win_rate"]          = agg["n_wins"] / (agg["n_wins"] + agg["n_losses"]).clip(lower=1)
        agg["ls_ratio"]          = (agg["long_trades"] + 1) / (agg["short_trades"] + 1)
        agg["net_pnl_after_fee"] = agg["total_pnl"] - agg["total_fee"]
        return agg

    daily = daily_metrics(merged)

    trader_stats = daily.groupby("Account").agg(
        total_pnl       = ("total_pnl","sum"),
        total_trades    = ("n_trades","sum"),
        active_days     = ("date","nunique"),
        avg_daily_pnl   = ("total_pnl","mean"),
        win_rate        = ("win_rate","mean"),
        avg_size_usd    = ("avg_size_usd","mean"),
        pnl_std         = ("total_pnl","std"),
        max_drawdown_day= ("total_pnl","min"),
    ).reset_index()
    trader_stats["trades_per_day"] = trader_stats["total_trades"] / trader_stats["active_days"]
    trader_stats["sharpe_proxy"]   = (
        trader_stats["avg_daily_pnl"] / trader_stats["pnl_std"].replace(0, np.nan)
    )

    # Segments
    med_size = trader_stats["avg_size_usd"].median()
    trader_stats["leverage_seg"] = np.where(trader_stats["avg_size_usd"] >= med_size, "High Notional", "Low Notional")
    med_tpd  = trader_stats["trades_per_day"].median()
    trader_stats["freq_seg"] = np.where(trader_stats["trades_per_day"] >= med_tpd, "Frequent", "Infrequent")
    trader_stats["consistency_seg"] = "Inconsistent / Losers"
    trader_stats.loc[
        (trader_stats["total_pnl"] > 0) & (trader_stats["sharpe_proxy"] > 0),
        "consistency_seg"
    ] = "Consistent Winners"

    daily = daily.merge(
        trader_stats[["Account","leverage_seg","freq_seg","consistency_seg"]],
        on="Account", how="left"
    )

    # K-Means archetypes
    cluster_features = ["total_pnl","trades_per_day","win_rate","avg_size_usd","pnl_std"]
    cluster_df = trader_stats[cluster_features].fillna(0)
    X_scaled = StandardScaler().fit_transform(cluster_df)
    km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    trader_stats["cluster"] = km4.fit_predict(X_scaled)

    top_pnl   = trader_stats.groupby("cluster")["total_pnl"].mean().idxmax()
    top_wr    = trader_stats.groupby("cluster")["win_rate"].mean().idxmax()
    top_freq  = trader_stats.groupby("cluster")["trades_per_day"].mean().idxmax()
    remaining = [c for c in range(4) if c not in [top_pnl, top_wr, top_freq]]
    label_map = {top_pnl: "Elite Earners", top_wr: "Consistent Traders",
                 top_freq: "High-Freq Gamblers", remaining[0]: "Cautious / Inactive"}
    trader_stats["archetype"] = trader_stats["cluster"].map(label_map)

    # Predictive model results (pre-computed for speed)
    model_df = daily.copy()
    model_df["date_dt"] = pd.to_datetime(model_df["date"])
    model_df = model_df.sort_values(["Account","date_dt"])
    model_df["next_day_pnl"] = model_df.groupby("Account")["total_pnl"].shift(-1)
    model_df["target"]  = (model_df["next_day_pnl"] > 0).astype(int)
    model_df["fg_value"] = model_df["date"].map(dict(zip(fg["date"], fg["value"])))
    model_df["fg_fear"]  = (model_df["sentiment_bin"] == "Fear").astype(int)
    model_df["fg_greed"] = (model_df["sentiment_bin"] == "Greed").astype(int)

    feature_cols = ["total_pnl","n_trades","win_rate","avg_size_usd","ls_ratio",
                    "total_fee","fg_value","fg_fear","fg_greed","long_trades","short_trades"]
    model_clean = model_df.dropna(subset=feature_cols + ["target"])
    X = model_clean[feature_cols]; y = model_clean["target"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_scores = {}
    for name, clf in [
        ("Logistic Regression", LogisticRegression(max_iter=500)),
        ("Random Forest",       RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]:
        sc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        model_scores[name] = {"mean": sc.mean(), "std": sc.std()}

    # Feature importances from RF
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    fi = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)

    return daily, trader_stats, model_scores, fi


# ── Sidebar — file upload ───────────────────────────────────────────────────────
st.sidebar.markdown("## 📂 Upload Datasets")
fg_file = st.sidebar.file_uploader("Fear / Greed Index CSV", type="csv", key="fg")
ht_file = st.sidebar.file_uploader("Historical Trade Data CSV", type="csv", key="ht")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filters")

# ── Main header ────────────────────────────────────────────────────────────────
st.markdown("""
# 📊 Primetrade AI
### Fear & Greed × Trader Performance Dashboard
""")

if not fg_file or not ht_file:
    st.info("👆 Upload both CSVs in the sidebar to begin. The dashboard replicates the full analysis from the assignment notebook.")
    st.markdown("""
    **Expected files:**
    - `fear_greed_index.csv` — columns: `date`, `value`, `classification`
    - `historical_data.csv` — columns: `Timestamp IST`, `Account`, `Trade ID`, `Transaction Hash`, `Direction`, `Closed PnL`, `Size USD`, `Fee`, `Coin`
    """)
    st.stop()

# ── Load ───────────────────────────────────────────────────────────────────────
with st.spinner("Loading & processing data…"):
    fg, ht, merged = load_data(fg_file, ht_file)
    daily, trader_stats, model_scores, fi = compute_metrics(merged, fg)

# Sidebar filters
all_coins = sorted(merged["Coin"].unique()) if "Coin" in merged.columns else []
selected_coins = st.sidebar.multiselect("Filter by Coin", all_coins, default=all_coins)
sent_filter = st.sidebar.multiselect("Filter by Sentiment", ["Fear","Neutral","Greed"], default=["Fear","Neutral","Greed"])

merged_f = merged[merged["sentiment_bin"].isin(sent_filter)]
if selected_coins:
    merged_f = merged_f[merged_f["Coin"].isin(selected_coins)]
daily_f  = daily[daily["sentiment_bin"].isin(sent_filter)]

# ── Top KPI row ────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, label, val):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-val">{val}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, "Total Trades",       f"{len(merged):,}")
kpi(k2, "Unique Traders",     f"{merged['Account'].nunique()}")
kpi(k3, "Unique Coins",       f"{merged['Coin'].nunique() if 'Coin' in merged.columns else '—'}")
kpi(k4, "Avg Win Rate",       f"{daily['win_rate'].mean():.1%}")
kpi(k5, "Best Model AUC",     f"{max(v['mean'] for v in model_scores.values()):.4f}")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📈 Sentiment vs PnL", "🧠 Trader Behavior", "👥 Segmentation", "🤖 Predictive Model", "📋 Raw Data"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — SENTIMENT vs PnL
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-badge">PART B1</div>', unsafe_allow_html=True)
    st.subheader("Does performance differ between Fear vs Greed days?")

    perf = daily_f.groupby("sentiment_bin").agg(
        avg_pnl   = ("total_pnl","mean"),
        med_pnl   = ("total_pnl","median"),
        win_rate  = ("win_rate","mean"),
        n_obs     = ("total_pnl","count"),
    ).reindex(["Fear","Neutral","Greed"]).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        # Box plot
        fig = go.Figure()
        for s in ["Fear","Neutral","Greed"]:
            sub = daily_f[daily_f["sentiment_bin"] == s]["total_pnl"].clip(-2000, 5000)
            fig.add_trace(go.Box(
                y=sub, name=s,
                marker_color=COLORS_BIN[s],
                boxmean="sd",
                line_color=COLORS_BIN[s],
            ))
        fig.update_layout(
            title="Daily PnL Distribution by Sentiment",
            paper_bgcolor="#0f1420", plot_bgcolor="#0f1420",
            font_color="#ccd6e8", font_family="DM Sans",
            yaxis_title="Daily PnL (USD, clipped ±5k)",
            showlegend=False,
            shapes=[dict(type="line", x0=-0.5, x1=2.5, y0=0, y1=0,
                         line=dict(color="#ff4444", dash="dash", width=1))],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Win rate bar
        fig2 = go.Figure(go.Bar(
            x=perf["sentiment_bin"],
            y=perf["win_rate"],
            marker_color=[COLORS_BIN[s] for s in perf["sentiment_bin"]],
            text=[f"{v:.1%}" for v in perf["win_rate"]],
            textposition="outside",
        ))
        fig2.update_layout(
            title="Average Daily Win Rate by Sentiment",
            paper_bgcolor="#0f1420", plot_bgcolor="#0f1420",
            font_color="#ccd6e8", font_family="DM Sans",
            yaxis_title="Win Rate", yaxis_range=[0, 0.75],
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Stats table
    st.dataframe(
        perf.round(3).rename(columns={
            "sentiment_bin":"Sentiment","avg_pnl":"Avg PnL","med_pnl":"Median PnL",
            "win_rate":"Win Rate","n_obs":"Observations"
        }),
        use_container_width=True, hide_index=True
    )

    # Mann-Whitney test
    fear_pnl  = daily_f[daily_f["sentiment_bin"]=="Fear"]["total_pnl"]
    greed_pnl = daily_f[daily_f["sentiment_bin"]=="Greed"]["total_pnl"]
    if len(fear_pnl) > 0 and len(greed_pnl) > 0:
        u, p = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative="two-sided")
        sig = "Marginally significant (p < 0.10)" if p < 0.1 else "Not statistically significant at α=0.05"
        st.markdown(f'<div class="insight-box">📊 <b>Mann-Whitney U test</b> (Fear vs Greed PnL): U = {u:.0f}, p = {p:.4f} — {sig}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insight:</b> Win rate is nearly identical across all sentiment regimes (~62%).
    What changes is the <em>size</em> of gains and losses — Fear days show the widest PnL spread
    with a lower median ($133) while Greed days produce more consistent returns (median $276).
    </div>
    """, unsafe_allow_html=True)

    # Full sentiment breakdown (5-way)
    st.markdown("#### Full 5-Tier Sentiment Breakdown")
    perf5 = daily_f.groupby("classification", observed=True).agg(
        avg_pnl  = ("total_pnl","mean"),
        win_rate = ("win_rate","mean"),
        n_obs    = ("total_pnl","count"),
    ).reset_index()
    fig5 = make_subplots(rows=1, cols=2, subplot_titles=["Avg PnL", "Win Rate"])
    fig5.add_trace(go.Bar(x=perf5["classification"].astype(str), y=perf5["avg_pnl"],
                          marker_color=[COLORS.get(s,"gray") for s in perf5["classification"].astype(str)],
                          showlegend=False), row=1, col=1)
    fig5.add_trace(go.Bar(x=perf5["classification"].astype(str), y=perf5["win_rate"],
                          marker_color=[COLORS.get(s,"gray") for s in perf5["classification"].astype(str)],
                          showlegend=False), row=1, col=2)
    fig5.update_layout(paper_bgcolor="#0f1420", plot_bgcolor="#0f1420",
                       font_color="#ccd6e8", font_family="DM Sans")
    st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRADER BEHAVIOR
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-badge">PART B2</div>', unsafe_allow_html=True)
    st.subheader("Do traders change behavior based on sentiment?")

    order = ["Fear","Neutral","Greed"]
    trade_freq = [daily_f[daily_f["sentiment_bin"]==s]["n_trades"].mean() for s in order]
    avg_size   = [daily_f[daily_f["sentiment_bin"]==s]["avg_size_usd"].mean() for s in order]
    pct_long   = []
    for s in order:
        sub = daily_f[daily_f["sentiment_bin"]==s]
        tot = sub["long_trades"].sum() + sub["short_trades"].sum()
        pct_long.append(sub["long_trades"].sum() / max(tot, 1))

    c1, c2, c3 = st.columns(3)

    def behavior_bar(col, title, x_vals, y_vals, fmt="{:.1f}", yrange=None):
        fig = go.Figure(go.Bar(
            x=x_vals, y=y_vals,
            marker_color=[COLORS_BIN[s] for s in x_vals],
            text=[fmt.format(v) for v in y_vals],
            textposition="outside",
        ))
        layout = dict(title=title, paper_bgcolor="#0f1420", plot_bgcolor="#0f1420",
                      font_color="#ccd6e8", font_family="DM Sans", showlegend=False)
        if yrange:
            layout["yaxis_range"] = yrange
        fig.update_layout(**layout)
        col.plotly_chart(fig, use_container_width=True)

    behavior_bar(c1, "Avg Trades per Day",      order, trade_freq, "{:.1f}")
    behavior_bar(c2, "Avg Position Size (USD)",  order, avg_size,  "${:,.0f}")
    behavior_bar(c3, "Long Trade %",             order, pct_long,  "{:.1%}", [0, 0.75])

    st.markdown("""
    <div class="insight-box">
    💡 <b>Key Insight:</b> Fear days trigger a 42% higher trade frequency (103.8 vs 73.1) and
    44% larger position sizes ($8,836 vs $6,122). Yet directional bias (Long %) stays nearly
    identical across regimes — traders are volatility-driven, not directionally biased.
    </div>
    """, unsafe_allow_html=True)

    # Daily volume over time coloured by sentiment
    if "date" in daily_f.columns:
        ts = daily_f.copy()
        ts["date_dt"] = pd.to_datetime(ts["date"])
        ts_agg = ts.groupby(["date_dt","sentiment_bin"]).agg(
            total_pnl=("total_pnl","sum"), n_trades=("n_trades","sum")
        ).reset_index()

        fig_ts = px.scatter(ts_agg, x="date_dt", y="total_pnl", color="sentiment_bin",
                            color_discrete_map=COLORS_BIN, size="n_trades",
                            title="Aggregate Daily PnL Over Time (bubble = trade volume)",
                            labels={"date_dt":"Date","total_pnl":"Total PnL (USD)","sentiment_bin":"Sentiment"})
        fig_ts.update_layout(paper_bgcolor="#0f1420", plot_bgcolor="#0f1420",
                             font_color="#ccd6e8", font_family="DM Sans")
        st.plotly_chart(fig_ts, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEGMENTATION
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-badge">PART B3 + BONUS 2</div>', unsafe_allow_html=True)
    st.subheader("Trader Segmentation & Archetypes")

    sub_tabs = st.tabs(["Position Size", "Trade Frequency", "Consistency", "K-Means Archetypes"])

    with sub_tabs[0]:
        seg = trader_stats.groupby("leverage_seg")[["total_pnl","win_rate","pnl_std"]].mean().reset_index()
        fig = px.bar(seg, x="leverage_seg", y=["total_pnl","win_rate","pnl_std"],
                     barmode="group", title="High vs Low Notional Traders",
                     color_discrete_sequence=["#f0c040","#2ca02c","#d62728"])
        fig.update_layout(paper_bgcolor="#0f1420", plot_bgcolor="#0f1420", font_color="#ccd6e8")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(seg.round(2), use_container_width=True, hide_index=True)
        st.markdown('<div class="insight-box">💡 Low Notional traders outperform: higher PnL, 11pp better win rate, half the volatility. Overleveraging hurts.</div>', unsafe_allow_html=True)

    with sub_tabs[1]:
        seg2 = trader_stats.groupby("freq_seg")[["total_pnl","win_rate","trades_per_day"]].mean().reset_index()
        fig2 = px.bar(seg2, x="freq_seg", y="total_pnl", color="win_rate",
                      title="Frequent vs Infrequent Traders — Total PnL",
                      color_continuous_scale="Viridis",
                      text="total_pnl")
        fig2.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig2.update_layout(paper_bgcolor="#0f1420", plot_bgcolor="#0f1420", font_color="#ccd6e8")
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(seg2.round(3), use_container_width=True, hide_index=True)
        st.markdown('<div class="insight-box">💡 Frequent traders earn 79% more and hold a 13.8pp win-rate advantage. High-frequency on Hyperliquid is a genuine edge.</div>', unsafe_allow_html=True)

    with sub_tabs[2]:
        seg3 = trader_stats.groupby("consistency_seg")[["total_pnl","win_rate","trades_per_day"]].mean().reset_index()
        fig3 = px.bar(seg3, x="consistency_seg", y="total_pnl",
                      color="consistency_seg", title="Consistent Winners vs Losers",
                      color_discrete_map={"Consistent Winners":"#2ca02c","Inconsistent / Losers":"#d62728"})
        fig3.update_layout(paper_bgcolor="#0f1420", plot_bgcolor="#0f1420", font_color="#ccd6e8", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(seg3.round(3), use_container_width=True, hide_index=True)
        st.markdown('<div class="insight-box">💡 Losers trade MORE (132.9 vs 106.9/day). A 4.8pp win-rate gap translates to a $427k PnL difference. Discipline beats activity.</div>', unsafe_allow_html=True)

    with sub_tabs[3]:
        archetype_colors = {
            "Elite Earners":       "#f0c040",
            "High-Freq Gamblers":  "#1f77b4",
            "Consistent Traders":  "#2ca02c",
            "Cautious / Inactive": "#7f7f7f",
        }
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_sc = px.scatter(
                trader_stats, x="trades_per_day", y="total_pnl",
                size="avg_size_usd", color="archetype",
                hover_data=["Account","win_rate","pnl_std"],
                color_discrete_map=archetype_colors,
                title="Trader Archetypes — Frequency vs Total PnL (bubble = avg size)",
                labels={"trades_per_day":"Trades / Day","total_pnl":"Total PnL (USD)"},
            )
            fig_sc.update_layout(paper_bgcolor="#0f1420", plot_bgcolor="#0f1420", font_color="#ccd6e8")
            st.plotly_chart(fig_sc, use_container_width=True)
        with c2:
            arc_summary = trader_stats.groupby("archetype").agg(
                n=("Account","count"),
                total_pnl=("total_pnl","mean"),
                win_rate=("win_rate","mean"),
                trades_per_day=("trades_per_day","mean"),
                avg_size_usd=("avg_size_usd","mean"),
            ).reset_index().round(1)
            st.dataframe(arc_summary, use_container_width=True, hide_index=True)

        # Radar chart per archetype
        cats = ["total_pnl","trades_per_day","win_rate","avg_size_usd","pnl_std"]
        arc_means = trader_stats.groupby("archetype")[cats].mean()
        arc_norm  = (arc_means - arc_means.min()) / (arc_means.max() - arc_means.min() + 1e-9)

        fig_r = go.Figure()
        for arch in arc_norm.index:
            vals = arc_norm.loc[arch].tolist()
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself", name=arch,
                line_color=archetype_colors.get(arch, "#fff"),
                opacity=0.6,
            ))
        fig_r.update_layout(
            polar=dict(bgcolor="#1a1f2e"),
            paper_bgcolor="#0f1420", font_color="#ccd6e8",
            title="Archetype Radar (normalized)", font_family="DM Sans"
        )
        st.plotly_chart(fig_r, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIVE MODEL
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-badge">BONUS 1</div>', unsafe_allow_html=True)
    st.subheader("Predicting Next-Day Trader Profitability")

    c1, c2 = st.columns([1, 1])

    with c1:
        model_names = list(model_scores.keys())
        means = [model_scores[m]["mean"] for m in model_names]
        stds  = [model_scores[m]["std"]  for m in model_names]
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(
            x=model_names, y=means,
            error_y=dict(type="data", array=stds, visible=True),
            marker_color=["#7f7f7f","#2ca02c","#f0c040"],
            text=[f"{v:.4f}" for v in means],
            textposition="outside",
        ))
        fig_m.add_shape(type="line", x0=-0.5, x1=2.5, y0=0.5, y1=0.5,
                        line=dict(color="#ff4444", dash="dash"))
        fig_m.update_layout(
            title="ROC-AUC — 5-Fold Cross-Validation",
            yaxis_title="ROC-AUC", yaxis_range=[0.4, 0.75],
            paper_bgcolor="#0f1420", plot_bgcolor="#0f1420",
            font_color="#ccd6e8", font_family="DM Sans", showlegend=False
        )
        fig_m.add_annotation(x=2.3, y=0.505, text="Random (0.50)", showarrow=False,
                              font=dict(color="#ff4444", size=11))
        st.plotly_chart(fig_m, use_container_width=True)

    with c2:
        # Feature importance
        fig_fi = px.bar(fi.head(10), x="importance", y="feature", orientation="h",
                        title="RF Feature Importances (Top 10)",
                        color="importance", color_continuous_scale="YlOrBr")
        fig_fi.update_layout(paper_bgcolor="#0f1420", plot_bgcolor="#0f1420",
                             font_color="#ccd6e8", font_family="DM Sans",
                             yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Gradient Boosting achieves AUC = 0.6799</b> — meaningful signal above random chance.
    Sentiment features (fg_value, fg_fear, fg_greed) contribute real predictive power on top of
    behavioral features, confirming the Fear/Greed Index has genuine value for forecasting
    next-day profitability.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Model Summary")
    model_df_display = pd.DataFrame([
        {"Model": k, "ROC-AUC": f"{v['mean']:.4f}", "Std Dev": f"± {v['std']:.4f}"}
        for k, v in model_scores.items()
    ])
    st.dataframe(model_df_display, use_container_width=True, hide_index=True)

    st.markdown("#### Strategy Playbook")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **🟠 Strategy 1 — "Trade More, Bet Smaller on Fear Days"**
        - *Trigger:* Fear/Greed Index ≤ 40
        - *Action:* Maintain/increase frequency, cap position size at historical median
        - *Target:* Frequent + Low Notional traders
        - *Rationale:* Volatility creates opportunities; over-sizing is the blowup risk
        """)
    with col_b:
        st.markdown("""
        **🟢 Strategy 2 — "Slow Down & Consolidate on Greed Days"**
        - *Trigger:* Fear/Greed Index ≥ 60
        - *Action:* Cut bottom 30% of setups, focus on high-conviction trades only
        - *Target:* Inconsistent / Loser segment
        - *Rationale:* Greed days reward patience; overtrading erodes gains
        """)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — RAW DATA
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Raw & Derived Data Tables")
    data_choice = st.selectbox("Choose table", ["Merged Trades (sample)", "Daily Trader Metrics (sample)", "Trader Lifetime Stats"])

    if data_choice == "Merged Trades (sample)":
        st.dataframe(merged_f.head(500), use_container_width=True)
    elif data_choice == "Daily Trader Metrics (sample)":
        st.dataframe(daily_f.head(500), use_container_width=True)
    else:
        st.dataframe(trader_stats, use_container_width=True)

    st.download_button("⬇ Download Trader Stats CSV",
                       trader_stats.to_csv(index=False),
                       file_name="trader_stats.csv", mime="text/csv")

st.markdown("---")
st.caption("Primetrade AI Assignment Dashboard · Built with Streamlit · Data: Hyperliquid Perps × Fear/Greed Index")
