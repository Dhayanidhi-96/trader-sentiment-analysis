
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Trader Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_data():
    historical_data = pd.read_csv('data/historical_data.csv')
    fear_greed      = pd.read_csv('data/fear_greed_index.csv')

    historical_data['date'] = pd.to_datetime(
        historical_data['Timestamp IST'], format='%d-%m-%Y %H:%M'
    ).dt.date
    fear_greed['date'] = pd.to_datetime(fear_greed['date']).dt.date

    historical_data['is_long'] = historical_data['Side'].str.upper() == 'BUY'
    closing = historical_data[historical_data['Closed PnL'] != 0].copy()
    closing['is_win'] = (closing['Closed PnL'] > 0).astype(int)

    daily_behavior = historical_data.groupby(['Account','date']).agg(
        trade_count  = ('Trade ID','count'),
        avg_size_usd = ('Size USD','mean'),
        total_volume = ('Size USD','sum'),
        long_ratio   = ('is_long','mean'),
    ).reset_index()

    daily_pnl = closing.groupby(['Account','date']).agg(
        daily_pnl     = ('Closed PnL','sum'),
        win_rate      = ('is_win','mean'),
        closed_trades = ('Trade ID','count'),
    ).reset_index()

    daily_stats = pd.merge(daily_behavior, daily_pnl, on=['Account','date'], how='left')
    daily_stats['daily_pnl'] = daily_stats['daily_pnl'].fillna(0)
    daily_stats['win_rate']  = daily_stats['win_rate'].fillna(0)

    sentiment_map = {
        'Extreme Fear':'Fear','Fear':'Fear',
        'Neutral':'Neutral',
        'Greed':'Greed','Extreme Greed':'Greed'
    }
    fear_greed['sentiment'] = fear_greed['classification'].map(sentiment_map)

    merged = pd.merge(daily_stats,
                      fear_greed[['date','value','classification','sentiment']],
                      on='date', how='inner')
    return merged

merged = load_data()

#header
st.title("📊 Trader Performance vs Market Sentiment")
st.markdown("**Hyperliquid trader data analyzed against Bitcoin Fear/Greed Index**")
st.divider()

#sidebar filters
st.sidebar.header("🔍 Filters")

sentiments = st.sidebar.multiselect(
    "Select Sentiment",
    options=['Fear','Neutral','Greed'],
    default=['Fear','Neutral','Greed']
)

accounts = st.sidebar.multiselect(
    "Select Trader (Account)",
    options=sorted(merged['Account'].unique()),
    default=[]
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=[merged['date'].min(), merged['date'].max()]
)

# apply filters
filtered = merged[merged['sentiment'].isin(sentiments)]
if accounts:
    filtered = filtered[filtered['Account'].isin(accounts)]
if len(date_range) == 2:
    filtered = filtered[
        (filtered['date'] >= date_range[0]) &
        (filtered['date'] <= date_range[1])
    ]


# KPI METRICS ROW

col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 Days in View",        f"{filtered['date'].nunique():,}")
col2.metric("👥 Unique Traders",      f"{filtered['Account'].nunique()}")
col3.metric("💰 Avg Daily PnL",       f"${filtered['daily_pnl'].mean():,.0f}")
col4.metric("🏆 Avg Win Rate",        f"{filtered['win_rate'].mean()*100:.1f}%")
st.divider()


# CHARTS

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Avg Daily PnL by Sentiment")
    order = ['Fear','Neutral','Greed']
    colors = ['#e74c3c','#f39c12','#2ecc71']
    pnl_data = filtered.groupby('sentiment')['daily_pnl'].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(pnl_data.index, pnl_data.values, color=colors, edgecolor='white')
    ax.set_ylabel("Avg PnL (USD)")
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    for bar, val in zip(bars, pnl_data.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                f'${val:,.0f}', ha='center', fontsize=10, fontweight='bold')
    st.pyplot(fig)
    plt.close()

with col_b:
    st.subheader("Avg Trade Count by Sentiment")
    tc_data = filtered.groupby('sentiment')['trade_count'].mean().reindex(order)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    bars2 = ax2.bar(tc_data.index, tc_data.values, color=colors, edgecolor='white')
    ax2.set_ylabel("Avg Trades/Day")
    for bar, val in zip(bars2, tc_data.values):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    st.pyplot(fig2)
    plt.close()

st.divider()

# ── PnL Over Time ──
st.subheader("📈 Daily Aggregate PnL Over Time")
daily_total = filtered.groupby('date')['daily_pnl'].sum().reset_index()
daily_total['date'] = pd.to_datetime(daily_total['date'])
fig3, ax3 = plt.subplots(figsize=(14, 4))
ax3.plot(daily_total['date'], daily_total['daily_pnl'], color='#3498db', linewidth=1.5)
ax3.fill_between(daily_total['date'], daily_total['daily_pnl'],
                 where=daily_total['daily_pnl']>=0, alpha=0.3, color='#2ecc71')
ax3.fill_between(daily_total['date'], daily_total['daily_pnl'],
                 where=daily_total['daily_pnl']<0, alpha=0.3, color='#e74c3c')
ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax3.set_ylabel("Total PnL (USD)")
ax3.set_xlabel("Date")
st.pyplot(fig3)
plt.close()

st.divider()

# ── Raw Data Table ──
st.subheader("📋 Filtered Data Table")
st.dataframe(
    filtered[['Account','date','sentiment','daily_pnl','win_rate',
              'trade_count','avg_size_usd','long_ratio']].round(2),
    use_container_width=True
)
