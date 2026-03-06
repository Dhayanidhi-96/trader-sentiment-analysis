# Trader Performance vs Market Sentiment Analysis
### Primetrade.ai — Data Science Intern Assignment

Analyzing how Bitcoin market sentiment (Fear/Greed Index) relates to trader behavior and performance on Hyperliquid.

---

## 📁 Project Structure

```
trader-sentiment-analysis/
│
├── data/
│   ├── fear_greed_index.csv        # Bitcoin Fear/Greed Index (2018–2025)
│   └── historical_data.csv         # Hyperliquid trader data (2023–2025)
│
├── notebooks/
│   ├── analysis.ipynb              # Main analysis (Parts A, B, C)
│   └── bonus_ml_clustering.ipynb   # Bonus: ML model + clustering
│
├── outputs/
│   └── charts/                     # All saved charts (auto-generated)
│
├── app.py                          # Streamlit interactive dashboard
├── requirements.txt                # Python dependencies
└── README.md
```

---

## ⚙️ Setup & How to Run

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd trader-sentiment-analysis
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the main analysis notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```
> Run all cells in order (Kernel → Restart & Run All)

### 5. Run the bonus ML + Clustering notebook
```bash
jupyter notebook notebooks/bonus_ml_clustering.ipynb
```

### 6. Launch the Streamlit dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## 📊 Datasets

| Dataset | Rows | Columns | Period |
|---|---|---|---|
| Bitcoin Fear/Greed Index | 2,644 | 4 | Feb 2018 – May 2025 |
| Hyperliquid Trader Data | 211,224 | 16 | May 2023 – May 2025 |

**Key columns used:**
- `Timestamp IST`, `Account`, `Side`, `Size USD`, `Closed PnL`, `Direction`, `Fee`
- `classification` (Extreme Fear / Fear / Neutral / Greed / Extreme Greed), `value` (0–100)

**Overlap after merge:** 479 common dates — 99.8% of trade dates have matching sentiment data.

---

## 🔬 Methodology

### Data Preparation
1. Parsed `Timestamp IST` to extract daily dates from trade records
2. Aggregated trade-level data to **account × day** granularity:
   - `daily_pnl` → sum of `Closed PnL` per trader per day
   - `win_rate` → fraction of trades with PnL > 0
   - `trade_count` → number of trades executed
   - `long_ratio` → fraction of BUY-side trades
   - `avg_size_usd`, `total_volume`
3. Mapped 5-class sentiment to 3 groups: **Fear / Neutral / Greed**
4. Merged on `date` using inner join → 2,340 account-day records across 32 unique traders

### Analysis Approach
- **Part B Q1:** Compared PnL, median PnL, and win rate across sentiment groups
- **Part B Q2:** Compared trade frequency, position size, and directional bias by sentiment
- **Part B Q3:** Segmented traders by frequency, net profitability, and long/short bias
- **Bonus ML:** Random Forest classifier predicting next-day profitability (67.5% accuracy)
- **Bonus Clustering:** KMeans (K=3) to identify behavioral archetypes via PCA

---

## 💡 Key Insights

### Insight 1 — Fear Inflates Average PnL, But Greed Is Better for Typical Traders
> Average PnL is highest during Fear ($5,185) due to outlier traders making large gains.
> However, **median PnL is highest during Greed ($265 vs $122 during Fear)** — the typical trader does better when the market is optimistic. Win rate is virtually flat across sentiments (~61%), meaning sentiment does not affect how often you win, only how much.

### Insight 2 — Fear Drives More Activity, Not Less ("Buy the Dip" Effect)
> Contrary to intuition, Fear days generate **37% more trades** (105 vs 77 avg) and **43% larger position sizes** ($8,530 vs $5,955 avg) compared to Greed days. Traders treat Fear as a buying opportunity — long ratio rises to 52.2% on Fear days vs 47.2% on Greed days.

### Insight 3 — Greed Exposes the Skills Gap; Net Losers Blow Up During Euphoria
> Net Winner traders average **$5,446/day during Fear and $4,737 during Greed** — consistently profitable. Net Loser traders, however, average **-$7,800/day during Greed** (vs +$2,100 during Fear). FOMO-driven behavior during market euphoria is where weak traders are punished most severely.

---

## 🎯 Strategy Recommendations

### Strategy 1 — "Scale Up During Fear — Only If You're Already a Winner"
Profitable traders (win rate > 60%) should **increase position frequency and size during Fear** (Fear/Greed index < 30):
- Increase trade count by ~30% on Fear days
- Maintain slight long bias (~52%) to capture dip-buying opportunities
- Avoid fully short positions during Extreme Fear (index < 15)
- **Rationale:** Net Winners averaged $5,446 during Fear. Net Losers only $2,100. Fear rewards skilled traders disproportionately.

### Strategy 2 — "High Frequency Is an Edge — Maintain It Regardless of Sentiment"
High-frequency traders (>80 trades/day) earn **2–3× more** than low-frequency traders across every sentiment condition:
- During Greed: maintain execution frequency, reduce individual trade size slightly
- During Fear: maintain or increase frequency + size
- During Neutral: use lower-reward environment to refine entry/exit rules
- **Rationale:** Volume and position size are the top 2 predictors of next-day profitability (Random Forest feature importance), outranking win rate.

---

## 🤖 Bonus Results

### ML Model — Random Forest Classifier
- **Task:** Predict whether a trader will be profitable the next day
- **Features:** Fear/Greed value, trade count, avg trade size, volume, long ratio, win rate, daily PnL
- **Accuracy:** 67.5% | F1 (Profitable class): 0.77
- **Top features:** Total volume, avg trade size, Fear/Greed index score — confirming sentiment carries real predictive signal

### Clustering — 3 Behavioral Archetypes
| Archetype | Win Rate | Avg Trades/Day | Avg Size | Trading Days | Total PnL |
|---|---|---|---|---|---|
| 🔴 High-Stakes Sprinters | 60% | 185 | $15,007 | 34 | $340K |
| 🔵 Struggling Dabblers | 40% | 42 | $5,170 | 45 | $155K |
| 🟢 Consistent Grinders | **73%** | 91 | $3,755 | **175** | **$514K** |

> **Key finding:** Consistent Grinders win the most total money — not by being the most active or using the biggest positions, but through high win rate and sustained participation. Longevity + discipline outperforms aggression.

---

## 🖥️ Streamlit Dashboard Features

- **Sidebar filters:** Sentiment type, individual trader, date range
- **KPI cards:** Days in view, unique traders, avg PnL, avg win rate
- **Charts:** PnL by sentiment, trade frequency by sentiment, daily PnL time series
- **Data table:** Filtered raw data view

---

## 📦 Dependencies

```
pandas==2.3.3
numpy==2.2.6
matplotlib==3.10.8
seaborn==0.13.2
scikit-learn==1.7.2
jupyter
streamlit==1.55.0
ipykernel
openpyxl
```
> Full pinned versions in `requirements.txt`
