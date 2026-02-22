# What YC Is Really Betting On?

An interactive dashboard analyzing **793 companies** across 5 Y Combinator batches (Winter 2025 – Winter 2026), with **1,625 founder bios** scraped and analyzed.

## Live Dashboard

Open `dashboard_final.html` in your browser to explore the full interactive dashboard.

## What's Inside

- **AI Wrapper vs Deep Tech breakdown** — Only 15% are thin LLM wrappers, and that number is declining
- **Non-obvious correlations** — SF companies hire less, AI is anti-correlated with Fintech, and more
- **Competitive crowding analysis** — YC funds near-identical competitors in the same batch
- **Buzzword evolution** — What's rising ("infrastructure", "autonomous") and what's dying ("compliance")
- **Founder DNA** — Background analysis of 1,625 founders (ex-FAANG, PhDs, repeat YC founders)
- **YC Partner fingerprints** — Each partner has distinct preferences for verticals, founder types, and company profiles
- **NLP clustering** — 15 hidden themes discovered via TF-IDF + K-Means on company descriptions
- **Naming patterns, hiring signals, and the statistical YC archetype**

## Data Pipeline

1. **`fetch_companies.py`** — Scrapes YC companies via Algolia API
2. **`fetch_founder_details.py`** — Enriches data with founder bios from individual company pages
3. **`deep_analysis.py`** — NLP clustering, correlations, competitive overlap, partner analysis
4. **`build_dashboard.py`** — Merges all insights and injects data into the dashboard template

## Tech Stack

- **Data**: Python, Algolia API, web scraping
- **Analysis**: TF-IDF, K-Means clustering, Pearson correlation, cosine similarity
- **Visualization**: Chart.js (27 interactive charts), vanilla HTML/CSS/JS

## Disclaimer

This is an independent analysis and is not affiliated with, endorsed by, or connected to Y Combinator. All data is sourced from publicly available information.

---

Built by [Krishna Goyal](https://www.linkedin.com/in/krishnaa-goyal/)
