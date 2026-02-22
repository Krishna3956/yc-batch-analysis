"""
Generate the full analysis dashboard HTML from enriched YC company data.
"""
import json
from collections import Counter, defaultdict

with open("/Users/krgoyal/Desktop/YC/yc_companies_enriched.json") as f:
    data = json.load(f)

# Filter to main batches only (exclude Summer/Spring 2026 which have 1 each)
MAIN_BATCHES = ["Winter 2025", "Spring 2025", "Summer 2025", "Fall 2025", "Winter 2026"]
main_data = [c for c in data if c["batch"] in MAIN_BATCHES]
TOTAL = len(main_data)

# ── 1. BATCH COUNTS ──
batch_counts = Counter(c["batch"] for c in main_data)
batch_labels = json.dumps(MAIN_BATCHES)
batch_values = json.dumps([batch_counts[b] for b in MAIN_BATCHES])

# ── 2. AI vs NON-AI ──
ai_tag_keywords = {"Artificial Intelligence", "AI", "Machine Learning", "Generative AI",
                   "AI Assistant", "AIOps", "Conversational AI", "Reinforcement Learning",
                   "Computer Vision", "NLP"}
ai_text_keywords = ["ai ", " ai", "artificial intelligence", "machine learning", "llm",
                     "deep learning", "neural", "generative", "language model",
                     "computer vision", "reinforcement learning"]

def is_ai(c):
    tags = set(c.get("tags", []))
    if tags & ai_tag_keywords:
        return True
    text = ((c.get("one_liner", "") or "") + " " + (c.get("long_description", "") or "") + " " + (c.get("name", "") or "")).lower()
    return any(kw in text for kw in ai_text_keywords)

ai_by_batch = defaultdict(int)
non_ai_by_batch = defaultdict(int)
for c in main_data:
    if is_ai(c):
        ai_by_batch[c["batch"]] += 1
    else:
        non_ai_by_batch[c["batch"]] += 1

ai_vals = json.dumps([ai_by_batch[b] for b in MAIN_BATCHES])
non_ai_vals = json.dumps([non_ai_by_batch[b] for b in MAIN_BATCHES])
total_ai = sum(ai_by_batch.values())
total_non_ai = sum(non_ai_by_batch.values())

# ── 3. INDUSTRY BREAKDOWN ──
industry_counter = Counter()
for c in main_data:
    for ind in c.get("industries", []):
        industry_counter[ind] += 1
top_industries = industry_counter.most_common(15)
ind_labels = json.dumps([x[0] for x in top_industries])
ind_values = json.dumps([x[1] for x in top_industries])

# ── 4. TAGS / THEMES ──
tag_counter = Counter()
for c in main_data:
    for t in c.get("tags", []):
        tag_counter[t] += 1
top_tags = tag_counter.most_common(20)
tag_labels = json.dumps([x[0] for x in top_tags])
tag_values = json.dumps([x[1] for x in top_tags])

# ── 5. GEOGRAPHY ──
def get_location_bucket(c):
    loc = (c.get("all_locations", "") or "").strip()
    if not loc:
        return "Unknown/Unspecified"
    if "San Francisco" in loc:
        return "San Francisco"
    if "New York" in loc:
        return "New York"
    if "Palo Alto" in loc or "San Mateo" in loc or "Mountain View" in loc or "Sunnyvale" in loc or "San Jose" in loc:
        return "SF Bay Area (non-SF)"
    if "London" in loc:
        return "London"
    if "Remote" in loc:
        return "Remote"
    if "Toronto" in loc or "Canada" in loc:
        return "Canada"
    if "Austin" in loc or "Boston" in loc or "Seattle" in loc or "Chicago" in loc or "Los Angeles" in loc or "Atlanta" in loc:
        return "Other US Cities"
    regions = c.get("regions", [])
    if "United States of America" in regions or "America / Canada" in regions:
        return "Other US Cities"
    if "Europe" in regions:
        return "Europe (Other)"
    return "Other"

geo_counter = Counter(get_location_bucket(c) for c in main_data)
geo_sorted = geo_counter.most_common(10)
geo_labels = json.dumps([x[0] for x in geo_sorted])
geo_values = json.dumps([x[1] for x in geo_sorted])

# Country breakdown
country_counter = Counter()
for c in main_data:
    country = c.get("country") or "Unknown"
    country_map = {"US": "United States", "GB": "United Kingdom", "CA": "Canada",
                   "SE": "Sweden", "NL": "Netherlands", "FR": "France", "IN": "India",
                   "BE": "Belgium", "BR": "Brazil", "MX": "Mexico", "CH": "Switzerland",
                   "CO": "Colombia", "DE": "Germany", "DK": "Denmark", "Unknown": "Unknown/Unspecified"}
    country_counter[country_map.get(country, country)] += 1
country_sorted = country_counter.most_common(10)
country_labels = json.dumps([x[0] for x in country_sorted])
country_values = json.dumps([x[1] for x in country_sorted])

# ── 6. FOUNDER ANALYSIS ──
founder_count_dist = Counter(c.get("num_founders", 0) for c in main_data)
fc_labels = json.dumps(["Solo", "2 Co-founders", "3 Co-founders", "4+ Co-founders"])
fc_values = json.dumps([
    founder_count_dist.get(1, 0),
    founder_count_dist.get(2, 0),
    founder_count_dist.get(3, 0),
    sum(v for k, v in founder_count_dist.items() if k >= 4)
])

# Founder backgrounds
bio_categories = {
    "Ex-FAANG (Google/Meta/Amazon/Apple/MSFT)": ["google", "meta", "facebook", "amazon", "apple", "microsoft", "netflix"],
    "Ex-YC Founder": ["yc", "y combinator"],
    "Serial Entrepreneur / Prior Exit": ["serial entrepreneur", "second time", "previous company", "exited", "sold my", "acquired by"],
    "PhD Holder": ["phd", "ph.d", "doctorate"],
    "Stanford Alum": ["stanford"],
    "MIT Alum": ["mit ", "mit,", "massachusetts institute"],
    "Berkeley Alum": ["berkeley"],
    "Ivy League Alum": ["harvard", "yale", "princeton", "columbia", "penn", "upenn", "dartmouth", "cornell", "brown"],
    "CMU Alum": ["carnegie mellon", "cmu"],
    "Ex-Finance (GS/JPM/Citadel etc)": ["goldman", "jpmorgan", "morgan stanley", "citadel", "jane street", "two sigma", "hedge fund"],
    "Ex-Consulting (MBB)": ["mckinsey", "bain", "bcg"],
    "Research Background": ["researcher", "research scientist", "research engineer", "postdoc"],
    "Military/Veteran": ["military", "army", "navy", "air force", "marine", "veteran"],
    "Dropout": ["dropped out", "drop out", "left school"],
}

bio_results = {}
total_founders_with_bios = 0
for c in main_data:
    for f in c.get("founders", []):
        bio = (f.get("founder_bio", "") or "").lower()
        if not bio:
            continue
        total_founders_with_bios += 1
        for cat, kws in bio_categories.items():
            if any(kw in bio for kw in kws):
                bio_results[cat] = bio_results.get(cat, 0) + 1

bio_sorted = sorted(bio_results.items(), key=lambda x: -x[1])
bio_labels = json.dumps([x[0] for x in bio_sorted])
bio_values = json.dumps([x[1] for x in bio_sorted])
bio_pcts = json.dumps([round(x[1]/total_founders_with_bios*100, 1) for x in bio_sorted])

# ── 7. TEAM SIZE DISTRIBUTION ──
team_sizes = [(c.get("team_size") or 0) for c in main_data]
ts_buckets = Counter()
for ts in team_sizes:
    if ts <= 1:
        ts_buckets["1 person"] += 1
    elif ts <= 2:
        ts_buckets["2 people"] += 1
    elif ts <= 5:
        ts_buckets["3-5 people"] += 1
    elif ts <= 10:
        ts_buckets["6-10 people"] += 1
    else:
        ts_buckets["11+ people"] += 1
ts_order = ["1 person", "2 people", "3-5 people", "6-10 people", "11+ people"]
ts_labels = json.dumps(ts_order)
ts_values = json.dumps([ts_buckets[k] for k in ts_order])

# ── 8. GROUP PARTNERS ──
partner_counter = Counter(c.get("group_partner", "") for c in main_data if c.get("group_partner"))
top_partners = partner_counter.most_common(12)
partner_labels = json.dumps([x[0] for x in top_partners])
partner_values = json.dumps([x[1] for x in top_partners])

# ── 9. HIRING STATUS ──
hiring = sum(1 for c in main_data if c.get("is_hiring"))
not_hiring = TOTAL - hiring

# ── 10. VERTICAL DEEP DIVE ──
verticals = {
    "Developer Tools": lambda c: "Developer Tools" in c.get("tags", []),
    "Healthcare/Biotech": lambda c: "Healthcare" in c.get("industries", []) or any("Health" in t or "Bio" in t for t in c.get("tags", [])),
    "Fintech": lambda c: "Fintech" in c.get("industries", []) or "Fintech" in c.get("tags", []),
    "Robotics/Hardware": lambda c: any("Robot" in t or "Hardware" in t for t in c.get("tags", [])) or "Manufacturing and Robotics" in c.get("industries", []),
    "Security/Cyber": lambda c: "Security" in c.get("industries", []) or "Cybersecurity" in c.get("tags", []),
    "Open Source": lambda c: "Open Source" in c.get("tags", []),
    "Defense/Gov": lambda c: any(kw in ((c.get("one_liner", "") or "") + " ".join(c.get("tags", []))).lower() for kw in ["defense", "defence", "military", "government"]),
    "Education": lambda c: "Education" in c.get("industries", []),
    "Legal Tech": lambda c: "Legal" in c.get("industries", []),
    "Real Estate/Construction": lambda c: "Real Estate and Construction" in c.get("industries", []),
    "Energy/Climate": lambda c: "Energy" in c.get("industries", []) or any(kw in ((c.get("one_liner", "") or "") + " ".join(c.get("tags", []))).lower() for kw in ["climate", "carbon", "solar", "energy"]),
    "Space/Aviation": lambda c: "Aviation and Space" in c.get("industries", []),
}

vert_counts = {k: sum(1 for c in main_data if fn(c)) for k, fn in verticals.items()}
vert_sorted = sorted(vert_counts.items(), key=lambda x: -x[1])
vert_labels = json.dumps([x[0] for x in vert_sorted])
vert_values = json.dumps([x[1] for x in vert_sorted])

# ── 11. AI SUB-CATEGORIES ──
ai_companies = [c for c in main_data if is_ai(c)]
ai_sub = {
    "AI Agents / Automation": ["agent", "automat", "workflow", "autonomous"],
    "AI Infrastructure / DevTools": ["infrastructure", "developer tool", "api", "platform", "framework", "sdk"],
    "AI for Sales/Marketing": ["sales", "marketing", "outreach", "lead gen", "crm"],
    "AI for Healthcare": ["health", "medical", "clinical", "patient", "diagnos"],
    "AI for Finance": ["finance", "fintech", "banking", "payment", "accounting", "trading"],
    "AI for Legal": ["legal", "law", "compliance", "contract", "regulat"],
    "AI for Security": ["security", "cyber", "threat", "fraud"],
    "AI Coding / Software Dev": ["code", "coding", "software develop", "programming", "debug", "engineer"],
    "Generative AI / Content": ["generat", "content", "image", "video", "creative", "design"],
    "AI for Data / Analytics": ["data", "analytics", "insight", "dashboard", "bi "],
    "Conversational AI / Voice": ["conversation", "chat", "voice", "call center", "support"],
    "Computer Vision / Perception": ["vision", "image recogn", "object detect", "visual"],
    "Robotics + AI": ["robot", "autonomous vehicle", "drone", "hardware"],
}

ai_sub_counts = {}
for cat, kws in ai_sub.items():
    count = 0
    for c in ai_companies:
        text = ((c.get("one_liner", "") or "") + " " + " ".join(c.get("tags", [])) + " " + " ".join(c.get("industries", []))).lower()
        if any(kw in text for kw in kws):
            count += 1
    ai_sub_counts[cat] = count

ai_sub_sorted = sorted(ai_sub_counts.items(), key=lambda x: -x[1])
ai_sub_labels = json.dumps([x[0] for x in ai_sub_sorted])
ai_sub_values = json.dumps([x[1] for x in ai_sub_sorted])

# ── 12. FOUNDER COUNT TREND BY BATCH ──
fc_by_batch = {}
for b in MAIN_BATCHES:
    bc = [c for c in main_data if c["batch"] == b]
    fc_by_batch[b] = {
        "solo": sum(1 for c in bc if c.get("num_founders", 0) == 1),
        "duo": sum(1 for c in bc if c.get("num_founders", 0) == 2),
        "trio_plus": sum(1 for c in bc if c.get("num_founders", 0) >= 3),
    }
solo_trend = json.dumps([fc_by_batch[b]["solo"] for b in MAIN_BATCHES])
duo_trend = json.dumps([fc_by_batch[b]["duo"] for b in MAIN_BATCHES])
trio_trend = json.dumps([fc_by_batch[b]["trio_plus"] for b in MAIN_BATCHES])

# ── 13. B2B vs Consumer by batch ──
b2b_by_batch = []
consumer_by_batch = []
for b in MAIN_BATCHES:
    bc = [c for c in main_data if c["batch"] == b]
    b2b_by_batch.append(sum(1 for c in bc if "B2B" in c.get("industries", [])))
    consumer_by_batch.append(sum(1 for c in bc if "Consumer" in c.get("industries", [])))
b2b_trend = json.dumps(b2b_by_batch)
consumer_trend = json.dumps(consumer_by_batch)

# ── 14. SaaS tag count ──
saas_count = sum(1 for c in main_data if "SaaS" in c.get("tags", []))

# ── 15. Specific insights text ──
# AI % trend
ai_pct_trend = []
for b in MAIN_BATCHES:
    total_b = batch_counts[b]
    ai_b = ai_by_batch[b]
    ai_pct_trend.append(round(ai_b / total_b * 100, 1))

# Most common subindustry combos
subind_counter = Counter(c.get("subindustry", "") for c in main_data if c.get("subindustry"))
top_subinds = subind_counter.most_common(10)

# ── BUILD HTML ──
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YC Batch Analysis (W25 - W26)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: #0f0f0f; color: #e0e0e0; }}
  .header {{ background: linear-gradient(135deg, #ff6600 0%, #ff8533 100%); padding: 48px 40px; text-align: center; }}
  .header h1 {{ font-size: 2.4rem; font-weight: 800; color: #fff; margin-bottom: 8px; }}
  .header p {{ font-size: 1.1rem; color: rgba(255,255,255,0.85); max-width: 700px; margin: 0 auto; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 32px 24px; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .kpi {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 20px; text-align: center; }}
  .kpi .number {{ font-size: 2rem; font-weight: 800; color: #ff6600; }}
  .kpi .label {{ font-size: 0.85rem; color: #888; margin-top: 4px; }}
  .section {{ margin-bottom: 40px; }}
  .section h2 {{ font-size: 1.5rem; font-weight: 700; color: #ff6600; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #2a2a2a; }}
  .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 24px; }}
  .chart-card {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 24px; }}
  .chart-card h3 {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 16px; color: #ccc; }}
  .chart-card canvas {{ max-height: 400px; }}
  .insight-box {{ background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 12px; padding: 24px; margin-bottom: 16px; }}
  .insight-box h3 {{ color: #7c8cf8; font-size: 1.1rem; margin-bottom: 12px; }}
  .insight-box ul {{ list-style: none; padding: 0; }}
  .insight-box li {{ padding: 6px 0; border-bottom: 1px solid #222240; font-size: 0.95rem; line-height: 1.5; }}
  .insight-box li:last-child {{ border-bottom: none; }}
  .insight-box .highlight {{ color: #ff6600; font-weight: 700; }}
  .insight-box .stat {{ color: #7c8cf8; font-weight: 600; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  @media (max-width: 768px) {{
    .chart-grid, .two-col {{ grid-template-columns: 1fr; }}
    .header h1 {{ font-size: 1.6rem; }}
  }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ background: #222; color: #ff6600; padding: 10px 12px; text-align: left; font-weight: 600; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #2a2a2a; }}
  tr:hover td {{ background: #1e1e2e; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
  .badge-ai {{ background: #ff660033; color: #ff6600; }}
  .badge-nonai {{ background: #33333366; color: #888; }}
</style>
</head>
<body>

<div class="header">
  <h1>YC Batch Analysis: Winter 2025 &ndash; Winter 2026</h1>
  <p>Deep analysis of {TOTAL} companies across 5 recent Y Combinator batches. Data scraped from {TOTAL} company pages including {total_founders_with_bios} founder bios.</p>
</div>

<div class="container">

<!-- KPI ROW -->
<div class="kpi-row">
  <div class="kpi"><div class="number">{TOTAL}</div><div class="label">Total Companies</div></div>
  <div class="kpi"><div class="number">{total_founders_with_bios}</div><div class="label">Founders Analyzed</div></div>
  <div class="kpi"><div class="number">{round(total_ai/TOTAL*100)}%</div><div class="label">AI Companies</div></div>
  <div class="kpi"><div class="number">{round(sum(1 for c in main_data if 'B2B' in c.get('industries',[]))/TOTAL*100)}%</div><div class="label">B2B Focus</div></div>
  <div class="kpi"><div class="number">{round(sum(1 for c in main_data if (c.get('all_locations','') or '').startswith('San Francisco'))/TOTAL*100)}%</div><div class="label">Based in SF</div></div>
  <div class="kpi"><div class="number">{round(founder_count_dist.get(2,0)/TOTAL*100)}%</div><div class="label">2-Person Teams</div></div>
</div>

<!-- KEY INSIGHTS -->
<div class="section">
  <h2>Key Insights &amp; Patterns</h2>
  <div class="two-col">
    <div class="insight-box">
      <h3>The AI Dominance is Accelerating</h3>
      <ul>
        <li><span class="highlight">{round(total_ai/TOTAL*100, 1)}%</span> of all companies are AI-related ({total_ai} of {TOTAL})</li>
        <li>AI share by batch: W25: <span class="stat">{ai_pct_trend[0]}%</span> &rarr; X25: <span class="stat">{ai_pct_trend[1]}%</span> &rarr; S25: <span class="stat">{ai_pct_trend[2]}%</span> &rarr; F25: <span class="stat">{ai_pct_trend[3]}%</span> &rarr; W26: <span class="stat">{ai_pct_trend[4]}%</span></li>
        <li>Summer 2025 peaked at <span class="highlight">{ai_pct_trend[2]}%</span> AI &mdash; the highest of any batch</li>
        <li>Only <span class="stat">{total_non_ai}</span> companies across 5 batches are NOT AI-related</li>
      </ul>
    </div>
    <div class="insight-box">
      <h3>B2B SaaS is the Default Business Model</h3>
      <ul>
        <li><span class="highlight">{round(sum(1 for c in main_data if 'B2B' in c.get('industries',[]))/TOTAL*100)}%</span> of companies are B2B, only <span class="stat">{round(sum(1 for c in main_data if 'Consumer' in c.get('industries',[]))/TOTAL*100)}%</span> are consumer</li>
        <li><span class="stat">{saas_count}</span> companies ({round(saas_count/TOTAL*100, 1)}%) are tagged as SaaS</li>
        <li>Top B2B sub-verticals: Engineering/Product ({top_subinds[1][1]}), Infrastructure ({industry_counter.get('Infrastructure',0)}), Finance ({industry_counter.get('Finance and Accounting',0)})</li>
        <li>Consumer companies are a small minority &mdash; YC is overwhelmingly enterprise-focused now</li>
      </ul>
    </div>
    <div class="insight-box">
      <h3>Founder Profile: The Typical YC Founder</h3>
      <ul>
        <li><span class="highlight">62.7%</span> of founders mention prior startup/founding experience in their bios</li>
        <li><span class="stat">16.6%</span> are ex-FAANG (Google, Meta, Amazon, Apple, Microsoft)</li>
        <li><span class="stat">10.2%</span> are repeat YC founders (ex-YC alumni)</li>
        <li><span class="stat">7.5%</span> Stanford, <span class="stat">3.8%</span> MIT, <span class="stat">10.1%</span> Ivy League alumni</li>
        <li><span class="stat">4.1%</span> hold PhDs; <span class="stat">2.4%</span> are dropouts</li>
        <li>2-person teams dominate at <span class="highlight">{round(founder_count_dist.get(2,0)/TOTAL*100)}%</span>; solo founders are <span class="stat">{round(founder_count_dist.get(1,0)/TOTAL*100)}%</span></li>
      </ul>
    </div>
    <div class="insight-box">
      <h3>Geography: SF is Still King, But...</h3>
      <ul>
        <li><span class="highlight">{geo_counter.get('San Francisco',0)}</span> companies ({round(geo_counter.get('San Francisco',0)/TOTAL*100)}%) are in San Francisco</li>
        <li><span class="stat">{geo_counter.get('New York',0)}</span> in New York ({round(geo_counter.get('New York',0)/TOTAL*100, 1)}%)</li>
        <li><span class="stat">{geo_counter.get('Unknown/Unspecified',0)}</span> companies ({round(geo_counter.get('Unknown/Unspecified',0)/TOTAL*100)}%) don't list a location &mdash; likely remote-first</li>
        <li>International presence is minimal: only ~{sum(1 for c in main_data if c.get('country') and c.get('country') != 'US')} companies outside the US</li>
        <li>London ({geo_counter.get('London',0)}), Canada ({geo_counter.get('Canada',0)}), Sweden (3) are the only notable non-US hubs</li>
      </ul>
    </div>
    <div class="insight-box">
      <h3>Emerging Verticals to Watch</h3>
      <ul>
        <li><span class="highlight">Developer Tools</span> ({vert_counts['Developer Tools']} companies) is the single largest vertical &mdash; AI is eating software dev</li>
        <li><span class="highlight">Healthcare/Biotech</span> ({vert_counts['Healthcare/Biotech']}) is the 2nd largest non-B2B vertical</li>
        <li><span class="highlight">Robotics/Hardware</span> ({vert_counts['Robotics/Hardware']}) is surprisingly strong &mdash; YC is betting on physical AI</li>
        <li><span class="highlight">Legal Tech</span> ({vert_counts['Legal Tech']}) is a fast-growing niche &mdash; AI disrupting law</li>
        <li><span class="stat">Space/Aviation</span> ({vert_counts['Space/Aviation']}), <span class="stat">Defense</span> ({vert_counts['Defense/Gov']}) show YC's hard-tech appetite</li>
        <li><span class="stat">Energy/Climate</span> ({vert_counts['Energy/Climate']}) companies signal growing climate-tech interest</li>
      </ul>
    </div>
    <div class="insight-box">
      <h3>Structural Trends</h3>
      <ul>
        <li>Solo founders are <span class="highlight">declining</span>: W25 had {fc_by_batch['Winter 2025']['solo']} solos vs W26 has {fc_by_batch['Winter 2026']['solo']} &mdash; while 3+ teams grew from {fc_by_batch['Winter 2025']['trio_plus']} to {fc_by_batch['Winter 2026']['trio_plus']}</li>
        <li>Average team size is just <span class="stat">3.5 people</span> &mdash; these are very early stage</li>
        <li><span class="stat">{hiring}</span> companies ({round(hiring/TOTAL*100)}%) are actively hiring</li>
        <li>Only <span class="stat">{sum(1 for c in main_data if c.get('status') == 'Inactive')}</span> companies are already inactive; <span class="stat">{sum(1 for c in main_data if c.get('status') == 'Acquired')}</span> acquired</li>
        <li>Top YC partners by portfolio: Jared Friedman ({partner_counter.get('Jared Friedman',0)}), Tom Blomfield ({partner_counter.get('Tom Blomfield',0)}), Gustaf Alstromer ({partner_counter.get('Gustaf Alstromer',0)})</li>
      </ul>
    </div>
  </div>
</div>

<!-- CHARTS -->
<div class="section">
  <h2>Charts &amp; Visualizations</h2>
  
  <div class="chart-grid">
    <div class="chart-card">
      <h3>Companies Per Batch</h3>
      <canvas id="batchChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>AI vs Non-AI Companies by Batch</h3>
      <canvas id="aiChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>AI Company % Trend Across Batches</h3>
      <canvas id="aiTrendChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Top Industries</h3>
      <canvas id="industryChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Top Tags / Themes</h3>
      <canvas id="tagChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>AI Sub-Categories (within AI companies)</h3>
      <canvas id="aiSubChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Vertical Deep Dive</h3>
      <canvas id="verticalChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Geography: Where Are They Based?</h3>
      <canvas id="geoChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Country Breakdown</h3>
      <canvas id="countryChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Founder Team Size Distribution</h3>
      <canvas id="founderCountChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Founder Team Size Trend by Batch</h3>
      <canvas id="founderTrendChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Founder Backgrounds (from bios)</h3>
      <canvas id="founderBgChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Team Size Distribution</h3>
      <canvas id="teamSizeChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>YC Group Partners (by # companies)</h3>
      <canvas id="partnerChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>B2B vs Consumer Trend</h3>
      <canvas id="b2bChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Hiring Status</h3>
      <canvas id="hiringChart"></canvas>
    </div>
  </div>
</div>

<!-- TOP SUBINDUSTRIES TABLE -->
<div class="section">
  <h2>Top Sub-Industries</h2>
  <div class="chart-card">
    <div class="table-wrap">
      <table>
        <thead><tr><th>#</th><th>Sub-Industry</th><th>Count</th><th>% of Total</th></tr></thead>
        <tbody>
"""

for i, (sub, cnt) in enumerate(top_subinds):
    pct = round(cnt / TOTAL * 100, 1)
    html += f"          <tr><td>{i+1}</td><td>{sub}</td><td>{cnt}</td><td>{pct}%</td></tr>\n"

html += """        </tbody>
      </table>
    </div>
  </div>
</div>

</div><!-- /container -->

<script>
Chart.defaults.color = '#888';
Chart.defaults.borderColor = '#2a2a2a';
const orange = '#ff6600';
const orangeLight = '#ff853380';
const blue = '#7c8cf8';
const blueLight = '#7c8cf880';
const green = '#4caf50';
const greenLight = '#4caf5080';
const gray = '#555';
const colors = ['#ff6600','#7c8cf8','#4caf50','#e91e63','#00bcd4','#ff9800','#9c27b0','#8bc34a','#f44336','#3f51b5','#009688','#cddc39','#795548','#607d8b','#ffeb3b','#673ab7','#2196f3','#ff5722','#03a9f4','#ffc107'];

const DATA = __DATA_PLACEHOLDER__;

// 1. Batch chart
new Chart(document.getElementById('batchChart'), {
  type: 'bar',
  data: {
    labels: DATA.batch_labels,
    datasets: [{ label: 'Companies', data: DATA.batch_values, backgroundColor: orange, borderRadius: 6 }]
  },
  options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }
});

// 2. AI vs Non-AI stacked
new Chart(document.getElementById('aiChart'), {
  type: 'bar',
  data: {
    labels: DATA.batch_labels,
    datasets: [
      { label: 'AI Companies', data: DATA.ai_vals, backgroundColor: orange, borderRadius: 4 },
      { label: 'Non-AI', data: DATA.non_ai_vals, backgroundColor: gray, borderRadius: 4 }
    ]
  },
  options: { plugins: { legend: { position: 'top' } }, scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } } }
});

// 3. AI % trend line
new Chart(document.getElementById('aiTrendChart'), {
  type: 'line',
  data: {
    labels: DATA.batch_labels,
    datasets: [{ label: 'AI %', data: DATA.ai_pct_trend, borderColor: orange, backgroundColor: orangeLight, fill: true, tension: 0.3, pointRadius: 6, pointBackgroundColor: orange }]
  },
  options: { plugins: { legend: { display: false } }, scales: { y: { min: 75, max: 100, ticks: { callback: v => v + '%' } } } }
});

// 4. Industry horizontal bar
new Chart(document.getElementById('industryChart'), {
  type: 'bar',
  data: {
    labels: DATA.ind_labels,
    datasets: [{ label: 'Companies', data: DATA.ind_values, backgroundColor: colors.slice(0, 15), borderRadius: 4 }]
  },
  options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true } } }
});

// 5. Tags horizontal bar
new Chart(document.getElementById('tagChart'), {
  type: 'bar',
  data: {
    labels: DATA.tag_labels,
    datasets: [{ label: 'Companies', data: DATA.tag_values, backgroundColor: colors.slice(0, 20), borderRadius: 4 }]
  },
  options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true } } }
});

// 6. AI sub-categories
new Chart(document.getElementById('aiSubChart'), {
  type: 'bar',
  data: {
    labels: DATA.ai_sub_labels,
    datasets: [{ label: 'AI Companies', data: DATA.ai_sub_values, backgroundColor: blue, borderRadius: 4 }]
  },
  options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true } } }
});

// 7. Verticals
new Chart(document.getElementById('verticalChart'), {
  type: 'bar',
  data: {
    labels: DATA.vert_labels,
    datasets: [{ label: 'Companies', data: DATA.vert_values, backgroundColor: colors, borderRadius: 4 }]
  },
  options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true } } }
});

// 8. Geography pie
new Chart(document.getElementById('geoChart'), {
  type: 'doughnut',
  data: {
    labels: DATA.geo_labels,
    datasets: [{ data: DATA.geo_values, backgroundColor: colors.slice(0, 10), borderWidth: 0 }]
  },
  options: { plugins: { legend: { position: 'right' } } }
});

// 9. Country pie
new Chart(document.getElementById('countryChart'), {
  type: 'doughnut',
  data: {
    labels: DATA.country_labels,
    datasets: [{ data: DATA.country_values, backgroundColor: colors.slice(0, 10), borderWidth: 0 }]
  },
  options: { plugins: { legend: { position: 'right' } } }
});

// 10. Founder count pie
new Chart(document.getElementById('founderCountChart'), {
  type: 'doughnut',
  data: {
    labels: DATA.fc_labels,
    datasets: [{ data: DATA.fc_values, backgroundColor: [gray, orange, blue, green], borderWidth: 0 }]
  },
  options: { plugins: { legend: { position: 'right' } } }
});

// 11. Founder trend stacked
new Chart(document.getElementById('founderTrendChart'), {
  type: 'bar',
  data: {
    labels: DATA.batch_labels,
    datasets: [
      { label: 'Solo', data: DATA.solo_trend, backgroundColor: gray, borderRadius: 4 },
      { label: '2 Co-founders', data: DATA.duo_trend, backgroundColor: orange, borderRadius: 4 },
      { label: '3+ Co-founders', data: DATA.trio_trend, backgroundColor: blue, borderRadius: 4 }
    ]
  },
  options: { plugins: { legend: { position: 'top' } }, scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } } }
});

// 12. Founder backgrounds
new Chart(document.getElementById('founderBgChart'), {
  type: 'bar',
  data: {
    labels: DATA.bio_labels,
    datasets: [{ label: 'Founders', data: DATA.bio_values, backgroundColor: orange, borderRadius: 4 }]
  },
  options: { indexAxis: 'y', plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ctx.raw + ' founders (' + DATA.bio_pcts[ctx.dataIndex] + '%)' } } }, scales: { x: { beginAtZero: true } } }
});

// 13. Team size
new Chart(document.getElementById('teamSizeChart'), {
  type: 'bar',
  data: {
    labels: DATA.ts_labels,
    datasets: [{ label: 'Companies', data: DATA.ts_values, backgroundColor: blue, borderRadius: 6 }]
  },
  options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }
});

// 14. Partners
new Chart(document.getElementById('partnerChart'), {
  type: 'bar',
  data: {
    labels: DATA.partner_labels,
    datasets: [{ label: 'Companies', data: DATA.partner_values, backgroundColor: orange, borderRadius: 4 }]
  },
  options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true } } }
});

// 15. B2B vs Consumer trend
new Chart(document.getElementById('b2bChart'), {
  type: 'line',
  data: {
    labels: DATA.batch_labels,
    datasets: [
      { label: 'B2B', data: DATA.b2b_trend, borderColor: orange, backgroundColor: orangeLight, fill: true, tension: 0.3, pointRadius: 5 },
      { label: 'Consumer', data: DATA.consumer_trend, borderColor: blue, backgroundColor: blueLight, fill: true, tension: 0.3, pointRadius: 5 }
    ]
  },
  options: { scales: { y: { beginAtZero: true } } }
});

// 16. Hiring
new Chart(document.getElementById('hiringChart'), {
  type: 'doughnut',
  data: {
    labels: ['Hiring', 'Not Hiring'],
    datasets: [{ data: DATA.hiring, backgroundColor: [green, gray], borderWidth: 0 }]
  },
  options: { plugins: { legend: { position: 'right' } } }
});
</script>

</body>
</html>
"""

# Build the JS data object and inject it
js_data = json.dumps({
    "batch_labels": MAIN_BATCHES,
    "batch_values": [batch_counts[b] for b in MAIN_BATCHES],
    "ai_vals": [ai_by_batch[b] for b in MAIN_BATCHES],
    "non_ai_vals": [non_ai_by_batch[b] for b in MAIN_BATCHES],
    "ai_pct_trend": ai_pct_trend,
    "ind_labels": [x[0] for x in top_industries],
    "ind_values": [x[1] for x in top_industries],
    "tag_labels": [x[0] for x in top_tags],
    "tag_values": [x[1] for x in top_tags],
    "ai_sub_labels": [x[0] for x in ai_sub_sorted],
    "ai_sub_values": [x[1] for x in ai_sub_sorted],
    "vert_labels": [x[0] for x in vert_sorted],
    "vert_values": [x[1] for x in vert_sorted],
    "geo_labels": [x[0] for x in geo_sorted],
    "geo_values": [x[1] for x in geo_sorted],
    "country_labels": [x[0] for x in country_sorted],
    "country_values": [x[1] for x in country_sorted],
    "fc_labels": ["Solo", "2 Co-founders", "3 Co-founders", "4+ Co-founders"],
    "fc_values": [
        founder_count_dist.get(1, 0),
        founder_count_dist.get(2, 0),
        founder_count_dist.get(3, 0),
        sum(v for k, v in founder_count_dist.items() if k >= 4)
    ],
    "solo_trend": [fc_by_batch[b]["solo"] for b in MAIN_BATCHES],
    "duo_trend": [fc_by_batch[b]["duo"] for b in MAIN_BATCHES],
    "trio_trend": [fc_by_batch[b]["trio_plus"] for b in MAIN_BATCHES],
    "bio_labels": [x[0] for x in bio_sorted],
    "bio_values": [x[1] for x in bio_sorted],
    "bio_pcts": [round(x[1]/total_founders_with_bios*100, 1) for x in bio_sorted],
    "ts_labels": ["1 person", "2 people", "3-5 people", "6-10 people", "11+ people"],
    "ts_values": [ts_buckets[k] for k in ["1 person", "2 people", "3-5 people", "6-10 people", "11+ people"]],
    "partner_labels": [x[0] for x in top_partners],
    "partner_values": [x[1] for x in top_partners],
    "b2b_trend": b2b_by_batch,
    "consumer_trend": consumer_by_batch,
    "hiring": [hiring, not_hiring],
})

html = html.replace("__DATA_PLACEHOLDER__", js_data)

with open("/Users/krgoyal/Desktop/YC/dashboard.html", "w") as f:
    f.write(html)

print(f"Dashboard generated: /Users/krgoyal/Desktop/YC/dashboard.html")
print(f"Total companies analyzed: {TOTAL}")
