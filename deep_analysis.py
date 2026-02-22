"""
Deep analysis of YC companies: correlations, NLP clustering, cross-dimensional patterns,
competitive overlap, partner preferences, naming patterns, and non-obvious insights.
Outputs a JSON file consumed by the advanced dashboard.
"""
import json
import re
import math
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats

# ─── LOAD DATA ───
with open("/Users/krgoyal/Desktop/YC/yc_companies_enriched.json") as f:
    raw = json.load(f)

MAIN_BATCHES = ["Winter 2025", "Spring 2025", "Summer 2025", "Fall 2025", "Winter 2026"]
data = [c for c in raw if c["batch"] in MAIN_BATCHES]
TOTAL = len(data)

# ─── HELPER: AI CLASSIFICATION ───
ai_tag_kw = {"Artificial Intelligence", "AI", "Machine Learning", "Generative AI",
             "AI Assistant", "AIOps", "Conversational AI", "Reinforcement Learning",
             "Computer Vision", "NLP"}
ai_text_kw = ["ai ", " ai", "artificial intelligence", "machine learning", "llm",
              "deep learning", "neural", "generative", "language model",
              "computer vision", "reinforcement learning"]

def is_ai(c):
    if set(c.get("tags", [])) & ai_tag_kw:
        return True
    text = ((c.get("one_liner", "") or "") + " " + (c.get("long_description", "") or "") + " " + (c.get("name", "") or "")).lower()
    return any(kw in text for kw in ai_text_kw)

# ─── HELPER: FOUNDER BACKGROUND TAGS ───
bg_categories = {
    "ex_faang": ["google", "meta", "facebook", "amazon", "apple", "microsoft", "netflix"],
    "ex_yc": ["yc", "y combinator"],
    "serial_entrepreneur": ["serial entrepreneur", "second time", "previous company", "exited", "sold my", "acquired by", "founded and sold"],
    "phd": ["phd", "ph.d", "doctorate"],
    "stanford": ["stanford"],
    "mit": ["mit ", "mit,", "massachusetts institute"],
    "berkeley": ["berkeley"],
    "ivy": ["harvard", "yale", "princeton", "columbia", "penn", "upenn", "dartmouth", "cornell", "brown"],
    "ex_finance": ["goldman", "jpmorgan", "morgan stanley", "citadel", "jane street", "two sigma", "hedge fund"],
    "ex_consulting": ["mckinsey", "bain", "bcg"],
    "research": ["researcher", "research scientist", "research engineer", "postdoc"],
    "dropout": ["dropped out", "drop out"],
    "military": ["military", "army", "navy", "air force", "marine", "veteran"],
    "big_tech_non_faang": ["uber", "stripe", "airbnb", "palantir", "salesforce", "oracle", "nvidia", "tesla", "spacex", "twitter", "snap", "linkedin", "databricks", "snowflake"],
}

def get_founder_tags(company):
    tags = set()
    for f in company.get("founders", []):
        bio = (f.get("founder_bio", "") or "").lower()
        for cat, kws in bg_categories.items():
            if any(kw in bio for kw in kws):
                tags.add(cat)
    return tags

# ─── ENRICH EACH COMPANY ───
for c in data:
    c["_is_ai"] = is_ai(c)
    c["_founder_tags"] = list(get_founder_tags(c))
    c["_num_founders"] = c.get("num_founders", 0) or 0
    c["_team_size"] = c.get("team_size") or 0
    c["_text"] = ((c.get("one_liner", "") or "") + " " + (c.get("long_description", "") or "")).strip()
    c["_location_bucket"] = "SF" if "San Francisco" in (c.get("all_locations", "") or "") else \
                            "NYC" if "New York" in (c.get("all_locations", "") or "") else \
                            "Other US" if "US" == (c.get("country") or "") else \
                            "International" if c.get("country") and c.get("country") != "US" else "Unknown/Remote"

insights = {}

# ═══════════════════════════════════════════════════════════════
# 1. NLP CLUSTERING — Find hidden themes beyond YC's own tags
# ═══════════════════════════════════════════════════════════════
print("1. NLP Clustering...")
texts = [c["_text"] for c in data]
tfidf = TfidfVectorizer(max_features=500, stop_words="english", min_df=3, max_df=0.5, ngram_range=(1, 2))
X = tfidf.fit_transform(texts)

n_clusters = 15
km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = km.fit_predict(X)

# Get top terms per cluster and assign to companies
feature_names = tfidf.get_feature_names_out()
cluster_info = []
for i in range(n_clusters):
    center = km.cluster_centers_[i]
    top_indices = center.argsort()[-8:][::-1]
    top_terms = [feature_names[j] for j in top_indices]
    members = [data[j]["name"] for j in range(len(data)) if labels[j] == i]
    cluster_info.append({
        "id": i,
        "top_terms": top_terms,
        "size": len(members),
        "sample_companies": members[:8],
    })
    data_indices = [j for j in range(len(data)) if labels[j] == i]
    # What % of this cluster is AI?
    ai_pct = sum(1 for j in data_indices if data[j]["_is_ai"]) / max(len(data_indices), 1) * 100
    cluster_info[-1]["ai_pct"] = round(ai_pct, 1)
    # Dominant batch?
    batch_dist = Counter(data[j]["batch"] for j in data_indices)
    cluster_info[-1]["batch_dist"] = dict(batch_dist)

# Sort clusters by size
cluster_info.sort(key=lambda x: -x["size"])
insights["nlp_clusters"] = cluster_info

# PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X.toarray())
scatter_data = []
for i, c in enumerate(data):
    scatter_data.append({
        "x": round(float(coords[i, 0]), 4),
        "y": round(float(coords[i, 1]), 4),
        "name": c["name"],
        "cluster": int(labels[i]),
        "is_ai": c["_is_ai"],
        "batch": c["batch"],
    })
insights["scatter_data"] = scatter_data

# ═══════════════════════════════════════════════════════════════
# 2. FOUNDER BACKGROUND × VERTICAL CORRELATIONS
# ═══════════════════════════════════════════════════════════════
print("2. Founder background correlations...")

verticals = {
    "Developer Tools": lambda c: "Developer Tools" in c.get("tags", []),
    "Healthcare": lambda c: "Healthcare" in c.get("industries", []) or any("Health" in t for t in c.get("tags", [])),
    "Fintech": lambda c: "Fintech" in c.get("industries", []) or "Fintech" in c.get("tags", []),
    "Robotics/HW": lambda c: any("Robot" in t or "Hardware" in t for t in c.get("tags", [])),
    "Security": lambda c: "Security" in c.get("industries", []) or "Cybersecurity" in c.get("tags", []),
    "Legal": lambda c: "Legal" in c.get("industries", []),
    "Infra": lambda c: "Infrastructure" in c.get("industries", []),
    "Sales/Marketing": lambda c: "Sales" in c.get("industries", []) or "Marketing" in c.get("industries", []),
    "Education": lambda c: "Education" in c.get("industries", []),
}

# Build a matrix: founder_bg × vertical
bg_vert_matrix = {}
for bg in bg_categories:
    bg_vert_matrix[bg] = {}
    bg_companies = [c for c in data if bg in c["_founder_tags"]]
    if not bg_companies:
        continue
    for vert_name, vert_fn in verticals.items():
        pct = sum(1 for c in bg_companies if vert_fn(c)) / len(bg_companies) * 100
        bg_vert_matrix[bg][vert_name] = round(pct, 1)

# Find statistically significant over-representations
bg_vert_insights = []
for bg in bg_categories:
    bg_companies = [c for c in data if bg in c["_founder_tags"]]
    if len(bg_companies) < 10:
        continue
    for vert_name, vert_fn in verticals.items():
        # Expected rate (base rate in full dataset)
        base_rate = sum(1 for c in data if vert_fn(c)) / TOTAL
        observed = sum(1 for c in bg_companies if vert_fn(c))
        expected = base_rate * len(bg_companies)
        if expected < 2:
            continue
        # Chi-squared-like ratio
        ratio = observed / max(expected, 0.1)
        if ratio > 1.5 and observed >= 5:
            bg_vert_insights.append({
                "background": bg,
                "vertical": vert_name,
                "observed": observed,
                "expected": round(expected, 1),
                "ratio": round(ratio, 2),
                "bg_total": len(bg_companies),
                "description": f"{bg} founders are {ratio:.1f}x more likely to build in {vert_name} (observed {observed} vs expected {expected:.0f})"
            })

bg_vert_insights.sort(key=lambda x: -x["ratio"])
insights["founder_vertical_correlations"] = bg_vert_insights[:20]
insights["bg_vert_matrix"] = bg_vert_matrix

# ═══════════════════════════════════════════════════════════════
# 3. TEAM COMPOSITION PATTERNS
# ═══════════════════════════════════════════════════════════════
print("3. Team composition patterns...")

team_patterns = {}
for team_type, filter_fn in [
    ("solo", lambda c: c["_num_founders"] == 1),
    ("duo", lambda c: c["_num_founders"] == 2),
    ("trio_plus", lambda c: c["_num_founders"] >= 3),
]:
    subset = [c for c in data if filter_fn(c)]
    if not subset:
        continue
    team_patterns[team_type] = {
        "count": len(subset),
        "ai_pct": round(sum(1 for c in subset if c["_is_ai"]) / len(subset) * 100, 1),
        "hiring_pct": round(sum(1 for c in subset if c.get("is_hiring")) / len(subset) * 100, 1),
        "avg_team_size": round(sum(c["_team_size"] for c in subset) / len(subset), 1),
        "b2b_pct": round(sum(1 for c in subset if "B2B" in c.get("industries", [])) / len(subset) * 100, 1),
        "top_verticals": Counter(
            ind for c in subset for ind in c.get("industries", [])
        ).most_common(5),
        "sf_pct": round(sum(1 for c in subset if c["_location_bucket"] == "SF") / len(subset) * 100, 1),
        "has_phd_pct": round(sum(1 for c in subset if "phd" in c["_founder_tags"]) / len(subset) * 100, 1),
        "has_faang_pct": round(sum(1 for c in subset if "ex_faang" in c["_founder_tags"]) / len(subset) * 100, 1),
        "has_yc_repeat_pct": round(sum(1 for c in subset if "ex_yc" in c["_founder_tags"]) / len(subset) * 100, 1),
    }

insights["team_patterns"] = team_patterns

# ═══════════════════════════════════════════════════════════════
# 4. BATCH EVOLUTION — What's growing, dying, emerging
# ═══════════════════════════════════════════════════════════════
print("4. Batch evolution analysis...")

# Track tag prevalence across batches
tag_evolution = {}
for tag in ["Artificial Intelligence", "AI", "Developer Tools", "SaaS", "Generative AI",
            "Robotics", "Open Source", "Fintech", "Healthcare", "Machine Learning",
            "AI Assistant", "Reinforcement Learning", "Hardware", "Conversational AI",
            "Computer Vision", "Cybersecurity", "AIOps", "Workflow Automation",
            "Hard Tech", "Infrastructure"]:
    tag_evolution[tag] = {}
    for b in MAIN_BATCHES:
        bc = [c for c in data if c["batch"] == b]
        count = sum(1 for c in bc if tag in c.get("tags", []))
        tag_evolution[tag][b] = round(count / len(bc) * 100, 1) if bc else 0

# Find tags with biggest growth/decline
tag_trends = []
for tag, batches in tag_evolution.items():
    vals = [batches[b] for b in MAIN_BATCHES]
    if max(vals) < 2:
        continue
    # Linear regression on batch index
    x = np.arange(len(vals))
    slope, intercept, r, p, se = stats.linregress(x, vals)
    tag_trends.append({
        "tag": tag,
        "values": vals,
        "slope": round(slope, 2),
        "r_squared": round(r**2, 3),
        "p_value": round(p, 4),
        "direction": "growing" if slope > 0.5 else "declining" if slope < -0.5 else "stable",
        "latest": vals[-1],
        "earliest": vals[0],
    })

tag_trends.sort(key=lambda x: -abs(x["slope"]))
insights["tag_evolution"] = tag_evolution
insights["tag_trends"] = tag_trends

# Industry evolution
industry_evolution = {}
for ind in ["B2B", "Consumer", "Healthcare", "Fintech", "Industrials", "Infrastructure",
            "Engineering, Product and Design", "Security", "Legal", "Education"]:
    industry_evolution[ind] = {}
    for b in MAIN_BATCHES:
        bc = [c for c in data if c["batch"] == b]
        count = sum(1 for c in bc if ind in c.get("industries", []))
        industry_evolution[ind][b] = round(count / len(bc) * 100, 1) if bc else 0

insights["industry_evolution"] = industry_evolution

# ═══════════════════════════════════════════════════════════════
# 5. YC PARTNER PREFERENCES
# ═══════════════════════════════════════════════════════════════
print("5. YC Partner preferences...")

partner_profiles = {}
top_partners = Counter(c.get("group_partner", "") for c in data if c.get("group_partner")).most_common(12)

for partner_name, total_count in top_partners:
    pc = [c for c in data if c.get("group_partner") == partner_name]
    if len(pc) < 15:
        continue
    
    profile = {
        "total": total_count,
        "ai_pct": round(sum(1 for c in pc if c["_is_ai"]) / len(pc) * 100, 1),
        "top_industries": dict(Counter(ind for c in pc for ind in c.get("industries", [])).most_common(5)),
        "top_tags": dict(Counter(t for c in pc for t in c.get("tags", [])).most_common(8)),
        "avg_team_size": round(sum(c["_team_size"] for c in pc) / len(pc), 1),
        "avg_founders": round(sum(c["_num_founders"] for c in pc) / len(pc), 1),
        "solo_pct": round(sum(1 for c in pc if c["_num_founders"] == 1) / len(pc) * 100, 1),
        "sf_pct": round(sum(1 for c in pc if c["_location_bucket"] == "SF") / len(pc) * 100, 1),
        "hiring_pct": round(sum(1 for c in pc if c.get("is_hiring")) / len(pc) * 100, 1),
        "faang_founder_pct": round(sum(1 for c in pc if "ex_faang" in c["_founder_tags"]) / len(pc) * 100, 1),
        "phd_founder_pct": round(sum(1 for c in pc if "phd" in c["_founder_tags"]) / len(pc) * 100, 1),
    }
    
    # Find what this partner over-indexes on vs the average
    overall_ai_pct = sum(1 for c in data if c["_is_ai"]) / TOTAL * 100
    profile["ai_delta"] = round(profile["ai_pct"] - overall_ai_pct, 1)
    
    partner_profiles[partner_name] = profile

insights["partner_profiles"] = partner_profiles

# ═══════════════════════════════════════════════════════════════
# 6. COMPETITIVE OVERLAP / CROWDING ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("6. Competitive overlap analysis...")

# Use TF-IDF similarity to find companies that are very similar to each other
from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(X)
np.fill_diagonal(sim_matrix, 0)

# Find highly similar pairs (potential competitors in same batch)
similar_pairs = []
for i in range(len(data)):
    for j in range(i+1, len(data)):
        if sim_matrix[i, j] > 0.45:
            similar_pairs.append({
                "company_a": data[i]["name"],
                "company_b": data[j]["name"],
                "similarity": round(float(sim_matrix[i, j]), 3),
                "same_batch": data[i]["batch"] == data[j]["batch"],
                "batch_a": data[i]["batch"],
                "batch_b": data[j]["batch"],
                "one_liner_a": (data[i].get("one_liner", "") or "")[:100],
                "one_liner_b": (data[j].get("one_liner", "") or "")[:100],
            })

similar_pairs.sort(key=lambda x: -x["similarity"])
insights["similar_pairs"] = similar_pairs[:40]

# Crowding score: for each company, how many near-competitors exist?
crowding_scores = []
for i, c in enumerate(data):
    near_competitors = sum(1 for j in range(len(data)) if j != i and sim_matrix[i, j] > 0.3)
    crowding_scores.append(near_competitors)

# Most crowded spaces
crowded_companies = sorted(
    [(data[i]["name"], data[i]["batch"], crowding_scores[i], (data[i].get("one_liner", "") or "")[:80]) for i in range(len(data))],
    key=lambda x: -x[2]
)[:25]
insights["most_crowded"] = [{"name": x[0], "batch": x[1], "competitors": x[2], "one_liner": x[3]} for x in crowded_companies]

# Least crowded (most unique)
unique_companies = sorted(
    [(data[i]["name"], data[i]["batch"], crowding_scores[i], (data[i].get("one_liner", "") or "")[:80]) for i in range(len(data))],
    key=lambda x: x[2]
)[:25]
insights["most_unique"] = [{"name": x[0], "batch": x[1], "competitors": x[2], "one_liner": x[3]} for x in unique_companies]

# Average crowding by vertical
vert_crowding = {}
for vert_name, vert_fn in verticals.items():
    vert_indices = [i for i, c in enumerate(data) if vert_fn(c)]
    if vert_indices:
        vert_crowding[vert_name] = round(np.mean([crowding_scores[i] for i in vert_indices]), 1)
insights["vertical_crowding"] = dict(sorted(vert_crowding.items(), key=lambda x: -x[1]))

# ═══════════════════════════════════════════════════════════════
# 7. NAMING PATTERNS
# ═══════════════════════════════════════════════════════════════
print("7. Naming patterns...")

names = [c["name"] for c in data]
name_lengths = [len(n) for n in names]
name_word_counts = [len(n.split()) for n in names]

# Single word vs multi-word
single_word = sum(1 for n in names if len(n.split()) == 1)
two_word = sum(1 for n in names if len(n.split()) == 2)
three_plus = sum(1 for n in names if len(n.split()) >= 3)

# Names ending in common suffixes
suffix_patterns = {
    "ends_with_AI": sum(1 for n in names if n.lower().endswith(" ai") or n.lower().endswith(".ai")),
    "ends_with_Labs": sum(1 for n in names if "lab" in n.lower().split()[-1]),
    "ends_with_HQ": sum(1 for n in names if n.lower().endswith("hq")),
    "contains_dot": sum(1 for n in names if "." in n),
    "all_lowercase": sum(1 for n in names if n == n.lower()),
    "has_number": sum(1 for n in names if any(ch.isdigit() for ch in n)),
    "single_word_short": sum(1 for n in names if len(n.split()) == 1 and len(n) <= 5),
    "is_real_word": 0,  # placeholder
    "contains_AI_in_name": sum(1 for n in names if "ai" in n.lower()),
}

# AI in name vs not — is there a correlation with anything?
ai_in_name = [c for c in data if "ai" in c["name"].lower()]
ai_not_in_name = [c for c in data if "ai" not in c["name"].lower() and c["_is_ai"]]

naming_insights = {
    "total_names": len(names),
    "avg_length": round(np.mean(name_lengths), 1),
    "single_word_pct": round(single_word / len(names) * 100, 1),
    "two_word_pct": round(two_word / len(names) * 100, 1),
    "three_plus_pct": round(three_plus / len(names) * 100, 1),
    "suffix_patterns": suffix_patterns,
    "ai_in_name_count": len(ai_in_name),
    "ai_in_name_pct": round(len(ai_in_name) / len(names) * 100, 1),
}

insights["naming_patterns"] = naming_insights

# ═══════════════════════════════════════════════════════════════
# 8. DESCRIPTION BUZZWORD FREQUENCY & EVOLUTION
# ═══════════════════════════════════════════════════════════════
print("8. Buzzword analysis...")

buzzwords = ["agent", "agentic", "autonomous", "copilot", "autopilot", "platform",
             "infrastructure", "workflow", "automate", "automation", "real-time", "realtime",
             "enterprise", "api", "open source", "self-driving", "end-to-end",
             "vertical", "horizontal", "saas", "marketplace", "no-code", "low-code",
             "rag", "fine-tune", "fine-tuning", "embedding", "vector", "multimodal",
             "reasoning", "inference", "gpu", "model", "foundation model", "frontier",
             "compliance", "regulation", "hipaa", "soc2", "gdpr",
             "b2b", "smb", "mid-market", "enterprise",
             "10x", "100x", "1000x"]

buzzword_by_batch = {}
for bw in buzzwords:
    buzzword_by_batch[bw] = {}
    for b in MAIN_BATCHES:
        bc = [c for c in data if c["batch"] == b]
        count = sum(1 for c in bc if bw in c["_text"].lower())
        buzzword_by_batch[bw][b] = count

# Find trending buzzwords
buzzword_trends = []
for bw, batches in buzzword_by_batch.items():
    vals = [batches[b] for b in MAIN_BATCHES]
    total = sum(vals)
    if total < 5:
        continue
    x = np.arange(len(vals))
    slope, _, r, p, _ = stats.linregress(x, vals)
    buzzword_trends.append({
        "word": bw,
        "total": total,
        "values": vals,
        "slope": round(slope, 2),
        "direction": "rising" if slope > 0.5 else "falling" if slope < -0.5 else "stable",
    })

buzzword_trends.sort(key=lambda x: -abs(x["slope"]))
insights["buzzword_trends"] = buzzword_trends
insights["buzzword_by_batch"] = buzzword_by_batch

# ═══════════════════════════════════════════════════════════════
# 9. CROSS-CORRELATIONS: AI × Location × Hiring × Team Size
# ═══════════════════════════════════════════════════════════════
print("9. Cross-correlations...")

# Build a correlation matrix of binary/numeric features
features = {}
for i, c in enumerate(data):
    features[i] = {
        "is_ai": 1 if c["_is_ai"] else 0,
        "is_b2b": 1 if "B2B" in c.get("industries", []) else 0,
        "is_consumer": 1 if "Consumer" in c.get("industries", []) else 0,
        "is_sf": 1 if c["_location_bucket"] == "SF" else 0,
        "is_hiring": 1 if c.get("is_hiring") else 0,
        "num_founders": c["_num_founders"],
        "team_size": c["_team_size"],
        "is_solo": 1 if c["_num_founders"] == 1 else 0,
        "has_faang": 1 if "ex_faang" in c["_founder_tags"] else 0,
        "has_phd": 1 if "phd" in c["_founder_tags"] else 0,
        "has_yc_repeat": 1 if "ex_yc" in c["_founder_tags"] else 0,
        "has_serial": 1 if "serial_entrepreneur" in c["_founder_tags"] else 0,
        "is_devtools": 1 if "Developer Tools" in c.get("tags", []) else 0,
        "is_healthcare": 1 if "Healthcare" in c.get("industries", []) else 0,
        "is_fintech": 1 if "Fintech" in c.get("industries", []) else 0,
        "is_robotics": 1 if any("Robot" in t for t in c.get("tags", [])) else 0,
        "is_open_source": 1 if "Open Source" in c.get("tags", []) else 0,
        "is_saas": 1 if "SaaS" in c.get("tags", []) else 0,
        "ai_in_name": 1 if "ai" in c["name"].lower() else 0,
        "name_length": len(c["name"]),
    }

df = pd.DataFrame.from_dict(features, orient="index")
corr = df.corr()

# Find strongest non-trivial correlations
corr_pairs = []
cols = list(corr.columns)
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        r = corr.iloc[i, j]
        if abs(r) > 0.1 and not np.isnan(r):
            corr_pairs.append({
                "feature_a": cols[i],
                "feature_b": cols[j],
                "correlation": round(float(r), 3),
                "abs_corr": round(abs(float(r)), 3),
            })

corr_pairs.sort(key=lambda x: -x["abs_corr"])
insights["correlations"] = corr_pairs[:30]

# Full correlation matrix for heatmap
insights["correlation_matrix"] = {
    "labels": cols,
    "values": [[round(float(corr.iloc[i, j]), 3) for j in range(len(cols))] for i in range(len(cols))]
}

# ═══════════════════════════════════════════════════════════════
# 10. NON-AI COMPANIES — What are they?
# ═══════════════════════════════════════════════════════════════
print("10. Non-AI company analysis...")

non_ai = [c for c in data if not c["_is_ai"]]
non_ai_industries = Counter(ind for c in non_ai for ind in c.get("industries", []))
non_ai_tags = Counter(t for c in non_ai for t in c.get("tags", []))

insights["non_ai_analysis"] = {
    "count": len(non_ai),
    "pct": round(len(non_ai) / TOTAL * 100, 1),
    "top_industries": dict(non_ai_industries.most_common(10)),
    "top_tags": dict(non_ai_tags.most_common(10)),
    "examples": [{"name": c["name"], "one_liner": (c.get("one_liner", "") or "")[:100], "batch": c["batch"]} for c in non_ai[:15]],
    "sf_pct": round(sum(1 for c in non_ai if c["_location_bucket"] == "SF") / max(len(non_ai), 1) * 100, 1),
    "hiring_pct": round(sum(1 for c in non_ai if c.get("is_hiring")) / max(len(non_ai), 1) * 100, 1),
}

# ═══════════════════════════════════════════════════════════════
# 11. "GOLD STANDARD" PATTERN — What do the most-hired companies look like?
# ═══════════════════════════════════════════════════════════════
print("11. Hiring company patterns...")

hiring_companies = [c for c in data if c.get("is_hiring")]
not_hiring_companies = [c for c in data if not c.get("is_hiring")]

def profile_subset(subset, label):
    if not subset:
        return {}
    return {
        "label": label,
        "count": len(subset),
        "ai_pct": round(sum(1 for c in subset if c["_is_ai"]) / len(subset) * 100, 1),
        "avg_team_size": round(sum(c["_team_size"] for c in subset) / len(subset), 1),
        "avg_founders": round(sum(c["_num_founders"] for c in subset) / len(subset), 1),
        "sf_pct": round(sum(1 for c in subset if c["_location_bucket"] == "SF") / len(subset) * 100, 1),
        "b2b_pct": round(sum(1 for c in subset if "B2B" in c.get("industries", [])) / len(subset) * 100, 1),
        "faang_pct": round(sum(1 for c in subset if "ex_faang" in c["_founder_tags"]) / len(subset) * 100, 1),
        "phd_pct": round(sum(1 for c in subset if "phd" in c["_founder_tags"]) / len(subset) * 100, 1),
        "yc_repeat_pct": round(sum(1 for c in subset if "ex_yc" in c["_founder_tags"]) / len(subset) * 100, 1),
        "devtools_pct": round(sum(1 for c in subset if "Developer Tools" in c.get("tags", [])) / len(subset) * 100, 1),
        "open_source_pct": round(sum(1 for c in subset if "Open Source" in c.get("tags", [])) / len(subset) * 100, 1),
    }

insights["hiring_vs_not"] = {
    "hiring": profile_subset(hiring_companies, "Hiring"),
    "not_hiring": profile_subset(not_hiring_companies, "Not Hiring"),
}

# ═══════════════════════════════════════════════════════════════
# 12. FOUNDER PAIR PATTERNS — What backgrounds pair together?
# ═══════════════════════════════════════════════════════════════
print("12. Founder pair patterns...")

pair_counter = Counter()
for c in data:
    founders = c.get("founders", [])
    if len(founders) < 2:
        continue
    # Get background tags for each founder individually
    individual_tags = []
    for f in founders:
        bio = (f.get("founder_bio", "") or "").lower()
        ftags = set()
        for cat, kws in bg_categories.items():
            if any(kw in bio for kw in kws):
                ftags.add(cat)
        individual_tags.append(ftags)
    
    # Find all pairs of background types across founders
    all_tags = set()
    for ft in individual_tags:
        all_tags |= ft
    for t1, t2 in combinations(sorted(all_tags), 2):
        # Check if different founders have these tags
        f1_has = [ft for ft in individual_tags if t1 in ft]
        f2_has = [ft for ft in individual_tags if t2 in ft]
        if f1_has and f2_has:
            pair_counter[(t1, t2)] += 1

founder_pairs = [{"bg_a": p[0], "bg_b": p[1], "count": c} for p, c in pair_counter.most_common(20)]
insights["founder_pair_patterns"] = founder_pairs

# ═══════════════════════════════════════════════════════════════
# 13. "WHAT YC WANTS" — Composite signal
# ═══════════════════════════════════════════════════════════════
print("13. Composite YC archetype...")

# The "median" YC company profile
archetype = {
    "is_ai": round(sum(1 for c in data if c["_is_ai"]) / TOTAL * 100, 1),
    "is_b2b": round(sum(1 for c in data if "B2B" in c.get("industries", [])) / TOTAL * 100, 1),
    "is_sf": round(sum(1 for c in data if c["_location_bucket"] == "SF") / TOTAL * 100, 1),
    "median_founders": 2,
    "median_team_size": int(np.median([c["_team_size"] for c in data if c["_team_size"] > 0])),
    "is_saas": round(sum(1 for c in data if "SaaS" in c.get("tags", [])) / TOTAL * 100, 1),
    "has_faang_founder": round(sum(1 for c in data if "ex_faang" in c["_founder_tags"]) / TOTAL * 100, 1),
    "has_repeat_yc": round(sum(1 for c in data if "ex_yc" in c["_founder_tags"]) / TOTAL * 100, 1),
    "is_devtools": round(sum(1 for c in data if "Developer Tools" in c.get("tags", [])) / TOTAL * 100, 1),
    "is_hiring": round(sum(1 for c in data if c.get("is_hiring")) / TOTAL * 100, 1),
}
insights["yc_archetype"] = archetype

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
with open("/Users/krgoyal/Desktop/YC/deep_insights.json", "w") as f:
    json.dump(insights, f, indent=2, default=str)

print(f"\nDone! Deep insights saved. Keys: {list(insights.keys())}")
print(f"Companies analyzed: {TOTAL}")
print(f"NLP clusters: {n_clusters}")
print(f"Similar pairs found: {len(similar_pairs)}")
print(f"Founder-vertical correlations: {len(bg_vert_insights)}")
print(f"Cross-correlations: {len(corr_pairs)}")
