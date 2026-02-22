"""
Build the final merged dashboard: all basic charts from v1 + all deep insights from v2.
Clean, modern design. Single HTML file with embedded data.
"""
import json
from collections import Counter, defaultdict
import numpy as np

# ─── LOAD ───
with open("/Users/krgoyal/Desktop/YC/yc_companies_enriched.json") as f:
    raw = json.load(f)
with open("/Users/krgoyal/Desktop/YC/deep_insights.json") as f:
    deep = json.load(f)

MAIN_BATCHES = ["Winter 2025", "Spring 2025", "Summer 2025", "Fall 2025", "Winter 2026"]
data = [c for c in raw if c["batch"] in MAIN_BATCHES]
TOTAL = len(data)

# ─── AI CLASSIFICATION ───
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

# ─── BASIC DATA (from v1) ───
batch_counts = {b: sum(1 for c in data if c["batch"] == b) for b in MAIN_BATCHES}

ai_by_batch = {b: sum(1 for c in data if c["batch"] == b and is_ai(c)) for b in MAIN_BATCHES}
non_ai_by_batch = {b: batch_counts[b] - ai_by_batch[b] for b in MAIN_BATCHES}
ai_pct_trend = [round(ai_by_batch[b] / batch_counts[b] * 100, 1) for b in MAIN_BATCHES]
total_ai = sum(ai_by_batch.values())

# Industries
industry_counter = Counter(ind for c in data for ind in c.get("industries", []))
top_industries = industry_counter.most_common(15)

# Tags
tag_counter = Counter(t for c in data for t in c.get("tags", []))
top_tags = tag_counter.most_common(20)

# Geography
def get_loc(c):
    loc = (c.get("all_locations", "") or "").strip()
    if not loc: return "Unknown/Remote"
    if "San Francisco" in loc: return "San Francisco"
    if "New York" in loc: return "New York"
    if any(x in loc for x in ["Palo Alto", "San Mateo", "Mountain View", "Sunnyvale", "San Jose"]): return "SF Bay Area (other)"
    if "London" in loc: return "London"
    if "Remote" in loc: return "Remote"
    if any(x in loc for x in ["Toronto", "Canada"]): return "Canada"
    regions = c.get("regions", [])
    if "United States of America" in regions: return "Other US"
    if "Europe" in regions: return "Europe (other)"
    return "Other"

geo_counter = Counter(get_loc(c) for c in data)
geo_sorted = geo_counter.most_common(10)

# Founder count distribution
fc_dist = Counter(c.get("num_founders", 0) or 0 for c in data)

# Founder backgrounds
bg_categories = {
    "Ex-FAANG": ["google", "meta", "facebook", "amazon", "apple", "microsoft", "netflix"],
    "Ex-YC Founder": ["yc", "y combinator"],
    "Serial Entrepreneur": ["serial entrepreneur", "second time", "previous company", "exited", "sold my", "acquired by"],
    "PhD Holder": ["phd", "ph.d", "doctorate"],
    "Stanford": ["stanford"],
    "MIT": ["mit ", "mit,", "massachusetts institute"],
    "Berkeley": ["berkeley"],
    "Ivy League": ["harvard", "yale", "princeton", "columbia", "penn", "upenn", "dartmouth", "cornell", "brown"],
    "Ex-Finance": ["goldman", "jpmorgan", "morgan stanley", "citadel", "jane street", "two sigma"],
    "Ex-Consulting (MBB)": ["mckinsey", "bain", "bcg"],
    "Research Background": ["researcher", "research scientist", "research engineer", "postdoc"],
    "Dropout": ["dropped out", "drop out"],
    "Military/Veteran": ["military", "army", "navy", "air force", "marine", "veteran"],
}

bio_results = {}
total_founders_with_bios = 0
for c in data:
    for f in c.get("founders", []):
        bio = (f.get("founder_bio", "") or "").lower()
        if not bio: continue
        total_founders_with_bios += 1
        for cat, kws in bg_categories.items():
            if any(kw in bio for kw in kws):
                bio_results[cat] = bio_results.get(cat, 0) + 1

bio_sorted = sorted(bio_results.items(), key=lambda x: -x[1])

# Team size buckets
ts_buckets = Counter()
for c in data:
    ts = c.get("team_size") or 0
    if ts <= 1: ts_buckets["1"] += 1
    elif ts <= 2: ts_buckets["2"] += 1
    elif ts <= 5: ts_buckets["3-5"] += 1
    elif ts <= 10: ts_buckets["6-10"] += 1
    else: ts_buckets["11+"] += 1

# Group partners
partner_counter = Counter(c.get("group_partner", "") for c in data if c.get("group_partner"))
top_partners = partner_counter.most_common(12)

# B2B vs Consumer by batch
b2b_by_batch = [sum(1 for c in data if c["batch"] == b and "B2B" in c.get("industries", [])) for b in MAIN_BATCHES]
consumer_by_batch = [sum(1 for c in data if c["batch"] == b and "Consumer" in c.get("industries", [])) for b in MAIN_BATCHES]

# Hiring
hiring = sum(1 for c in data if c.get("is_hiring"))

# Verticals
verticals_data = {}
for name, fn in [
    ("Developer Tools", lambda c: "Developer Tools" in c.get("tags", [])),
    ("Healthcare/Biotech", lambda c: "Healthcare" in c.get("industries", []) or any("Health" in t or "Bio" in t for t in c.get("tags", []))),
    ("Fintech", lambda c: "Fintech" in c.get("industries", []) or "Fintech" in c.get("tags", [])),
    ("Robotics/Hardware", lambda c: any("Robot" in t or "Hardware" in t for t in c.get("tags", []))),
    ("Security/Cyber", lambda c: "Security" in c.get("industries", []) or "Cybersecurity" in c.get("tags", [])),
    ("Open Source", lambda c: "Open Source" in c.get("tags", [])),
    ("Legal Tech", lambda c: "Legal" in c.get("industries", [])),
    ("Education", lambda c: "Education" in c.get("industries", [])),
    ("Energy/Climate", lambda c: "Energy" in c.get("industries", []) or any(kw in ((c.get("one_liner","") or "")+" ".join(c.get("tags",[]))).lower() for kw in ["climate","carbon","solar","energy"])),
    ("Space/Aviation", lambda c: "Aviation and Space" in c.get("industries", [])),
    ("Defense/Gov", lambda c: any(kw in ((c.get("one_liner","") or "")+" ".join(c.get("tags",[]))).lower() for kw in ["defense","defence","military","government"])),
]:
    verticals_data[name] = sum(1 for c in data if fn(c))

vert_sorted = sorted(verticals_data.items(), key=lambda x: -x[1])

# AI sub-categories
ai_companies = [c for c in data if is_ai(c)]
ai_sub = {}
for cat, kws in [
    ("AI Agents / Automation", ["agent", "automat", "workflow", "autonomous"]),
    ("AI Infra / DevTools", ["infrastructure", "developer tool", "api", "platform", "framework"]),
    ("AI for Sales/Marketing", ["sales", "marketing", "outreach", "lead gen", "crm"]),
    ("AI for Healthcare", ["health", "medical", "clinical", "patient", "diagnos"]),
    ("AI for Finance", ["finance", "fintech", "banking", "payment", "accounting"]),
    ("AI for Legal", ["legal", "law", "compliance", "contract"]),
    ("AI Coding / Dev", ["code", "coding", "software develop", "programming", "debug"]),
    ("Generative AI / Content", ["generat", "content", "image", "video", "creative"]),
    ("Conversational AI / Voice", ["conversation", "chat", "voice", "call center"]),
    ("Computer Vision", ["vision", "image recogn", "object detect", "visual"]),
    ("Robotics + AI", ["robot", "autonomous vehicle", "drone"]),
]:
    count = sum(1 for c in ai_companies if any(kw in ((c.get("one_liner","") or "")+" "+" ".join(c.get("tags",[]))+" "+" ".join(c.get("industries",[]))).lower() for kw in kws))
    ai_sub[cat] = count

ai_sub_sorted = sorted(ai_sub.items(), key=lambda x: -x[1])

# Founder team trend by batch
fc_by_batch = {}
for b in MAIN_BATCHES:
    bc = [c for c in data if c["batch"] == b]
    fc_by_batch[b] = {
        "solo": sum(1 for c in bc if (c.get("num_founders", 0) or 0) == 1),
        "duo": sum(1 for c in bc if (c.get("num_founders", 0) or 0) == 2),
        "trio_plus": sum(1 for c in bc if (c.get("num_founders", 0) or 0) >= 3),
    }

# Subindustries
subind_counter = Counter(c.get("subindustry", "") for c in data if c.get("subindustry"))
top_subinds = subind_counter.most_common(10)

# SaaS count
saas_count = sum(1 for c in data if "SaaS" in c.get("tags", []))

# ─── BUILD COMBINED DATA OBJECT ───
basic = {
    "total": TOTAL,
    "total_founders": total_founders_with_bios,
    "total_ai": total_ai,
    "total_non_ai": TOTAL - total_ai,
    "batches": MAIN_BATCHES,
    "batch_values": [batch_counts[b] for b in MAIN_BATCHES],
    "ai_vals": [ai_by_batch[b] for b in MAIN_BATCHES],
    "non_ai_vals": [non_ai_by_batch[b] for b in MAIN_BATCHES],
    "ai_pct_trend": ai_pct_trend,
    "ind_labels": [x[0] for x in top_industries],
    "ind_values": [x[1] for x in top_industries],
    "tag_labels": [x[0] for x in top_tags],
    "tag_values": [x[1] for x in top_tags],
    "geo_labels": [x[0] for x in geo_sorted],
    "geo_values": [x[1] for x in geo_sorted],
    "fc_labels": ["Solo", "2 Co-founders", "3 Co-founders", "4+"],
    "fc_values": [fc_dist.get(1, 0), fc_dist.get(2, 0), fc_dist.get(3, 0), sum(v for k, v in fc_dist.items() if k >= 4)],
    "bio_labels": [x[0] for x in bio_sorted],
    "bio_values": [x[1] for x in bio_sorted],
    "bio_pcts": [round(x[1] / total_founders_with_bios * 100, 1) for x in bio_sorted],
    "ts_labels": ["1", "2", "3-5", "6-10", "11+"],
    "ts_values": [ts_buckets[k] for k in ["1", "2", "3-5", "6-10", "11+"]],
    "partner_labels": [x[0] for x in top_partners],
    "partner_values": [x[1] for x in top_partners],
    "b2b_trend": b2b_by_batch,
    "consumer_trend": consumer_by_batch,
    "hiring": [hiring, TOTAL - hiring],
    "vert_labels": [x[0] for x in vert_sorted],
    "vert_values": [x[1] for x in vert_sorted],
    "ai_sub_labels": [x[0] for x in ai_sub_sorted],
    "ai_sub_values": [x[1] for x in ai_sub_sorted],
    "solo_trend": [fc_by_batch[b]["solo"] for b in MAIN_BATCHES],
    "duo_trend": [fc_by_batch[b]["duo"] for b in MAIN_BATCHES],
    "trio_trend": [fc_by_batch[b]["trio_plus"] for b in MAIN_BATCHES],
    "subind_labels": [x[0] for x in top_subinds],
    "subind_values": [x[1] for x in top_subinds],
    "saas_count": saas_count,
    "sf_count": geo_counter.get("San Francisco", 0),
    "nyc_count": geo_counter.get("New York", 0),
    "b2b_pct": round(sum(1 for c in data if "B2B" in c.get("industries", [])) / TOTAL * 100),
    "consumer_pct": round(sum(1 for c in data if "Consumer" in c.get("industries", [])) / TOTAL * 100),
    "sf_pct": round(geo_counter.get("San Francisco", 0) / TOTAL * 100),
    "duo_pct": round(fc_dist.get(2, 0) / TOTAL * 100),
    "inactive_count": sum(1 for c in data if c.get("status") == "Inactive"),
    "acquired_count": sum(1 for c in data if c.get("status") == "Acquired"),
}

# Merge basic + deep
all_data = {**basic, **deep}

# ─── WRITE HTML ───
with open("/Users/krgoyal/Desktop/YC/dashboard_final.html") as f:
    template = f.read()

template = template.replace("__DATA_PLACEHOLDER__", json.dumps(all_data, default=str))

with open("/Users/krgoyal/Desktop/YC/dashboard_final.html", "w") as f:
    f.write(template)

print(f"Final dashboard generated: {len(template)} chars")
print(f"Data keys: {len(all_data)} keys")
