"""
Fetch founder details for all YC companies by scraping individual company pages.
"""
import json
import urllib.request
import re
import time
import ssl
import certifi
import sys
import os

ssl_context = ssl.create_default_context(cafile=certifi.where())

def fetch_company_page(slug):
    """Fetch and parse a YC company page for founder info."""
    url = f"https://www.ycombinator.com/companies/{slug}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    })
    try:
        with urllib.request.urlopen(req, context=ssl_context, timeout=15) as resp:
            html = resp.read().decode("utf-8")
    except Exception as e:
        return {"error": str(e)}
    
    # Extract Inertia page data
    match = re.search(r'data-page="(\{.*?\})"', html, re.DOTALL)
    if not match:
        return {"error": "no page data found"}
    
    raw = match.group(1).replace("&quot;", '"').replace("&amp;", "&").replace("&#39;", "'")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    
    props = data.get("props", {})
    company = props.get("company", {})
    
    founders = []
    for f in company.get("founders", []):
        founders.append({
            "full_name": f.get("full_name", ""),
            "title": f.get("title", ""),
            "founder_bio": f.get("founder_bio", ""),
            "linkedin_url": f.get("linkedin_url", ""),
            "twitter_url": f.get("twitter_url", ""),
        })
    
    partner = company.get("primary_group_partner", {})
    
    return {
        "year_founded": company.get("year_founded"),
        "location": company.get("location", ""),
        "city": company.get("city", ""),
        "country": company.get("country", ""),
        "linkedin_url": company.get("linkedin_url", ""),
        "twitter_url": company.get("twitter_url", ""),
        "github_url": company.get("github_url", ""),
        "founders": founders,
        "num_founders": len(founders),
        "group_partner": partner.get("full_name", "") if partner else "",
        "tags": company.get("tags", []),
    }

def main():
    # Load companies
    with open("/Users/krgoyal/Desktop/YC/yc_companies_raw.json") as f:
        companies = json.load(f)
    
    # Check for existing progress
    output_file = "/Users/krgoyal/Desktop/YC/yc_companies_enriched.json"
    if os.path.exists(output_file):
        with open(output_file) as f:
            enriched = json.load(f)
        done_slugs = {c["slug"] for c in enriched if "founders" in c}
        print(f"Resuming: {len(done_slugs)} already done")
    else:
        enriched = []
        done_slugs = set()
    
    total = len(companies)
    for i, company in enumerate(companies):
        slug = company["slug"]
        if slug in done_slugs:
            continue
        
        print(f"[{i+1}/{total}] {company['name']} ({slug})...", end=" ", flush=True)
        details = fetch_company_page(slug)
        
        merged = {**company, **details}
        enriched.append(merged)
        done_slugs.add(slug)
        
        if "error" in details:
            print(f"ERROR: {details['error']}")
        else:
            nf = details.get("num_founders", 0)
            print(f"OK ({nf} founders)")
        
        # Save progress every 50 companies
        if len(enriched) % 50 == 0:
            with open(output_file, "w") as f:
                json.dump(enriched, f, indent=2)
            print(f"  [Saved progress: {len(enriched)} companies]")
        
        time.sleep(0.25)
    
    # Final save
    with open(output_file, "w") as f:
        json.dump(enriched, f, indent=2)
    
    print(f"\nDone! {len(enriched)} companies enriched with founder data.")
    
    # Quick stats
    total_founders = sum(c.get("num_founders", 0) for c in enriched)
    errors = sum(1 for c in enriched if "error" in c)
    print(f"Total founders found: {total_founders}")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    main()
