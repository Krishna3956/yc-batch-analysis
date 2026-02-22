"""
Fetch all YC companies from recent batches via Algolia API + scrape individual pages for founder info.
"""
import json
import urllib.request
import time
import ssl
import certifi

ssl_context = ssl.create_default_context(cafile=certifi.where())

APP_ID = "45BWZJ1SGC"
# The base64-encoded key is passed as-is (that's how Algolia secured API keys work)
API_KEY = "ZjA3NWMwMmNhMzEwZmMxOThkZDlkMjFmNDAwNTNjNjdkZjdhNWJkOWRjMThiODQwMjUyZTVkYjA4YjFlMmU2YnJlc3RyaWN0SW5kaWNlcz0lNUIlMjJZQ0NvbXBhbnlfcHJvZHVjdGlvbiUyMiUyQyUyMllDQ29tcGFueV9CeV9MYXVuY2hfRGF0ZV9wcm9kdWN0aW9uJTIyJTVEJnRhZ0ZpbHRlcnM9JTVCJTIyeWNkY19wdWJsaWMlMjIlNUQmYW5hbHl0aWNzVGFncz0lNUIlMjJ5Y2RjJTIyJTVE"

SEARCH_URL = f"https://{APP_ID}-dsn.algolia.net/1/indexes/*/queries"

BATCHES = [
    "Summer 2026", "Spring 2026", "Winter 2026",
    "Fall 2025", "Summer 2025", "Spring 2025", "Winter 2025"
]

def algolia_search(batch, page=0, hits_per_page=100):
    headers = {
        "x-algolia-application-id": APP_ID,
        "x-algolia-api-key": API_KEY,
        "Content-Type": "application/json",
    }
    params = f"query=&page={page}&hitsPerPage={hits_per_page}&facetFilters=%5B%5B%22batch%3A{urllib.parse.quote(batch)}%22%5D%5D"
    body = json.dumps({
        "requests": [{
            "indexName": "YCCompany_production",
            "params": params
        }]
    }).encode("utf-8")
    
    req = urllib.request.Request(SEARCH_URL, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, context=ssl_context) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["results"][0]

import urllib.parse

def fetch_all_companies():
    all_companies = []
    for batch in BATCHES:
        print(f"\n=== {batch} ===")
        page = 0
        while True:
            try:
                result = algolia_search(batch, page=page, hits_per_page=100)
            except Exception as e:
                print(f"  Error page {page}: {e}")
                break
            
            hits = result.get("hits", [])
            total = result.get("nbHits", 0)
            num_pages = result.get("nbPages", 0)
            
            if not hits:
                if page == 0:
                    print(f"  No companies found (batch may not exist yet)")
                break
            
            for h in hits:
                company = {
                    "name": h.get("name", ""),
                    "slug": h.get("slug", ""),
                    "batch": h.get("batch", batch),
                    "one_liner": h.get("one_liner", ""),
                    "long_description": h.get("long_description", ""),
                    "tags": h.get("tags", []),
                    "industries": h.get("industries", []),
                    "subindustry": h.get("subindustry", ""),
                    "industry": h.get("industry", ""),
                    "team_size": h.get("team_size", 0),
                    "all_locations": h.get("all_locations", ""),
                    "regions": h.get("regions", []),
                    "is_hiring": h.get("isHiring", False),
                    "status": h.get("status", ""),
                    "stage": h.get("stage", ""),
                    "website": h.get("website", ""),
                    "top_company": h.get("top_company", False),
                    "nonprofit": h.get("nonprofit", False),
                    "highlight_black": h.get("highlight_black", False),
                    "highlight_latinx": h.get("highlight_latinx", False),
                    "highlight_women": h.get("highlight_women", False),
                    "objectID": h.get("objectID", ""),
                }
                all_companies.append(company)
            
            if page == 0:
                print(f"  Total: {total} companies across {num_pages} pages")
            print(f"  Page {page}: fetched {len(hits)}")
            
            page += 1
            if page >= num_pages:
                break
            time.sleep(0.3)
    
    return all_companies

if __name__ == "__main__":
    companies = fetch_all_companies()
    
    with open("/Users/krgoyal/Desktop/YC/yc_companies_raw.json", "w") as f:
        json.dump(companies, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Total companies: {len(companies)}")
    
    from collections import Counter
    batch_counts = Counter(c["batch"] for c in companies)
    for b in BATCHES:
        print(f"  {b}: {batch_counts.get(b, 0)}")
