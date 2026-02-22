"""
Scrape YC companies from Algolia API for recent batches and save structured data.
"""
import json
import urllib.request
import urllib.parse
import time
import ssl
import certifi

# Fix SSL
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Algolia credentials from YC's public page
ALGOLIA_APP_ID = "45BWZJ1SGC"
ALGOLIA_API_KEY = "f075c02ca310fc198dd9d21f40053c67df7a5bd9dc18b840252e5db08b1e2e6b"

INDEX_NAME = "YCCompany_production"
ALGOLIA_URL = f"https://{ALGOLIA_APP_ID}-dsn.algolia.net/1/indexes/{INDEX_NAME}/query"

BATCHES = [
    "Summer 2026", "Spring 2026", "Winter 2026",
    "Fall 2025", "Summer 2025", "Spring 2025", "Winter 2025"
]

def search_batch(batch, page=0, hits_per_page=100):
    """Search for companies in a specific batch."""
    headers = {
        "X-Algolia-Application-Id": ALGOLIA_APP_ID,
        "X-Algolia-API-Key": ALGOLIA_API_KEY,
        "Content-Type": "application/json"
    }
    
    body = json.dumps({
        "query": "",
        "page": page,
        "hitsPerPage": hits_per_page,
        "facetFilters": [["batch:" + batch]],
        "tagFilters": ["ycdc_public"]
    }).encode('utf-8')
    
    req = urllib.request.Request(ALGOLIA_URL, data=body, headers=headers, method='POST')
    
    try:
        with urllib.request.urlopen(req, context=ssl_context) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data
    except Exception as e:
        print(f"Error fetching batch {batch} page {page}: {e}")
        return None

def fetch_all_companies():
    """Fetch all companies across all target batches."""
    all_companies = []
    
    for batch in BATCHES:
        print(f"\n--- Fetching {batch} ---")
        page = 0
        batch_total = 0
        
        while True:
            result = search_batch(batch, page=page)
            if result is None:
                break
            
            hits = result.get('hits', [])
            total = result.get('nbHits', 0)
            num_pages = result.get('nbPages', 0)
            
            if not hits:
                break
            
            for hit in hits:
                company = {
                    'name': hit.get('name', ''),
                    'slug': hit.get('slug', ''),
                    'batch': hit.get('batch', batch),
                    'one_liner': hit.get('one_liner', ''),
                    'long_description': hit.get('long_description', ''),
                    'tags': hit.get('tags', []),
                    'industries': hit.get('industries', []),  
                    'subindustry': hit.get('subindustry', ''),
                    'industry': hit.get('industry', ''),
                    'team_size': hit.get('team_size', 0),
                    'location': hit.get('location', ''),
                    'country': hit.get('country', ''),
                    'city': hit.get('city', ''),
                    'regions': hit.get('regions', []),
                    'is_hiring': hit.get('isHiring', False),
                    'status': hit.get('status', ''),
                    'stage': hit.get('stage', ''),
                    'url': hit.get('website', ''),
                    'yc_url': f"https://www.ycombinator.com/companies/{hit.get('slug', '')}",
                    'all_locations': hit.get('all_locations', ''),
                    'top_company': hit.get('top_company', False),
                    'nonprofit': hit.get('nonprofit', False),
                    'highlight_black': hit.get('highlight_black', False),
                    'highlight_latinx': hit.get('highlight_latinx', False),
                    'highlight_women': hit.get('highlight_women', False),
                    'app_video_public': hit.get('app_video_public', False),
                    'demo_day_video_public': hit.get('demo_day_video_public', False),
                    'objectID': hit.get('objectID', ''),
                }
                all_companies.append(company)
            
            batch_total += len(hits)
            print(f"  Page {page}: got {len(hits)} hits (total available: {total}, pages: {num_pages})")
            
            page += 1
            if page >= num_pages:
                break
            time.sleep(0.3)
        
        print(f"  Total for {batch}: {batch_total}")
    
    return all_companies

if __name__ == '__main__':
    companies = fetch_all_companies()
    
    # Save raw data
    with open('/Users/krgoyal/Desktop/YC/yc_companies_raw.json', 'w') as f:
        json.dump(companies, f, indent=2)
    
    print(f"\n=== Total companies fetched: {len(companies)} ===")
    
    # Print batch breakdown
    from collections import Counter
    batch_counts = Counter(c['batch'] for c in companies)
    for batch in BATCHES:
        print(f"  {batch}: {batch_counts.get(batch, 0)}")
