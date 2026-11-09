"""
High-performance downloader for ShuttleSet with proper URL encoding.
"""

import os
import json
import urllib.request
import urllib.parse
import time

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "shuttleset")
TREE_API_URL = "https://api.github.com/repos/wywyWang/CoachAI-Projects/git/trees/main?recursive=1"
RAW_BASE_URL = "https://raw.githubusercontent.com/wywyWang/CoachAI-Projects/main/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def download_file(url, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return False  # Already exists
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as resp, open(target_path, "wb") as f:
        f.write(resp.read())
    return True

def download_full_shuttleset():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    req = urllib.request.Request(TREE_API_URL, headers=HEADERS)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))

    tree_items = data.get("tree", [])
    csv_items = [
        it for it in tree_items 
        if it["path"].startswith("ShuttleSet/set/") and it["path"].endswith(".csv")
    ]
    
    downloaded = 0
    skipped = 0

    for i, item in enumerate(csv_items, 1):
        rel_path = os.path.relpath(item["path"], "ShuttleSet/set")
        target_path = os.path.join(RAW_DATA_DIR, rel_path)
        encoded_path = urllib.parse.quote(item["path"])
        raw_url = RAW_BASE_URL + encoded_path
        
        try:
            is_new = download_file(raw_url, target_path)
            if is_new:
                downloaded += 1
                print(f"[{i}/{len(csv_items)}] Downloaded {rel_path}")
            else:
                skipped += 1
        except Exception as e:
            print(f"[{i}/{len(csv_items)}] Error downloading {rel_path}: {e}")

    print(f"\nFinal Summary:")
    print(f"- Downloaded: {downloaded}")
    print(f"- Cached: {skipped}")
    print(f"- Total set files available: {downloaded + skipped} / {len(csv_items)}")

if __name__ == "__main__":
    download_full_shuttleset()
