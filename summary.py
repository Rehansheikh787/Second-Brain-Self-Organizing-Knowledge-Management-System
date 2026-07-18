"""Helper script to display a summary of captured and classified Second Brain notes."""

from pathlib import Path
from collections import Counter
from utils import list_raw_captures, list_wiki_notes, read_frontmatter
import config

def display_summary():
    raw_files = list_raw_captures()
    wiki_files = list_wiki_notes()
    
    print("=" * 60)
    print("SECOND BRAIN DATABASE SUMMARY")
    print("=" * 60)
    print(f"Total Raw Captures (in raw/): {len(raw_files)}")
    print(f"Total Wiki Notes (in wiki/):   {len(wiki_files)}")
    print("-" * 60)
    
    if not wiki_files:
        print("No wiki notes found yet. Run classify.py first!")
        return
        
    # Count by category
    category_counts = Counter()
    total_links = 0
    note_details = []
    
    # Track title-by-id mapping for pretty link printing
    id_to_title = {}
    for filepath in wiki_files:
        meta, _ = read_frontmatter(filepath)
        note_id = meta.get("id")
        title = meta.get("title", filepath.stem)
        if note_id:
            id_to_title[note_id] = title
            
    for filepath in wiki_files:
        meta, _ = read_frontmatter(filepath)
        category = meta.get("category", "Uncategorized")
        category_counts[category] += 1
        
        links = meta.get("links", [])
        total_links += len(links)
        
        tags = meta.get("tags", [])
        note_details.append({
            "title": meta.get("title", filepath.stem),
            "category": category,
            "tags": tags,
            "links": [l["id"] for l in links]
        })
        
    print("PARA Category Breakdown:")
    for cat in config.PARA_CATEGORIES:
        count = category_counts.get(cat, 0)
        print(f"  - {cat:<10}: {count} notes")
    print(f"  - Uncategorized: {category_counts.get('Uncategorized', 0)} notes")
    print(f"Total Bidirectional Links: {total_links // 2}")
    print("=" * 60)
    
    print(f"{'TITLE':<35} | {'CATEGORY':<10} | {'LINKS TO'}")
    print("-" * 60)
    for detail in note_details:
        linked_titles = [id_to_title.get(lid, lid[:8]) for lid in detail["links"]]
        links_str = ", ".join(linked_titles) if linked_titles else "None"
        
        title_truncated = detail["title"][:33] + ".." if len(detail["title"]) > 35 else detail["title"]
        print(f"{title_truncated:<35} | {detail['category']:<10} | {links_str}")
    print("=" * 60)

if __name__ == "__main__":
    display_summary()
