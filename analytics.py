"""Analytics module — calculate knowledge base growth, graph connectivity, and category metrics."""

from collections import Counter
from datetime import datetime
from utils import list_raw_captures, list_wiki_notes, read_frontmatter
from config import PARA_CATEGORIES


def get_analytics_data() -> dict:
    """
    Compute comprehensive analytics metrics across raw captures and wiki notes.
    
    Returns:
        dict containing:
        - raw_count: int
        - wiki_count: int
        - total_links: int
        - avg_links_per_note: float
        - knowledge_density: float
        - categories: dict (count per PARA category)
        - category_percents: dict (percentage per PARA category)
        - growth_timeline: list[dict] ({"date": "YYYY-MM-DD", "daily": int, "cumulative": int})
        - top_connected: list[dict] (top 5 notes by link count)
        - top_tags: list[tuple[str, int]] (top 10 tags by frequency)
    """
    raw_files = list_raw_captures()
    wiki_files = list_wiki_notes()
    
    raw_count = len(raw_files)
    wiki_count = len(wiki_files)
    
    category_counts = {cat: 0 for cat in PARA_CATEGORIES}
    total_links = 0
    all_tags = []
    note_details = []
    daily_counts = Counter()
    
    for note_path in wiki_files:
        meta, body = read_frontmatter(note_path)
        note_id = meta.get("id", note_path.stem)
        title = meta.get("title", note_path.stem)
        cat = meta.get("category", "Resources")
        links = meta.get("links", [])
        tags = meta.get("tags", [])
        created_str = meta.get("created", "")
        
        if cat in category_counts:
            category_counts[cat] += 1
        else:
            category_counts["Resources"] += 1
            
        total_links += len(links)
        all_tags.extend(tags)
        
        # Date parsing for growth timeline
        date_key = "Unknown"
        if created_str:
            try:
                dt = datetime.fromisoformat(created_str)
                date_key = dt.strftime("%Y-%m-%d")
            except Exception:
                date_key = created_str[:10] if len(created_str) >= 10 else "Unknown"
                
        daily_counts[date_key] += 1
        
        note_details.append({
            "id": note_id,
            "title": title,
            "category": cat,
            "link_count": len(links),
            "created": date_key
        })

    avg_links = round(total_links / wiki_count, 2) if wiki_count > 0 else 0.0
    density = round(total_links / (wiki_count * (wiki_count - 1)), 3) if wiki_count > 1 else 0.0

    # Category percentages
    cat_percents = {}
    for cat, count in category_counts.items():
        cat_percents[cat] = round((count / wiki_count) * 100, 1) if wiki_count > 0 else 0.0

    # Growth timeline (sorted chronologically with cumulative count)
    sorted_dates = sorted([d for d in daily_counts.keys() if d != "Unknown"])
    growth_timeline = []
    cum_sum = 0
    for d in sorted_dates:
        cnt = daily_counts[d]
        cum_sum += cnt
        growth_timeline.append({"date": d, "daily": cnt, "cumulative": cum_sum})

    if "Unknown" in daily_counts and not growth_timeline:
        growth_timeline.append({"date": "Initial Notes", "daily": daily_counts["Unknown"], "cumulative": daily_counts["Unknown"]})

    # Top connected notes
    note_details.sort(key=lambda x: x["link_count"], reverse=True)
    top_connected = note_details[:5]

    # Top tags
    top_tags = Counter(all_tags).most_common(10)

    return {
        "raw_count": raw_count,
        "wiki_count": wiki_count,
        "total_links": total_links // 2,
        "avg_links_per_note": avg_links,
        "knowledge_density": density,
        "categories": category_counts,
        "category_percents": cat_percents,
        "growth_timeline": growth_timeline,
        "top_connected": top_connected,
        "top_tags": top_tags
    }
