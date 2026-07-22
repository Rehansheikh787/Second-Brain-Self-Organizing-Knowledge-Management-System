"""Build knowledge graph from classified wiki notes and export to graph.json."""

import logging
from pathlib import Path
from datetime import datetime, timezone

from config import WIKI_DIR, GRAPH_JSON, PARA_CATEGORIES
from utils import list_wiki_notes, read_frontmatter, save_json

logger = logging.getLogger(__name__)


def build_graph() -> dict:
    """
    Read all wiki Markdown notes, parse metadata & body,
    construct nodes and deduplicated edges, and calculate node degrees.
    
    Returns: Cytoscape-compatible graph dictionary.
    """
    notes = list_wiki_notes()
    
    nodes = []
    edges_set = set()
    edges = []
    
    # First pass: Collect all node metadata and map note_id to frontmatter
    id_map = {}
    note_data_list = []
    
    for note_path in notes:
        meta, body = read_frontmatter(note_path)
        note_id = meta.get("id", note_path.stem)
        title = meta.get("title", note_path.stem)
        category = meta.get("category", "Resources")
        tags = meta.get("tags", [])
        links = meta.get("links", [])
        
        if category not in PARA_CATEGORIES:
            category = "Resources"
            
        id_map[note_id] = {
            "title": title,
            "category": category,
            "tags": tags,
            "body": body,
            "path": str(note_path.relative_to(WIKI_DIR.parent))
        }
        
        note_data_list.append({
            "id": note_id,
            "title": title,
            "category": category,
            "tags": tags,
            "body": body,
            "links": links
        })
    
    # Calculate degree (link counts) and collect edges
    degree_map = {n["id"]: 0 for n in note_data_list}
    
    for note in note_data_list:
        source_id = note["id"]
        for link in note["links"]:
            target_id = link.get("id")
            similarity = link.get("similarity", 0.5)
            
            if not target_id or target_id not in id_map:
                continue
                
            # Create an undirected unique edge pair key
            edge_key = tuple(sorted([source_id, target_id]))
            
            if edge_key not in edges_set:
                edges_set.add(edge_key)
                edges.append({
                    "data": {
                        "id": f"e_{edge_key[0]}_{edge_key[1]}",
                        "source": edge_key[0],
                        "target": edge_key[1],
                        "weight": similarity
                    }
                })
                degree_map[edge_key[0]] += 1
                degree_map[edge_key[1]] += 1

    # Build node structure for Cytoscape.js
    for note in note_data_list:
        note_id = note["id"]
        nodes.append({
            "data": {
                "id": note_id,
                "label": note["title"],
                "category": note["category"],
                "tags": note["tags"],
                "summary": note["body"][:150] + ("..." if len(note["body"]) > 150 else ""),
                "body": note["body"],
                "degree": degree_map.get(note_id, 0)
            }
        })

    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
    }
    
    return graph_data


def export_graph(output_path: Path = GRAPH_JSON) -> Path:
    """Build graph and save to JSON file."""
    graph = build_graph()
    save_json(output_path, graph)
    logger.info(f"Graph exported to {output_path} ({graph['metadata']['node_count']} nodes, {graph['metadata']['edge_count']} edges)")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Building knowledge graph...")
    path = export_graph()
    print(f"Exported graph to {path}")
