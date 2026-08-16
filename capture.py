"""Capture pipeline — one command saves anything to raw/ with timestamp + UUID."""

import uuid as uuid_mod
from datetime import datetime, timezone
from pathlib import Path

from config import RAW_DIR
from utils import compute_content_hash, save_json, list_raw_captures, load_json


import re

class DuplicateError(Exception):
    """Raised when content has already been captured."""
    pass


def clean_extracted_pdf_text(raw_text: str) -> str:
    """
    Clean text extracted from PDFs by normalizing line breaks,
    joining fragmented words/lines, and separating markdown headers.
    """
    if not raw_text:
        return ""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # Fix hyphenated words broken across line breaks (e.g. "Product- \n Manager" -> "ProductManager")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    
    # Strip redundant leading `# PDF Document: ...` or `# PDF ok filename.pdf` headers
    text = re.sub(r"^#\s*PDF\s*(?:Document|ok)?:?\s*[\w\.\s-]+\.(?:pdf|txt|png|jpg|md|json)\b\s*", "", text, flags=re.IGNORECASE).strip()
    
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    paragraphs = []
    current_para = []
    
    for l in lines:
        if l.startswith("#"):
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
            paragraphs.append(l)
        else:
            current_para.append(l)
            
    if current_para:
        paragraphs.append(" ".join(current_para))
        
    cleaned = "\n\n".join(paragraphs).strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned


from urllib.parse import urlparse


def fetch_and_clean_web_article(url: str) -> dict:
    """
    Fetch web article HTML via requests, extract page title and clean article body
    text using BeautifulSoup, stripping navigation, footer, and script clutter.
    """
    clean_url = url.strip()
    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        resp = requests.get(clean_url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.content, "html.parser")
        
        # Strip script, style, nav, footer, header, form elements
        for element in soup(["script", "style", "nav", "footer", "header", "form", "aside"]):
            element.decompose()
            
        # Extract title
        title_el = soup.find("title")
        page_title = title_el.get_text().strip() if title_el else urlparse(clean_url).netloc
        
        # Extract paragraphs & headings from article, main, or body
        target_container = soup.find("article") or soup.find("main") or soup.body or soup
        
        paragraphs = []
        for tag in target_container.find_all(["h1", "h2", "h3", "h4", "p"]):
            t_text = tag.get_text().strip()
            if t_text and len(t_text.split()) > 2:
                if tag.name.startswith("h"):
                    paragraphs.append(f"## {t_text}")
                else:
                    paragraphs.append(t_text)
                    
        cleaned_body = "\n\n".join(paragraphs).strip()
        cleaned_body = re.sub(r"\n{3,}", "\n\n", cleaned_body)
        
        if not cleaned_body or len(cleaned_body.split()) < 8:
            cleaned_body = soup.get_text(separator="\n", strip=True)
            lines = [l for l in cleaned_body.split("\n") if len(l.split()) > 3]
            cleaned_body = "\n\n".join(lines[:40]).strip()
            
        formatted_content = f"# {page_title}\nSource URL: {clean_url}\n\n{cleaned_body}"
        return {
            "title": page_title,
            "content": formatted_content,
            "url": clean_url
        }
        
    except Exception as e:
        domain = urlparse(clean_url).netloc or clean_url
        fallback_content = f"# Web Link: {domain}\nSource URL: {clean_url}\n\nCaptured link reference to {clean_url} (fetch note: {e})."
        return {
            "title": f"Link: {domain}",
            "content": fallback_content,
            "url": clean_url
        }


def capture(content: str, source_type: str, original_filename: str = None) -> str:
    """
    Validate input, generate UUID + timestamp, compute content hash,
    check for duplicates, write to raw/<uuid>.json.
    
    Returns: the UUID string of the created capture.
    Raises: ValueError for empty/invalid input, DuplicateError for hash collision.
    """
    # Validate source type
    valid_types = ["note", "link", "file"]
    if source_type not in valid_types:
        raise ValueError(f"source_type must be one of {valid_types}, got '{source_type}'")
    
    # Validate content
    if not content or not content.strip():
        raise ValueError("Content cannot be empty")
    
    # If source_type is "link", scrape web article content
    if source_type == "link":
        web_data = fetch_and_clean_web_article(content.strip())
        content = web_data["content"]
        if not original_filename:
            original_filename = web_data["url"]
    
    # If source_type is "file"
    elif source_type == "file":
        try:
            file_path = Path(content)
            if len(content) < 512 and "\n" not in content:
                if file_path.exists() and file_path.is_file():
                    original_filename = file_path.name
                    if file_path.suffix.lower() == ".pdf":
                        try:
                            import pypdf
                            reader = pypdf.PdfReader(str(file_path))
                            text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
                            extracted = clean_extracted_pdf_text("\n\n".join(text_parts))
                            content = extracted if extracted else f"PDF Document: {file_path.name} (no extractable text)"
                        except Exception as pdf_err:
                            content = f"PDF Document: {file_path.name} (extraction failed: {pdf_err})"
                    else:
                        try:
                            file_content = file_path.read_text(encoding="utf-8")
                        except UnicodeDecodeError:
                            file_content = file_path.read_text(encoding="latin-1")
                        content = file_content
                elif not original_filename and (file_path.suffix or "/" in content or "\\" in content):
                    raise FileNotFoundError(f"File not found: {content}")
        except FileNotFoundError:
            raise
        except Exception:
            pass
            
    # Compute content hash
    content_hash = compute_content_hash(content)
    
    # Check for duplicates
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for existing_file in list_raw_captures():
        existing_data = load_json(existing_file)
        if existing_data.get("metadata", {}).get("content_hash") == content_hash:
            raise DuplicateError(
                f"Duplicate content detected. Existing capture: {existing_data['id']}"
            )
    
    # Generate UUID and timestamp
    capture_id = str(uuid_mod.uuid4())
    timestamp = datetime.now(timezone.utc).astimezone().isoformat()
    
    # Build capture data
    data = {
        "id": capture_id,
        "created": timestamp,
        "source_type": source_type,
        "content": content,
        "metadata": {
            "original_filename": original_filename,
            "char_count": len(content),
            "content_hash": content_hash
        }
    }
    
    # Write to file
    output_path = RAW_DIR / f"{capture_id}.json"
    save_json(output_path, data)
    
    return capture_id

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Capture a note, link, or file into Second Brain")
    parser.add_argument("--type", "-t", required=True, choices=["note", "link", "file"],
                        help="Type of content to capture")
    parser.add_argument("--content", "-c", required=True,
                        help="The content to capture (text, URL, or file path)")
    
    args = parser.parse_args()
    
    try:
        capture_id = capture(args.content, args.type)
        print(f"SUCCESS: Captured! ID: {capture_id}")
        print(f"         Saved to: raw/{capture_id}.json")
    except DuplicateError as e:
        print(f"WARNING: Duplicate: {e}")
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
