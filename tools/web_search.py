import sys
import warnings

# Suppress RuntimeWarning from package rename before any import
warnings.filterwarnings("ignore", category=RuntimeWarning)

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Performs a web search using ddgs (or legacy duckduckgo-search) and returns results.
    
    Each result is a dict with: 'title', 'url', 'snippet'
    """
    DDGS = None
    
    # Try the new package name first, then fall back to the old one
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            print("\n[!] Neither 'ddgs' nor 'duckduckgo-search' is installed.")
            print("    Please run: pip install ddgs\n")
            return []

    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
            return results
    except Exception as e:
        print(f"[!] Web Search failed: {e}")
        return []

