import asyncio
import sys
import warnings
from pathlib import Path

from backend import BackendFactory
from cli.context_reader import read_directory_context
from rich.console import Console
from rich.table import Table
from router_agent import route_query
from tools.web_search import web_search

# Suppress noisy third-party warnings for readable test output.
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ddgs")

console = Console()


def _configure_stdio() -> None:
    """Avoid UnicodeEncodeError in legacy Windows terminals."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


async def run_verification():
    console.print("\n[bold purple]PolyReasoner v4 Verification Suite[/bold purple]")
    console.print("[dim]Checking system components, routing, and tool integration...[/dim]\n")

    # Test 1: backend availability.
    backend_table = Table(title="Component Health Check", show_header=True, header_style="cyan")
    backend_table.add_column("Component", style="bold")
    backend_table.add_column("Details", style="yellow")
    backend_table.add_column("Status", style="bold")

    try:
        fast_backend = BackendFactory.create(tier="fast")
        fast_ok = fast_backend.is_available()
        fast_status = "[bold green]PASS[/bold green]" if fast_ok else "[bold red]FAIL[/bold red]"
        backend_table.add_row(
            "Fast/Weak Backend",
            f"{fast_backend.__class__.__name__} ({fast_backend.model_name})",
            fast_status,
        )
    except Exception as e:
        fast_ok = False
        backend_table.add_row("Fast/Weak Backend", f"Error: {e}", "[bold red]FAIL[/bold red]")

    try:
        smart_backend = BackendFactory.create(tier="smart")
        smart_ok = smart_backend.is_available()
        smart_status = "[bold green]PASS[/bold green]" if smart_ok else "[bold red]FAIL[/bold red]"
        backend_table.add_row(
            "Smart/Heavy Backend",
            f"{smart_backend.__class__.__name__} ({smart_backend.model_name})",
            smart_status,
        )
    except Exception as e:
        smart_ok = False
        backend_table.add_row("Smart/Heavy Backend", f"Error: {e}", "[bold red]FAIL[/bold red]")

    console.print(backend_table)
    if not (fast_ok and smart_ok):
        console.print("[bold red][!] One or more backends are unreachable. Check Ollama or API keys.[/bold red]\n")

    # Test 2: intent router classification.
    console.print("\n[bold cyan]Testing Autonomous Intent Router[/bold cyan]")
    test_queries = {
        "Hello! How are you today?": "chat",
        "Should I transition to a Rust backend role or stick to Node.js?": "multiperspective",
        "Search the web for the latest Gemini 1.5 model updates": "search",
        "Explain what pyproject.toml does in this folder": "analyze_folder",
        "Analyze this from a security specialist and a QA engineer perspective: 'A new microservices payment API'": "manual",
    }

    route_table = Table(show_header=True, border_style="dim", box=None)
    route_table.add_column("User Query", style="white")
    route_table.add_column("Expected", style="magenta")
    route_table.add_column("Routed Action", style="yellow")
    route_table.add_column("Status", style="bold")

    for query, expected in test_queries.items():
        try:
            decision = await route_query(query)
            action = decision.get("action", "unknown")
            status = "[bold green]PASS[/bold green]" if action == expected else "[bold yellow]DIFF[/bold yellow]"
            route_table.add_row(f"\"{query}\"", expected, action, status)
        except Exception:
            route_table.add_row(f"\"{query}\"", expected, "ERROR", "[bold red]FAIL[/bold red]")

    console.print(route_table)

    # Test 3: web search plumbing.
    console.print("\n[bold cyan]Testing Web Search (DuckDuckGo Search)[/bold cyan]")
    try:
        results = web_search("Python news", max_results=2)
        if results:
            console.print(f"[bold green]PASS: web search returned {len(results)} results.[/bold green]")
            console.print(f"  [dim]Sample: {results[0]['title']} ({results[0]['url']})[/dim]")
        else:
            console.print("[bold yellow]DIFF: web search returned no results (network-limited or rate-limited).[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]FAIL: web search tool error: {e}[/bold red]")

    # Test 4: file/directory context reader.
    console.print("\n[bold cyan]Testing Context File Reader[/bold cyan]")
    try:
        pyproject_path = Path(__file__).parent / "pyproject.toml"
        context = read_directory_context(str(pyproject_path))
        if context and "File: pyproject.toml" in context:
            console.print(f"[bold green]PASS: context reader parsed pyproject.toml ({len(context)} chars).[/bold green]")
        else:
            console.print("[bold red]FAIL: context reader returned empty or malformed output.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]FAIL: context reader error: {e}[/bold red]")

    console.print("\n[bold purple]Verification Suite Completed.[/bold purple]\n")


if __name__ == "__main__":
    _configure_stdio()
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_verification())
