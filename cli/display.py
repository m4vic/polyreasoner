import json

from rich.box import DOUBLE, ROUNDED
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


class CLI:
    """Terminal presentation helpers for PolyReasoner."""

    @staticmethod
    def print_banner():
        banner_text = Text()
        banner_text.append("\n PolyReasoner v4 ", style="bold violet")
        banner_text.append("- Multi-Perspective Decision and Safety Engine\n", style="italic cyan")
        banner_text.append(" Multi-perspective concurrency | Dynamic backends | Continuous alignment\n", style="grey70")

        panel = Panel(
            banner_text,
            box=ROUNDED,
            border_style="purple",
            padding=(1, 2),
            expand=False,
            width=90,
            subtitle="[italic white]Type /help for commands[/]",
            subtitle_align="right",
        )
        console.print(panel)

    @staticmethod
    def print_synthesis(markdown_text: str, query: str):
        md = Markdown(markdown_text)
        panel = Panel(
            md,
            title=f"[bold green]Synthesis: {query[:50]}...[/]",
            box=DOUBLE,
            border_style="bright_magenta",
            padding=(1, 3),
            expand=False,
            width=90,
        )
        console.print(panel)

    @staticmethod
    def print_settings(settings: dict):
        table = Table(box=ROUNDED, show_header=False, border_style="cyan")
        table.add_column("Key", style="bold yellow")
        table.add_column("Value", style="white")

        table.add_row("Default Backend", settings.get("POLYREASONER_BACKEND", "ollama").upper())
        table.add_row("Ollama Host", settings.get("OLLAMA_HOST", "N/A"))
        table.add_row("Ollama Model (Default)", settings.get("OLLAMA_MODEL", "N/A"))
        table.add_row("Ollama Model (Fast/Weak)", settings.get("OLLAMA_FAST_MODEL", "N/A"))
        table.add_row("Ollama Model (Smart/Heavy)", settings.get("OLLAMA_SMART_MODEL", "N/A"))
        table.add_row("LiteLLM Model (Default)", settings.get("LITELLM_MODEL", "N/A"))
        table.add_row("LiteLLM Model (Fast/Weak)", settings.get("API_FAST_MODEL", "N/A"))
        table.add_row("LiteLLM Model (Smart/Heavy)", settings.get("API_SMART_MODEL", "N/A"))
        table.add_row("Shield Level", str(settings.get("SHIELD_LEVEL", 5)))

        api_keys = settings.get("API_KEYS", {})
        openai_key = "[green]Configured[/]" if api_keys.get("OPENAI_API_KEY") else "[red]Not Set[/]"
        anthropic_key = "[green]Configured[/]" if api_keys.get("ANTHROPIC_API_KEY") else "[red]Not Set[/]"
        gemini_key = "[green]Configured[/]" if api_keys.get("GEMINI_API_KEY") else "[red]Not Set[/]"

        table.add_row("OpenAI Key", openai_key)
        table.add_row("Anthropic Key", anthropic_key)
        table.add_row("Gemini Key", gemini_key)

        panel = Panel(
            table,
            title="[bold yellow]Active Persistent Settings[/]",
            box=ROUNDED,
            border_style="yellow",
            padding=(1, 1),
            expand=False,
            width=90,
        )
        console.print(panel)

    @staticmethod
    def print_agent_status(agent_results: list):
        table = Table(title="Active Agent Panel", box=ROUNDED, border_style="blue")
        table.add_column("Agent / Perspective", style="bold cyan")
        table.add_column("Status", style="bold")

        for res in agent_results:
            agent_name = res.get("agent", "unknown").upper()
            if "error" in res:
                table.add_row(agent_name, "[bold red]Failed: " + str(res["error"]) + "[/]")
            else:
                table.add_row(agent_name, "[bold green]Complete[/]")

        console.print(table)

    @staticmethod
    def print_json(data: dict):
        console.print(json.dumps(data, indent=2))
