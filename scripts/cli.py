from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console
from rich.table import Table

from pipeline.daily_run import run_pipeline
from retrieval.config_loader import load_brand_config, load_settings, load_source_config

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool) -> None:
    """NewsletterAgent — AI-powered newsletter generation pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


@main.command()
def run() -> None:
    """Execute a full pipeline run."""
    console.print("[bold]Starting newsletter pipeline...[/bold]")
    context = asyncio.run(run_pipeline())

    if context.newsletter:
        console.print(
            f"[green]Newsletter generated:[/green] {context.newsletter.edition_id} "
            f"({context.newsletter.total_articles} articles)"
        )
        if context.newsletter.markdown_body:
            console.print("\n" + context.newsletter.markdown_body)
    else:
        console.print("[red]Pipeline completed but no newsletter was generated[/red]")


@main.command()
def sources() -> None:
    """List configured sources."""
    config = load_source_config()
    table = Table(title="Configured Sources")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Reliability")
    table.add_column("URL")

    for feed in config.rss_feeds:
        table.add_row(
            feed.name,
            feed.category,
            f"{feed.reliability:.2f}",
            str(feed.url),
        )

    console.print(table)


@main.command()
def config() -> None:
    """Show current pipeline configuration."""
    settings = load_settings()
    brand = load_brand_config()

    console.print("[bold]Pipeline Settings[/bold]")
    pipeline = settings.get("pipeline", {})
    for key, value in pipeline.items():
        console.print(f"  {key}: {value}")

    console.print("\n[bold]Brand Config[/bold]")
    brand_info = brand.get("brand", {})
    for key, value in brand_info.items():
        console.print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
