"""Tayr CLI.

Owner-mode entry point. Every command is a thin wrapper: it parses arguments, calls
into the `tayr` library, and prints. No business logic lives in this module.

Commands that are not yet implemented raise loudly and name the phase that will
implement them. They do not return placeholder results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from tayr import __version__
from tayr.config import load_config
from tayr.errors import TayrError

app = typer.Typer(
    name="tayr",
    help="Detection, tracking and motion classification of small aerial objects in recorded video.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(
    name="config", help="Inspect and validate run configs.", no_args_is_help=True
)
app.add_typer(config_app)

ConfigPath = Annotated[Path, typer.Option("--config", "-c", help="Path to a run config YAML.")]


@app.command()
def version() -> None:
    """Print the Tayr version."""
    typer.echo(__version__)


@config_app.command("validate")
def config_validate(config: ConfigPath) -> None:
    """Validate a run config against the schema and print the resolved values.

    Unknown keys are rejected, so this also catches typos that would otherwise fall
    back to a default and silently change a result.
    """
    try:
        cfg = load_config(config)
    except TayrError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))


@app.command()
def train(config: ConfigPath) -> None:
    """Train a detector from a config. [Phase 3 - not yet implemented]"""
    load_config(config)  # validate now so a bad config fails before anything else
    raise NotImplementedError(
        "tayr train is implemented in Phase 3 (detection training pipeline). The config "
        "you passed is valid, but no training code exists yet. This command will not "
        "return a placeholder result."
    )


@app.command("eval")
def evaluate(
    run: Annotated[Path, typer.Option("--run", help="Run directory produced by `tayr train`.")],
    split: Annotated[str, typer.Option("--split", help="Dataset split to evaluate.")] = "test",
) -> None:
    """Evaluate a trained run. [Phase 3 - not yet implemented]"""
    raise NotImplementedError(
        f"tayr eval is implemented in Phase 3 (evaluation harness). Requested run={run} "
        f"split={split}. No evaluation code exists yet, and this command will not return "
        "a fabricated metric."
    )


if __name__ == "__main__":
    app()
