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
from tayr.datasets.census import take_census
from tayr.datasets.coco import to_coco
from tayr.datasets.loader import load_native_directory
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

dataset_app = typer.Typer(
    name="dataset", help="Convert and measure datasets.", no_args_is_help=True
)
app.add_typer(dataset_app)

watch_app = typer.Typer(
    name="watch", help="Tayr Watch: the airspace triage agent.", no_args_is_help=True
)
app.add_typer(watch_app)

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


@dataset_app.command("census")
def dataset_census(
    dataset: Annotated[Path, typer.Option("--dataset", help="Directory of annotation files.")],
    fmt: Annotated[str, typer.Option("--format", help="Native format: dvb | antiuav.")],
    name: Annotated[str, typer.Option("--name", help="Dataset name for the report.")] = "unnamed",
    split: Annotated[str, typer.Option("--split", help="Split name for the report.")] = "all",
) -> None:
    """Count what a dataset actually contains: videos, frames, boxes, and TRACKS.

    Run this before writing any training code. The motion classifier trains on tracks,
    and a dataset advertising thousands of images may contain none.
    """
    try:
        annotated = load_native_directory(dataset, fmt=fmt, name=name, split=split)
    except TayrError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(take_census(annotated).render())


@dataset_app.command("convert")
def dataset_convert(
    dataset: Annotated[Path, typer.Option("--dataset", help="Directory of annotation files.")],
    fmt: Annotated[str, typer.Option("--format", help="Native format: dvb | antiuav.")],
    out: Annotated[Path, typer.Option("--out", help="Output COCO JSON path.")],
    name: Annotated[str, typer.Option("--name", help="Dataset name.")] = "unnamed",
    split: Annotated[str, typer.Option("--split", help="Split name.")] = "all",
) -> None:
    """Convert native annotations to COCO JSON.

    The output may contain annotations derived from a dataset that grants no
    redistribution rights. Write it outside the repository and never commit it.
    """
    try:
        annotated = load_native_directory(dataset, fmt=fmt, name=name, split=split)
    except TayrError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_coco(annotated), indent=2), encoding="utf-8")

    census = take_census(annotated)
    typer.echo(f"Wrote {out}")
    typer.echo(
        f"  {census.n_videos} video(s), {census.n_frames} frame(s), "
        f"{census.n_boxes} box(es), {census.n_tracks} track(s)"
    )
    if not annotated.redistributable:
        typer.secho(
            "  NOTE: this dataset is marked non-redistributable. Do not commit this "
            "file or place it anywhere public.",
            fg=typer.colors.YELLOW,
        )


@watch_app.command("demo")
def watch_demo(
    video: Annotated[Path, typer.Option("--video", help="Video to decode for the demo.")],
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("demo-out"),
    sites: Annotated[Path, typer.Option("--sites", help="Site registry YAML.")] = Path(
        "configs/sites/demo.yaml"
    ),
    site_id: Annotated[
        str, typer.Option("--site", help="Site to evaluate against.")
    ] = "demo-north",
) -> None:
    """Run the end-to-end triage demo: decode, track, triage, notify.

    Everything produced is labelled synthetic. There is no trained detector, so the
    detections are scripted; the decode, tracking, motion features, tool calls, verdict
    rules and audit records are all real.
    """
    from tayr.agent.demo import run_authorized_demo, run_demo

    try:
        unexplained = run_demo(video, output_dir=out, site_registry_path=sites, site_id=site_id)
        authorized = run_authorized_demo(
            video, output_dir=out / "authorized", site_registry_path=sites, site_id=site_id
        )
    except TayrError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(
        "SYNTHETIC: detections were scripted, not detected. No number below describes "
        "real-world detection performance.",
        fg=typer.colors.YELLOW,
    )
    for result in (unexplained, authorized):
        for decision in result.decisions:
            d = decision.decision
            typer.echo("")
            typer.echo(f"  track {decision.track_id}")
            typer.echo(f"    verdict   {d.verdict.value.upper()}  ({d.rule_id})")
            typer.echo(f"    attention {d.attention.value}   uncertainty {d.uncertainty.value}")
            for line in d.rationale:
                typer.echo(f"      - {line}")
            typer.echo(
                f"    tools     {len(decision.tool_calls)} call(s), "
                f"{decision.rounds_used} model round(s)"
            )
            typer.echo(f"    audit     {decision.audit_hash()[:16]}...")
    typer.echo("")
    typer.echo(f"Decisions written to {out}")


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
