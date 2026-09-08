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

IndexBaseOption = Annotated[
    str,
    typer.Option(
        "--index-base",
        help="How to read VOC integer corners: 'one' (default, devkit - 1-based with an "
        "inclusive max) or 'zero' (0-based, exclusive max). The converter checks the "
        "choice against the data and says so if they disagree.",
    ),
]


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
    index_base: IndexBaseOption = "one",
) -> None:
    """Count what a dataset actually contains: videos, frames, boxes, and TRACKS.

    Run this before writing any training code. The motion classifier trains on tracks,
    and a dataset advertising thousands of images may contain none.
    """
    from tayr.datasets.converters.voc import parse_index_base

    try:
        annotated = load_native_directory(
            dataset, fmt=fmt, name=name, split=split, index_base=parse_index_base(index_base)
        )
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
    index_base: IndexBaseOption = "one",
) -> None:
    """Convert native annotations to COCO JSON.

    The output may contain annotations derived from a dataset that grants no
    redistribution rights. Write it outside the repository and never commit it.
    """
    from tayr.datasets.converters.voc import parse_index_base

    try:
        annotated = load_native_directory(
            dataset, fmt=fmt, name=name, split=split, index_base=parse_index_base(index_base)
        )
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


@dataset_app.command("prepare")
def dataset_prepare(
    dataset: Annotated[
        Path, typer.Option("--dataset", help="Dataset root holding the split directories.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Destination root for the detector tree.")],
    fmt: Annotated[str, typer.Option("--format", help="Native format: voc.")] = "voc",
    name: Annotated[str, typer.Option("--name", help="Dataset name for the manifest.")] = "unnamed",
    copy_images: Annotated[
        bool,
        typer.Option(
            "--copy-images",
            help="Copy images instead of symlinking them. Slower and duplicates licensed "
            "imagery on disk; use only where symlinks are unavailable.",
        ),
    ] = False,
    index_base: IndexBaseOption = "one",
) -> None:
    """Build the train/valid/test tree a detector trains from.

    Point `train.dataset_dir` at the output. Images are linked rather than copied, so
    there stays exactly one copy of the licensed imagery on disk.
    """
    from tayr.datasets.converters.voc import parse_index_base
    from tayr.datasets.prepare import prepare_detector_dataset

    try:
        prepared = prepare_detector_dataset(
            dataset,
            destination_root=out,
            fmt=fmt,
            name=name,
            copy_images=copy_images,
            index_base=parse_index_base(index_base),
        )
    except TayrError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(prepared.render())
    typer.secho(
        "  These are derived annotations. Do not commit them or place them anywhere "
        "public unless the dataset's licence permits redistribution.",
        fg=typer.colors.YELLOW,
    )


@dataset_app.command("preview")
def dataset_preview(
    dataset: Annotated[
        Path, typer.Option("--dataset", help="One split directory, e.g. <root>/train.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory to write annotated frames to.")],
    fmt: Annotated[str, typer.Option("--format", help="Native format: voc.")] = "voc",
    name: Annotated[str, typer.Option("--name", help="Dataset name.")] = "unnamed",
    split: Annotated[str, typer.Option("--split", help="Split name for the caption.")] = "train",
    limit: Annotated[int, typer.Option("--limit", help="How many frames to render.")] = 12,
    index_base: IndexBaseOption = "one",
) -> None:
    """Render annotated sample frames so box placement can be checked with an eye.

    The sample is chosen rather than random: the smallest and largest boxes in the split,
    a frame with several objects, and a frame with none. Those are the frames where a
    converter bug shows; twelve random frames are twelve easy ones.
    """
    from tayr.datasets.converters.voc import parse_index_base
    from tayr.datasets.preview import render_previews

    try:
        annotated = load_native_directory(
            dataset, fmt=fmt, name=name, split=split, index_base=parse_index_base(index_base)
        )
        rendered = render_previews(annotated, image_root=dataset, output_dir=out, limit=limit)
    except TayrError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote {len(rendered)} annotated frame(s) to {out}")
    for sample in rendered:
        boxes = sample.video.n_boxes
        typer.echo(
            f"  {sample.output_path.name if sample.output_path else '?':32s} "
            f"{boxes} box(es)   {sample.reason}"
        )
    typer.secho(
        "  Rendered frames contain dataset imagery. Do not commit them.",
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
def train(
    config: ConfigPath,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Name the run directory. Default: timestamped.")
    ] = None,
    synthetic: Annotated[
        bool,
        typer.Option(
            "--synthetic",
            help="Mark the run as consuming placeholder or synthetic data. "
            "The label reaches the manifest, the logs and every report.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Seed, resolve the device, build the RF-DETR kwargs and write the "
            "manifest, but do not train. Checks a config against a real machine before "
            "a rented GPU starts charging.",
        ),
    ] = False,
) -> None:
    """Train a detector from a config.

    Writes the run manifest before the first gradient step, so a run that dies partway
    still leaves behind the commit, seed and config that produced whatever is on disk.
    """
    from tayr.train import train_detector

    try:
        cfg = load_config(config)
        run = train_detector(
            cfg, config_path=config, run_id=run_id, synthetic=synthetic, dry_run=dry_run
        )
    except TayrError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if run.device.fell_back:
        typer.secho(f"  {run.device.note}", fg=typer.colors.YELLOW)
    if dry_run:
        typer.secho("DRY RUN - nothing was trained.", fg=typer.colors.YELLOW)
    typer.echo(f"run      {run.run_id}")
    typer.echo(f"device   {run.device.device} ({run.device.device_name or 'no accelerator'})")
    typer.echo(f"manifest {run.manifest_path}")
    for path in run.checkpoints:
        typer.echo(f"  checkpoint {path.name}")


@app.command("eval")
def evaluate(
    run: Annotated[Path, typer.Option("--run", help="Run directory produced by `tayr train`.")],
    split: Annotated[str, typer.Option("--split", help="Dataset split to evaluate.")] = "test",
    checkpoint: Annotated[
        str | None,
        typer.Option("--checkpoint", help="Checkpoint filename inside the run directory."),
    ] = None,
) -> None:
    """Evaluate a trained run: mAP, pixels-on-target buckets, and false alarms per hour.

    The config comes from the run's own manifest rather than a file passed here, so a
    config that has drifted since training cannot silently produce a number belonging to
    neither version.
    """
    from tayr.eval.harness import evaluate_split, load_run_config, resolve_run_checkpoint

    try:
        cfg, manifest_path = load_run_config(run)
        checkpoint_path, digest = resolve_run_checkpoint(run, name=checkpoint)
        scored = evaluate_split(
            cfg.model_copy(
                update={
                    "detector": cfg.detector.model_copy(
                        update={"checkpoint": checkpoint_path, "checkpoint_sha256": digest}
                    )
                }
            ),
            run_dir=run,
            split=split,
            config_path=manifest_path,
        )
    except TayrError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"checkpoint {checkpoint_path.name}  sha256 {digest[:16]}...")
    typer.echo("")
    typer.echo(scored.report.render())
    typer.echo("")
    typer.echo(
        f"  {scored.n_images} image(s) in {scored.seconds:.1f}s "
        f"({scored.images_per_second:.2f} img/s)"
    )
    typer.echo(f"  report written to {scored.report_path}")


if __name__ == "__main__":
    app()
