from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import DEFAULT_CONFIG_PATH, load_config
from .embeddings import warm_context_embeddings
from .maxdiff import rank_all_relations
from .questions import generate_questions
from .relations import (
    finalize_relations,
    mine_all_relations,
    prepare_relation_review,
)
from .validate import validate_input as validate_input_file
from .validate import validate_outputs
from .wiki import retrieve_all_wikipedia

app = typer.Typer(
    no_args_is_help=True,
    help="Reconstruct the complex-analogy dataset creation pipeline.",
)


def _config(path: Path):
    return load_config(path)


@app.command("validate-input")
def validate_input(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Check the input CSV and summarize base-relation group sizes."""
    report = validate_input_file(_config(config))
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


@app.command("retrieve-wikipedia")
def retrieve_wikipedia(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    resume: bool = typer.Option(True, "--resume/--restart"),
) -> None:
    """Retrieve co-occurrence sentences from English Wikipedia."""
    output = retrieve_all_wikipedia(_config(config), resume=resume)
    typer.echo(f"Wrote {output}")


@app.command("mine-relations")
def mine_relations(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    resume: bool = typer.Option(True, "--resume/--restart"),
) -> None:
    """Use an OpenAI model to mine evidence-grounded relations."""
    output = mine_all_relations(_config(config), resume=resume)
    typer.echo(f"Wrote {output}")


@app.command("prepare-review")
def prepare_review(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Create the human-editable relation filtering CSV."""
    output = prepare_relation_review(_config(config))
    typer.echo(f"Review relations in {output}")
    typer.echo("Fill final_decision with accept or reject before finalizing.")


@app.command("finalize-relations")
def finalize(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    use_suggestions: bool = typer.Option(
        False,
        "--use-suggestions",
        help="Accept automatic suggestions without completing the review CSV.",
    ),
) -> None:
    """Apply review decisions and create canonical relation sets."""
    output = finalize_relations(_config(config), use_suggestions=use_suggestions)
    typer.echo(f"Wrote {output}")


@app.command("rank-relations")
def rank_relations(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    resume: bool = typer.Option(True, "--resume/--restart"),
) -> None:
    """Run GPT-assisted Max-Diff ranking for each pair's relations."""
    output = rank_all_relations(_config(config), resume=resume)
    typer.echo(f"Wrote {output}")


@app.command("embed-contexts")
def embed_contexts(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Cache Wikipedia sentence embeddings."""
    output = warm_context_embeddings(_config(config))
    typer.echo(f"Updated {output}")


@app.command("generate-questions")
def build_questions(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Create four-option questions and their three semantic labels."""
    output = generate_questions(_config(config))
    typer.echo(f"Wrote {output}")


@app.command("validate")
def validate(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Validate generated artifacts and produce dataset statistics."""
    report = validate_outputs(_config(config))
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "ok":
        raise typer.Exit(code=1)


@app.command("run-until-review")
def run_until_review(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Run input validation, Wikipedia retrieval, relation mining, and review export."""
    cfg = _config(config)
    typer.echo(json.dumps(validate_input_file(cfg), indent=2, ensure_ascii=False))
    retrieve_all_wikipedia(cfg, resume=True)
    mine_all_relations(cfg, resume=True)
    output = prepare_relation_review(cfg)
    typer.echo(f"Stopped for manual review: {output}")


@app.command("resume-after-review")
def resume_after_review(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    use_suggestions: bool = typer.Option(False, "--use-suggestions"),
) -> None:
    """Continue from relation filtering through final dataset validation."""
    cfg = _config(config)
    finalize_relations(cfg, use_suggestions=use_suggestions)
    rank_all_relations(cfg, resume=True)
    warm_context_embeddings(cfg)
    generate_questions(cfg)
    report = validate_outputs(cfg)
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "ok":
        raise typer.Exit(code=1)


@app.command("run-auto")
def run_auto(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Run the entire pipeline using automatic relation-filter suggestions."""
    cfg = _config(config)
    typer.echo(json.dumps(validate_input_file(cfg), indent=2, ensure_ascii=False))
    retrieve_all_wikipedia(cfg, resume=True)
    mine_all_relations(cfg, resume=True)
    prepare_relation_review(cfg)
    finalize_relations(cfg, use_suggestions=True)
    rank_all_relations(cfg, resume=True)
    warm_context_embeddings(cfg)
    generate_questions(cfg)
    report = validate_outputs(cfg)
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "ok":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
