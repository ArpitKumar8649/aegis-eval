"""Command-line interface for Aegis-Eval."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aegis.config import ConfigError, Settings
from aegis.dataset import compute_file_sha256
from aegis.regression import compare_runs, load_run_summary
from aegis.runner import EvaluationRunner

console = Console()


def print_banner():
    console.print(
        Panel.fit(
            "[bold white]AEGIS-EVAL[/bold white] [dim]· Clinical Safety & Preventive AI Guidance Evaluation Harness[/dim]",
            border_style="cyan",
        )
    )


async def cmd_run(args: argparse.Namespace) -> int:
    try:
        settings = Settings.from_env()
    except ConfigError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        return 1

    if args.concurrency:
        object.__setattr__(settings, "max_concurrency", args.concurrency)

    console.print(f"[bold cyan]Subject Model:[/bold cyan] {settings.subject_model}")
    console.print(f"[bold cyan]Judge Model:[/bold cyan]   {settings.judge_model}")
    console.print(f"[bold cyan]Gateway:[/bold cyan]       {settings.base_url}")
    console.print(f"[bold cyan]Dataset:[/bold cyan]       {args.dataset}\n")

    runner = EvaluationRunner(settings)
    try:
        summary, summary_path = await runner.run_dataset(
            dataset_path=args.dataset,
            runs_dir=args.runs_dir,
            case_id_filter=args.filter,
        )
    except Exception as e:
        console.print(f"[bold red]Run Execution Failed:[/bold red] {e}")
        return 1

    # Render results table
    table = Table(title="Aegis-Eval Run Summary", border_style="cyan")
    table.add_column("Case ID", style="bold white")
    table.add_column("Category", style="cyan")
    table.add_column("Deterministic", justify="center")
    table.add_column("Judge Safety", justify="center")
    table.add_column("Judge Correctness", justify="center")
    table.add_column("Passed", justify="center")
    table.add_column("Latency", justify="right")

    for c in summary.cases:
        det_style = "[green]PASS[/green]" if c.deterministic.passed else "[red]FAIL[/red]"
        safety_str = (
            f"{c.judge.scores.safety}/5"
            if (c.judge and c.judge.scores)
            else ("[red]ERR[/red]" if (c.judge and c.judge.is_error) else "-")
        )
        corr_str = (
            f"{c.judge.scores.correctness}/5"
            if (c.judge and c.judge.scores)
            else ("[red]ERR[/red]" if (c.judge and c.judge.is_error) else "-")
        )
        passed_style = "[bold green]PASS[/bold green]" if c.passed else "[bold red]FAIL[/bold red]"
        table.add_row(
            c.case_id,
            c.category,
            det_style,
            safety_str,
            corr_str,
            passed_style,
            f"{c.total_latency_ms:.0f}ms",
        )

    console.print(table)

    # Metric Cards Panel
    metrics_text = (
        f"[bold]Pass Rate:[/bold] {summary.pass_rate}% ({summary.passed_cases}/{summary.total_cases})\n"
        f"[bold]Deterministic Pass Rate:[/bold] {summary.deterministic_pass_rate}%\n"
        f"[bold]Judge Pass Rate:[/bold] {summary.judge_pass_rate}%\n"
        f"[bold]Mean Clinical Safety:[/bold] {summary.mean_safety} / 5.0\n"
        f"[bold]Mean Correctness:[/bold] {summary.mean_correctness} / 5.0\n"
        f"[bold]Total Tokens:[/bold] {summary.total_tokens_in + summary.total_tokens_out:,} "
        f"(In: {summary.total_tokens_in:,}, Out: {summary.total_tokens_out:,})\n"
        f"[bold]Total Cost:[/bold] ${summary.total_cost_usd:.4f} USD\n"
        f"[bold]Dataset SHA-256:[/bold] {summary.dataset_hash[:16]}...\n"
        f"[bold]Summary Artifact:[/bold] {summary_path}\n"
        f"[bold]Trace Artifact:[/bold] {Path(args.runs_dir) / f'{summary.run_id}.trace.jsonl'}"
    )
    console.print(
        Panel(metrics_text, title="[bold green]Run Metrics[/bold green]", border_style="green")
    )

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    try:
        candidate = load_run_summary(args.candidate)
        baseline = load_run_summary(args.baseline)
    except Exception as e:
        console.print(f"[bold red]Failed to load run summaries:[/bold red] {e}")
        return 1

    diff = compare_runs(
        baseline=baseline,
        candidate=candidate,
        max_safety_drop=args.max_safety_drop,
    )

    table = Table(title="Aegis-Eval Regression Gate Comparison", border_style="cyan")
    table.add_column("Metric", style="bold white")
    table.add_column("Baseline", justify="center")
    table.add_column("Candidate", justify="center")
    table.add_column("Delta", justify="center")

    def color_delta(val: float, higher_is_better: bool = True) -> str:
        if val > 0:
            return f"[green]+{val:.2f}[/green]" if higher_is_better else f"[red]+{val:.2f}[/red]"
        elif val < 0:
            return f"[red]{val:.2f}[/red]" if higher_is_better else f"[green]{val:.2f}[/green]"
        return "0.00"

    table.add_row(
        "Pass Rate",
        f"{diff.baseline_pass_rate}%",
        f"{diff.candidate_pass_rate}%",
        color_delta(diff.candidate_pass_rate - diff.baseline_pass_rate),
    )
    table.add_row(
        "Mean Safety",
        f"{diff.baseline_mean_safety:.2f}",
        f"{diff.candidate_mean_safety:.2f}",
        color_delta(diff.safety_delta),
    )
    table.add_row(
        "Mean Correctness",
        f"{diff.baseline_mean_correctness:.2f}",
        f"{diff.candidate_mean_correctness:.2f}",
        color_delta(diff.correctness_delta),
    )
    table.add_row(
        "Dataset Hash Match",
        "✓",
        "✓" if diff.dataset_hash_match else "[yellow]MISMATCH[/yellow]",
        "Identical" if diff.dataset_hash_match else "Different",
    )

    console.print(table)

    if diff.regressed_case_ids:
        console.print(
            Panel(
                f"[bold red]Regressed Cases ({len(diff.regressed_case_ids)}):[/bold red]\n"
                + "\n".join(f"- [red]{cid}[/red]" for cid in diff.regressed_case_ids),
                border_style="red",
                title="[bold red]Regression Failure[/bold red]",
            )
        )

    if diff.improved_case_ids:
        console.print(
            Panel(
                f"[bold green]Improved Cases ({len(diff.improved_case_ids)}):[/bold green]\n"
                + "\n".join(f"- [green]{cid}[/green]" for cid in diff.improved_case_ids),
                border_style="green",
                title="[bold green]Quality Improvements[/bold green]",
            )
        )

    if diff.is_regression:
        console.print("[bold red]❌ REGRESSION GATE FAILED:[/bold red]")
        for reason in diff.rejection_reasons:
            console.print(f"  • {reason}")
        return 1
    else:
        console.print(
            "[bold green]✅ REGRESSION GATE PASSED:[/bold green] No quality regressions detected."
        )
        return 0


def cmd_hash(args: argparse.Namespace) -> int:
    try:
        checksum = compute_file_sha256(args.file)
        console.print(f"[bold cyan]File:[/bold cyan]   {args.file}")
        console.print(f"[bold cyan]SHA-256:[/bold cyan] [bold green]{checksum}[/bold green]")
        return 0
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return 1


def main() -> None:
    print_banner()
    parser = argparse.ArgumentParser(
        prog="aegis",
        description="Aegis-Eval: Clinical safety and preventive health AI guidance evaluation harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    parser_run = subparsers.add_parser("run", help="Execute an evaluation suite against a dataset")
    parser_run.add_argument(
        "--dataset", "-d", default="datasets/v1.0.0.jsonl", help="Path to JSONL dataset"
    )
    parser_run.add_argument(
        "--runs-dir", default="runs", help="Directory to save run summaries and traces"
    )
    parser_run.add_argument(
        "--filter", "-f", default=None, help="Filter test cases by ID substring"
    )
    parser_run.add_argument(
        "--concurrency", "-c", type=int, default=None, help="Maximum concurrent requests"
    )

    # Compare command
    parser_compare = subparsers.add_parser(
        "compare", help="Compare a candidate run against a baseline"
    )
    parser_compare.add_argument("candidate", help="Path to candidate run JSON summary")
    parser_compare.add_argument(
        "--baseline", "-b", default="baseline.json", help="Path to baseline JSON summary"
    )
    parser_compare.add_argument(
        "--max-safety-drop", type=float, default=0.15, help="Maximum allowable drop in mean safety"
    )

    # Dataset hash command
    parser_hash = subparsers.add_parser(
        "dataset-hash", help="Compute SHA-256 checksum for a dataset file"
    )
    parser_hash.add_argument("file", help="Path to dataset file")

    args = parser.parse_args()

    if args.command == "run":
        exit_code = asyncio.run(cmd_run(args))
        sys.exit(exit_code)
    elif args.command == "compare":
        exit_code = cmd_compare(args)
        sys.exit(exit_code)
    elif args.command == "dataset-hash":
        exit_code = cmd_hash(args)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
