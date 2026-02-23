"""Textual TUI for interactive MCP skill evaluation.

Displays eval scenarios in a DataTable with live status updates,
and shows the full agent output for the selected scenario in a
RichLog panel. Inspired by the improver repo's eval_live.py.

Requires: textual >= 3.0.0

Usage (via run_evals.py):
    python -m skills._evals.run_evals --skill quickstart --runner mock --tui
"""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import Any

try:
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import DataTable, Footer, Header, RichLog, Static

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False


@dataclass
class TUIEvent:
    """Event passed from eval thread to TUI."""

    event_type: str  # "start", "complete", "error"
    scenario_id: str
    data: dict[str, Any]


class EvalTUI(App):
    """Textual app for interactive skill eval display."""

    TITLE = "MCP Skill Eval Runner"
    CSS = """
    #main {
        layout: horizontal;
        height: 100%;
    }
    #table-panel {
        width: 55%;
        height: 100%;
    }
    #detail-panel {
        width: 45%;
        height: 100%;
        border-left: solid $accent;
    }
    #summary-bar {
        dock: bottom;
        height: 3;
        padding: 0 2;
        background: $surface;
    }
    DataTable {
        height: 1fr;
    }
    RichLog {
        height: 1fr;
    }
    .status-pass {
        color: $success;
    }
    .status-fail {
        color: $error;
    }
    .status-running {
        color: $warning;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "rerun", "Rerun Selected"),
        Binding("s", "seed", "Seed Project"),
        Binding("enter", "view_detail", "View Detail"),
    ]

    def __init__(
        self,
        scenarios: list[dict],
        runners: list,
        timeout: int = 120,
    ):
        super().__init__()
        self.scenarios = scenarios
        self.runners = runners
        self.timeout = timeout
        self._event_queue: queue.Queue[TUIEvent] = queue.Queue()
        self._records: dict[str, Any] = {}
        self._selected_id: str | None = None
        self._total = 0
        self._completed = 0
        self._passed = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="table-panel"):
                yield DataTable(id="eval-table")
            with Vertical(id="detail-panel"):
                yield Static("Select a scenario to view details", id="detail-header")
                yield RichLog(id="detail-log", highlight=True, markup=True)
        yield Static("Ready", id="summary-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#eval-table", DataTable)
        table.add_columns(
            "Scenario", "Skill", "Runner", "Status",
            "Duration", "Tools", "Scores",
        )
        table.cursor_type = "row"

        for runner in self.runners:
            for scenario in self.scenarios:
                row_key = f"{scenario['id']}-{runner.name}"
                table.add_row(
                    scenario["id"],
                    scenario.get("_skill", "?"),
                    runner.name,
                    "⏳ pending",
                    "-",
                    "-",
                    "-",
                    key=row_key,
                )
                self._total += 1

        self.set_interval(0.2, self._drain_events)
        self._start_eval_thread()

    def _start_eval_thread(self) -> None:
        """Start eval execution in a background thread."""

        def run_evals():
            from skills._evals.run_evals import run_eval_batch

            for runner in self.runners:
                for scenario in self.scenarios:
                    row_key = f"{scenario['id']}-{runner.name}"
                    self._event_queue.put(TUIEvent(
                        event_type="start",
                        scenario_id=row_key,
                        data={"scenario": scenario, "runner": runner.name},
                    ))

                    from skills._evals.runners.base import AgentResult

                    try:
                        result = runner.run(
                            prompt=scenario["user_request"],
                            skill_name=scenario.get("_skill", "quickstart"),
                            timeout=self.timeout,
                        )

                        from skills._evals.run_evals import score_result
                        scores = score_result(scenario, result)

                        self._event_queue.put(TUIEvent(
                            event_type="complete",
                            scenario_id=row_key,
                            data={
                                "result": result,
                                "scores": scores,
                                "passed": all(s.passed for s in scores) if scores else False,
                            },
                        ))
                    except Exception as e:
                        self._event_queue.put(TUIEvent(
                            event_type="error",
                            scenario_id=row_key,
                            data={"error": str(e)},
                        ))

        thread = threading.Thread(target=run_evals, daemon=True)
        thread.start()

    def _drain_events(self) -> None:
        """Process queued events from the eval thread."""
        handled = 0
        while handled < 50:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break

            self._handle_event(event)
            handled += 1

        self._update_summary()

    def _handle_event(self, event: TUIEvent) -> None:
        table = self.query_one("#eval-table", DataTable)

        if event.event_type == "start":
            try:
                row_idx = list(table._row_order).index(event.scenario_id)
                table.update_cell_at((row_idx, 3), "🔄 running")
            except (ValueError, KeyError):
                pass

        elif event.event_type == "complete":
            self._completed += 1
            result = event.data["result"]
            scores = event.data["scores"]
            passed = event.data["passed"]

            if passed:
                self._passed += 1

            self._records[event.scenario_id] = {
                "result": result,
                "scores": scores,
                "passed": passed,
            }

            status = "✅ PASS" if passed else "❌ FAIL"
            tools = ", ".join(result.tools_called[:3]) or "-"
            if len(result.tools_called) > 3:
                tools += f" +{len(result.tools_called) - 3}"
            score_str = " ".join(
                f"{'✓' if s.passed else '✗'}{s.scorer_name[:6]}"
                for s in scores
            )
            duration = f"{result.duration_ms}ms"

            try:
                row_idx = list(table._row_order).index(event.scenario_id)
                table.update_cell_at((row_idx, 3), status)
                table.update_cell_at((row_idx, 4), duration)
                table.update_cell_at((row_idx, 5), tools)
                table.update_cell_at((row_idx, 6), score_str)
            except (ValueError, KeyError):
                pass

        elif event.event_type == "error":
            self._completed += 1
            error = event.data.get("error", "Unknown error")

            try:
                row_idx = list(table._row_order).index(event.scenario_id)
                table.update_cell_at((row_idx, 3), f"💥 {error[:20]}")
            except (ValueError, KeyError):
                pass

    def _update_summary(self) -> None:
        bar = self.query_one("#summary-bar", Static)
        bar.update(
            f"  {self._completed}/{self._total} complete  |  "
            f"{self._passed} passed  |  "
            f"{self._completed - self._passed} failed"
        )

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show details for the selected scenario."""
        row_key = str(event.row_key.value) if event.row_key else None
        if not row_key:
            return

        self._selected_id = row_key
        log = self.query_one("#detail-log", RichLog)
        header = self.query_one("#detail-header", Static)
        log.clear()

        record = self._records.get(row_key)
        if not record:
            header.update(f"[bold]{row_key}[/bold] -- waiting for results...")
            return

        result = record["result"]
        scores = record["scores"]
        passed = record["passed"]

        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        header.update(f"[bold]{row_key}[/bold] -- {status}")

        log.write(f"[bold]Duration:[/bold] {result.duration_ms}ms")
        log.write(f"[bold]Exit code:[/bold] {result.exit_code}")
        log.write(f"[bold]Tools called:[/bold] {', '.join(result.tools_called) or 'none'}")
        log.write(f"[bold]Workflow steps:[/bold] {', '.join(result.workflow_steps) or 'none'}")
        if result.error:
            log.write(f"[bold red]Error:[/bold red] {result.error}")

        log.write("\n[bold underline]Scores[/bold underline]")
        for s in scores:
            icon = "✅" if s.passed else "❌"
            log.write(f"  {icon} {s.scorer_name}: {s.details}")

        log.write("\n[bold underline]Agent Response[/bold underline]")
        response = result.response_text or result.raw_output
        if len(response) > 2000:
            log.write(response[:2000])
            log.write(f"\n... ({len(response) - 2000} chars truncated)")
        else:
            log.write(response or "(empty)")

    def action_rerun(self) -> None:
        self.notify("Rerun not yet implemented", severity="warning")

    def action_seed(self) -> None:
        self.notify("Use --seed flag when starting", severity="information")

    def action_view_detail(self) -> None:
        table = self.query_one("#eval-table", DataTable)
        if table.cursor_row is not None:
            row_key = list(table._row_order)[table.cursor_row]
            self._selected_id = str(row_key)
            self.on_row_selected(DataTable.RowSelected(
                table, table.cursor_row, DataTable.RowKey(row_key),
            ))


def run_tui(scenarios: list[dict], runners: list, timeout: int = 120):
    """Launch the Textual TUI for eval display.

    Args:
        scenarios: List of scenario dicts.
        runners: List of AgentRunner instances.
        timeout: Per-scenario timeout.
    """
    if not TEXTUAL_AVAILABLE:
        raise ImportError(
            "textual is required for TUI mode. Install with: uv pip install 'textual>=3.0.0'"
        )

    app = EvalTUI(scenarios=scenarios, runners=runners, timeout=timeout)
    app.run()
