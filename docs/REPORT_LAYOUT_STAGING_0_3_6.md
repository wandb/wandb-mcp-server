# Report Layout Support for `staging/0.3.6`

## Summary

`create_wandb_report_tool` already has the right primitive for custom Vega
charts through `custom_chart` and `custom_chart_table`, but it cannot yet build
production-style reports that combine narrative sections with groups of charts
sharing a single runset.

This document describes the implementation needed for the `staging/0.3.6`
release so MCP can create full W&B Reports without asking the agent to generate
Python scripts.

The work should extend the existing `panels` input schema in a backward
compatible way. Existing flat `line`, `bar`, `scatter`, `run_comparison`,
`custom_chart`, and `custom_chart_table` panel specs should keep working.

## Current Behavior

The current implementation in `src/wandb_mcp_server/mcp_tools/create_report.py`
has these limitations:

1. `custom_chart` and `custom_chart_table` create one `PanelGrid` per chart.
2. `create_report()` appends all panel blocks under one trailing `wr.H2("Charts")`.
3. `run_ids` are converted into `wr.Runset(query=" ".join(run_ids))`, which is a
   search query, not a deterministic run filter.
4. There is no way to express this structure through MCP:

```python
[
    wr.H2("Average precision"),
    wr.MarkdownBlock("Overview text."),
    wr.PanelGrid(
        runsets=[shared_runset],
        panels=[car_ap, truck_ap, motorcycle_ap],
    ),
    wr.H2("Error analysis"),
    wr.MarkdownBlock("More text."),
    wr.PanelGrid(
        runsets=[shared_runset],
        panels=[error_chart_1, error_chart_2],
    ),
]
```

Customer impact:

- Agents can create individual custom charts.
- Agents cannot assemble full multi-section model-card/report layouts.
- Agents often fall back to writing Python scripts, which defeats the purpose of
  using the MCP report tool.

## Goals

- Support multiple charts in one `PanelGrid`.
- Support interleaved narrative and chart sections.
- Support deterministic run selection for explicit run IDs.
- Preserve existing flat-panel behavior.
- Keep the MCP input shape JSON-serializable and easy for an LLM to emit.
- Add tests that inspect real `wandb_workspaces.reports.v2` objects, not just
  mocks.

## Non-Goals

- Do not add create/update/delete automation behavior in this work.
- Do not introduce a separate scripting tool.
- Do not attempt to support every possible W&B Report block in this release.
- Do not remove existing `markdown_report_text`; it remains useful for simple
  reports.

## Proposed MCP Schema

Keep the `panels` parameter as the report layout surface, but allow it to contain
layout blocks as well as chart blocks.

### Top-Level Layout Blocks

Supported top-level `panels` item types:

```text
heading
markdown
panel_grid
line
bar
scatter
run_comparison
markdown_table
markdown_panel
custom_chart
custom_chart_table
```

`heading` and `markdown` are new.

`panel_grid` is new and contains child panels.

Existing chart types remain valid at the top level. If a chart appears at the
top level, the builder should preserve the current behavior and wrap that single
chart in its own `PanelGrid`.

### `heading`

```json
{
  "type": "heading",
  "level": 2,
  "text": "Average precision by class"
}
```

Rules:

- `level` supports `1`, `2`, and `3`.
- Map to `wr.H1`, `wr.H2`, or `wr.H3`.
- Invalid levels should fall back to `wr.H2`.
- Empty `text` should skip the block.

### `markdown`

```json
{
  "type": "markdown",
  "text": "These charts compare AP across the selected runs."
}
```

Rules:

- Use `wr.MarkdownBlock(text)`.
- This is intentionally separate from `markdown_panel`.
- `markdown` is narrative report content.
- `markdown_panel` remains a chart panel inside a `PanelGrid`.

### `panel_grid`

```json
{
  "type": "panel_grid",
  "title": "Average precision charts",
  "run_ids": ["run_a", "run_b"],
  "hide_run_sets": false,
  "panels": [
    {
      "type": "custom_chart",
      "query": {"summaryTable": {"tableKey": "car_ap"}},
      "chart_name": "cruise/bar_chart/v2",
      "chart_fields": {"x": "threshold", "y": "ap"},
      "chart_strings": {"title": "CAR AP"}
    },
    {
      "type": "custom_chart",
      "query": {"summaryTable": {"tableKey": "truck_ap"}},
      "chart_name": "cruise/bar_chart/v2",
      "chart_fields": {"x": "threshold", "y": "ap"},
      "chart_strings": {"title": "TRUCK AP"}
    }
  ]
}
```

Rules:

- Build exactly one `wr.PanelGrid`.
- Build one shared `wr.Runset`.
- Convert child panel specs into `wr.LinePlot`, `wr.BarPlot`, `wr.ScatterPlot`,
  `wr.CustomChart`, `wr.CustomChart.from_table(...)`, or `wr.MarkdownPanel`.
- Do not allow child `heading`, `markdown`, or nested `panel_grid`.
- If no child panels render successfully, skip the grid.
- If some child panels fail, include a fallback `wr.MarkdownPanel` or skip only
  failed children. Prefer skipping failed children and logging a warning.

## Deterministic Run Selection

The current `run_ids` behavior uses:

```python
wr.Runset(entity=entity, project=project, query=" ".join(run_ids))
```

This is not deterministic. It is a project search query and can match unrelated
runs if a run ID is a substring of another run name or display name.

### Preferred Behavior

For explicit `run_ids`, create a runset filter equivalent to:

```text
run.name in ["run_a", "run_b"]
```

Implementation should prefer a native `wandb-workspaces` filter string that
round-trips through `Runset._to_model()` on `wandb-workspaces>=0.4.2`.

Recommended helper:

```python
def _run_ids_to_filter(run_ids: list[str]) -> str:
    quoted = ", ".join(json.dumps(run_id) for run_id in run_ids)
    return f"name in [{quoted}]"
```

Then:

```python
wr.Runset(entity=entity_name, project=project_name, filters=_run_ids_to_filter(run_ids))
```

If `wandb-workspaces` expects section-qualified run keys, adjust to the canonical
syntax proven by tests. The test should assert the serialized runset contains an
`IN` filter on the run name field.

### Filter Passthrough

Also allow callers to pass `filters` directly:

```json
{
  "type": "panel_grid",
  "filters": "name in [\"run_a\", \"run_b\"]",
  "panels": [...]
}
```

Rules:

- If `filters` is present, it wins over `run_ids`.
- If `analysis_run_id` is present, preserve current behavior unless tests prove
  a deterministic filter is safe.
- Do not attempt to parse arbitrary filters in MCP. Pass the string through to
  `wr.Runset(filters=...)`.

## Implementation Plan

All code changes should stay in:

```text
src/wandb_mcp_server/mcp_tools/create_report.py
tests/test_create_report_panels.py
README.md
```

### 1. Add Layout Mode Detection

Introduce:

```python
_LAYOUT_BLOCK_TYPES = {"heading", "markdown", "panel_grid"}
```

Then:

```python
def _is_layout_mode(panels: list[dict[str, Any]]) -> bool:
    return any(panel.get("type", "").lower() in _LAYOUT_BLOCK_TYPES for panel in panels)
```

In `create_report()`:

- If `panels` is absent, behavior unchanged.
- If `panels` is present and not layout mode, behavior unchanged:
  - build panel blocks
  - prepend one `wr.H2("Charts")`
- If layout mode:
  - build layout blocks in order
  - do not insert `wr.H2("Charts")`
  - append layout blocks directly after markdown content

This keeps backward compatibility for old callers.

### 2. Add `_build_layout_blocks(...)`

```python
def _build_layout_blocks(
    specs: list[dict[str, Any]],
    entity_name: str,
    project_name: str,
) -> list[Any]:
    ...
```

Responsibilities:

- Preserve input order.
- Dispatch `heading`.
- Dispatch `markdown`.
- Dispatch `panel_grid`.
- For existing chart block types at top level, call existing `_build_panel_block`
  so mixed old/new layouts work.
- Log and continue on individual block errors.

### 3. Add `_build_heading_block(...)`

```python
def _build_heading_block(spec: dict[str, Any]):
    text = spec.get("text") or spec.get("title") or ""
    level = int(spec.get("level", 2))
    ...
```

Mapping:

- `1` -> `wr.H1`
- `2` -> `wr.H2`
- `3` -> `wr.H3`

### 4. Add `_build_markdown_block(...)`

```python
def _build_markdown_block(spec: dict[str, Any]):
    text = spec.get("text") or spec.get("markdown") or ""
    return wr.MarkdownBlock(text) if text else None
```

### 5. Split Panel Construction From Grid Wrapping

Current builders return `wr.PanelGrid` for most chart panel specs. For
`panel_grid`, the child builder needs to return only panel objects.

Add:

```python
def _build_panel_object(spec: dict[str, Any]):
    ...
```

It should produce:

- `wr.LinePlot`
- `wr.BarPlot`
- `wr.ScatterPlot`
- `wr.MarkdownPanel`
- `wr.CustomChart`
- `wr.CustomChart.from_table(...)`

Then:

- Existing standalone `_build_native_panel(...)` can wrap the object in
  `_build_panel_grid(...)`.
- New `_build_panel_grid_block(...)` can collect many panel objects and wrap them
  in one `wr.PanelGrid`.

Avoid duplicating custom chart construction logic.

### 6. Add `_build_panel_grid_block(...)`

```python
def _build_panel_grid_block(
    spec: dict[str, Any],
    entity_name: str,
    project_name: str,
):
    runset = _build_runset(spec, entity_name, project_name, include_run_ids=True)
    child_panels = [...]
    return wr.PanelGrid(
        runsets=[runset],
        hide_run_sets=bool(spec.get("hide_run_sets", False)),
        panels=child_panels,
    )
```

Rules:

- `panels` must be a non-empty list.
- Child chart errors should not fail the whole report if at least one child
  succeeds.
- Unknown child block types should be ignored with a warning.
- `title` on the grid is optional and should not automatically create a heading.
  Use an explicit `heading` block for visible titles.

### 7. Update `_build_runset(...)`

Order of precedence:

1. Explicit `filters`
2. Explicit `analysis_run_id`
3. Explicit `run_ids`
4. Default unfiltered runset

Pseudo-code:

```python
filters = panel_spec.get("filters")
if filters:
    return wr.Runset(entity=entity_name, project=project_name, filters=filters)

run_id = panel_spec.get("analysis_run_id")
if run_id:
    return wr.Runset(entity=entity_name, project=project_name, query=run_id)

run_ids = panel_spec.get("run_ids", [])
if include_run_ids and run_ids:
    return wr.Runset(
        entity=entity_name,
        project=project_name,
        filters=_run_ids_to_filter(run_ids),
    )
```

If `filters` as string fails with `wandb-workspaces`, use the v2 dict shape that
the workspaces tests prove serializes correctly. Do not ship an untested filter
shape.

## Example MCP Input

```json
{
  "entity_name": "my-team",
  "project_name": "my-project",
  "title": "Detector model card",
  "markdown_report_text": "# Detector model card\n\n[TOC]",
  "panels": [
    {
      "type": "heading",
      "level": 2,
      "text": "Average precision"
    },
    {
      "type": "markdown",
      "text": "Average precision by class for the selected runs."
    },
    {
      "type": "panel_grid",
      "run_ids": ["run_a", "run_b"],
      "hide_run_sets": false,
      "panels": [
        {
          "type": "custom_chart",
          "query": {"summaryTable": {"tableKey": "car_ap"}},
          "chart_name": "cruise/bar_chart/v2",
          "chart_fields": {"x": "threshold", "y": "ap"},
          "chart_strings": {"title": "CAR AP"}
        },
        {
          "type": "custom_chart",
          "query": {"summaryTable": {"tableKey": "truck_ap"}},
          "chart_name": "cruise/bar_chart/v2",
          "chart_fields": {"x": "threshold", "y": "ap"},
          "chart_strings": {"title": "TRUCK AP"}
        }
      ]
    },
    {
      "type": "heading",
      "level": 2,
      "text": "Error analysis"
    },
    {
      "type": "markdown",
      "text": "False-positive and false-negative breakdowns."
    },
    {
      "type": "panel_grid",
      "filters": "name in [\"run_a\", \"run_b\"]",
      "panels": [
        {
          "type": "custom_chart_table",
          "table_name": "error_table",
          "chart_name": "cruise/error_chart",
          "chart_fields": {"x": "class", "y": "count"},
          "chart_strings": {"title": "Errors by class"}
        }
      ]
    }
  ]
}
```

Expected report block order:

```text
P("*Report created via W&B MCP Server*")
H1("Detector model card")
TableOfContents()
H2("Average precision")
MarkdownBlock(...)
PanelGrid(panels=[CustomChart, CustomChart], runsets=[shared_runset])
H2("Error analysis")
MarkdownBlock(...)
PanelGrid(panels=[CustomChart], runsets=[shared_runset])
```

## Tests

Add or update tests in:

```text
tests/test_create_report_panels.py
```

### Unit Tests

1. `test_layout_heading_block`
   - `{"type": "heading", "level": 2, "text": "Section"}`
   - Returns `wr.H2("Section")`.

2. `test_layout_markdown_block`
   - `{"type": "markdown", "text": "Intro"}`
   - Returns `wr.MarkdownBlock("Intro")`.

3. `test_panel_grid_groups_multiple_custom_charts`
   - One `panel_grid` with two `custom_chart` children.
   - Asserts one `wr.PanelGrid`.
   - Asserts two custom chart panel objects.
   - Asserts one shared runset.

4. `test_layout_preserves_interleaved_block_order`
   - `heading`, `markdown`, `panel_grid`, `heading`, `markdown`, `panel_grid`
   - Asserts exact block order.
   - Asserts no automatic `H2("Charts")` is inserted in layout mode.

5. `test_legacy_flat_panels_still_get_charts_heading`
   - Existing flat `line` panel behavior unchanged.

6. `test_run_ids_use_deterministic_filter`
   - `run_ids=["run_a", "run_b"]`
   - Asserts serialized runset contains an `IN` filter on run name.
   - Do this with real `wandb_workspaces` objects, not only `MagicMock`.

7. `test_explicit_filters_win_over_run_ids`
   - Provide both `filters` and `run_ids`.
   - Assert `filters` is passed through and `run_ids` ignored.

8. `test_unknown_child_panel_does_not_break_grid`
   - One valid child panel and one unknown child.
   - Result grid contains only the valid child.

### Integration-Style Object Test

Use real `wandb_workspaces.reports.v2` objects and call `_to_model()` on the
captured report object to inspect the actual serialized shape:

- `Report.spec.blocks`
- `PanelGrid.metadata.run_sets`
- `PanelGrid.metadata.panel_bank_section_config.panels`
- Custom chart `panel_def_id`
- Runset filter model

This is the test that protects against mismatches between MCP assumptions and
`wandb-workspaces` serialization.

## README Update

Update the `Chart panels` section to mention:

- `panel_grid` groups multiple charts under one shared runset.
- `heading` and `markdown` allow interleaved report sections.
- `run_ids` are deterministic filters, not fuzzy search.
- `custom_chart` supports custom Vega panel definitions, including non-`wandb/*`
  `chart_name` values.

## Tool Description Update

Update `CREATE_WANDB_REPORT_TOOL_DESCRIPTION` in
`src/wandb_mcp_server/mcp_tools/create_report.py`.

Add a `layout_mode` section:

```text
Use panel_grid when multiple charts should share the same runset or filters.
Use heading and markdown blocks to interleave narrative sections with chart grids.
For production reports, prefer:
heading -> markdown -> panel_grid -> heading -> markdown -> panel_grid
```

Add a `run_filtering` section:

```text
Use run_ids for exact run selection. MCP converts run_ids into deterministic
run-name filters. Use filters for advanced W&B runset filter expressions.
```

## Validation

Run:

```bash
uv run pytest tests/test_create_report_panels.py -v
uv run pytest tests/test_markdown_parsing.py tests/test_tool_descriptions.py -v
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv build
```

Fresh wheel smoke:

```bash
uv venv /tmp/wandb-mcp-report-layout-smoke --python 3.12
uv pip install --python /tmp/wandb-mcp-report-layout-smoke/bin/python dist/*.whl
/tmp/wandb-mcp-report-layout-smoke/bin/python -c "import wandb_mcp_server.mcp_tools.create_report; print('ok')"
```

If a live key is available, run a live smoke that creates a report with:

- one `heading`
- one `markdown`
- one `panel_grid`
- two `custom_chart_table` panels
- `run_ids` selecting known runs

Delete the report afterward.

## Rollout for `staging/0.3.6`

1. Land this as a single PR against `staging/0.3.6`.
2. Keep it separate from the automations PR and SDK dependency PR.
3. Verify PR order:
   - SDK/workspaces compatibility branch is already in `staging/0.3.6`.
   - Automations PR can land independently.
   - Report layout PR can land independently.
4. After merge, tell the customer:
   - Custom Vega charts were already supported.
   - Full report composition now supports shared panel grids, interleaved sections,
     and deterministic run filters.

## Risks

- `wandb-workspaces` filter string syntax may not match our assumption.
  - Mitigation: assert the serialized runset model in tests.
- LLMs may overuse layout mode for simple reports.
  - Mitigation: keep flat panel behavior and document when to use `panel_grid`.
- Backward compatibility regression for old `panels` arrays.
  - Mitigation: explicit tests that old behavior still appends `H2("Charts")`.
- Silent chart child failures could produce incomplete reports.
  - Mitigation: log warnings and optionally include a fallback paragraph in the
    report when an entire grid cannot render.

## Acceptance Criteria

- Existing flat `panels` calls still work.
- `panel_grid` supports multiple child charts sharing one runset.
- `heading` and `markdown` blocks interleave correctly with grids.
- `run_ids` serialize to deterministic run filters.
- `filters` passthrough is supported and wins over `run_ids`.
- Real `wandb-workspaces` object serialization tests pass.
- README and tool description explain the new schema.
- Build and focused tests pass.
