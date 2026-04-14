"""List, inspect, and compare W&B artifact versions.

Provides three read-only tools for artifact discovery, inspection, and
comparison via the ``wandb.Api`` public interface.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)

DEFAULT_MAX_ITEMS = 50
MAX_ITEMS_CEILING = 200
MAX_USED_BY = 20


# ---------------------------------------------------------------------------
# Tool 1 – list_artifact_versions
# ---------------------------------------------------------------------------

LIST_ARTIFACT_VERSIONS_TOOL_DESCRIPTION = """List versions of an artifact collection from a project or registry.

Returns version numbers, aliases, tags, sizes, and state for each version.

<when_to_use>
Call this when the user wants to see available versions of a model, dataset, or
other artifact collection. Works with both project-level artifacts and registry
collections. Use the version identifiers in get_artifact_details_tool or
compare_artifact_versions_tool.

Typical workflow:
1. list_registries_tool → list_registry_collections_tool → list_artifact_versions_tool
   (for registry artifacts)
2. query_wandb_entity_projects → list_artifact_versions_tool
   (for project-level artifacts)
</when_to_use>

<critical_info>
For source="project", the W&B SDK requires a fully qualified artifact path.
You can EITHER pass collection_name as "entity/project/artifact-name" OR pass
entity_name + project_name separately and a bare collection_name.
For source="registry", just pass the unqualified collection name.
</critical_info>

Parameters
----------
collection_name : str
    For project source: "entity/project/artifact-name" (fully qualified) or
    just "artifact-name" if entity_name and project_name are also provided.
    For registry source: unqualified collection name (e.g., "sentiment-classifier").
entity_name : str, optional
    W&B entity. Used to qualify collection_name when source="project" and
    collection_name is not already fully qualified.
project_name : str, optional
    W&B project. Used to qualify collection_name when source="project" and
    collection_name is not already fully qualified.
registry_name : str, optional
    Registry name. Required when source="registry".
organization : str, optional
    W&B organization name. Only used when source="registry".
type_name : str, optional
    Artifact type (e.g., "model", "dataset"). Required for project source.
source : str, optional
    "project" (default) or "registry". Determines the API path.
max_items : int, optional
    Maximum versions to return. Default: 50, max: 200.

Returns
-------
JSON with:
  - collection: the queried collection name
  - source: "project" or "registry"
  - versions: list of version objects with version, aliases, tags, size, etc.
  - count: number of versions returned
  - truncated: whether more versions exist beyond max_items
"""


def list_artifact_versions(
    collection_name: str,
    entity_name: Optional[str] = None,
    project_name: Optional[str] = None,
    registry_name: Optional[str] = None,
    organization: Optional[str] = None,
    type_name: Optional[str] = None,
    source: str = "project",
    max_items: int = DEFAULT_MAX_ITEMS,
) -> str:
    """List versions of an artifact collection."""

    api = WandBApiManager.get_api()
    with track_tool_execution(
        "list_artifact_versions",
        api.viewer,
        {
            "collection_name": collection_name,
            "entity_name": entity_name,
            "project_name": project_name,
            "registry_name": registry_name,
            "type_name": type_name,
            "source": source,
            "max_items": max_items,
        },
    ) as ctx:
        max_items = min(max_items, MAX_ITEMS_CEILING)

        try:
            if source == "registry":
                if not registry_name:
                    return json.dumps(
                        {
                            "error": "invalid_input",
                            "message": "registry_name is required when source='registry'",
                        }
                    )
                reg_kwargs: Dict[str, Any] = {}
                if organization is not None:
                    reg_kwargs["organization"] = organization
                registry = api.registry(registry_name, **reg_kwargs)
                versions_iter = registry.collections(
                    filter={"name": collection_name},
                    per_page=min(max_items, 100),
                ).versions()
            else:
                if not type_name:
                    return json.dumps(
                        {
                            "error": "invalid_input",
                            "message": "type_name is required when source='project'",
                        }
                    )
                qualified_name = collection_name
                if "/" not in collection_name and entity_name and project_name:
                    qualified_name = f"{entity_name}/{project_name}/{collection_name}"
                versions_iter = api.artifacts(
                    type_name=type_name,
                    name=qualified_name,
                    per_page=min(max_items, 100),
                )

            versions: List[Dict[str, Any]] = []
            truncated = False
            for art in versions_iter:
                if len(versions) >= max_items:
                    truncated = True
                    break
                versions.append(_serialize_artifact_summary(art))

            return json.dumps(
                {
                    "collection": collection_name,
                    "source": source,
                    "versions": versions,
                    "count": len(versions),
                    "truncated": truncated,
                }
            )

        except Exception as e:
            logger.error(f"Error in list_artifact_versions: {e}", exc_info=True)
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "api_error", "message": str(e)[:500]})


# ---------------------------------------------------------------------------
# Tool 2 – get_artifact_details
# ---------------------------------------------------------------------------

GET_ARTIFACT_DETAILS_TOOL_DESCRIPTION = """Get full details for a specific artifact version including metadata and lineage.

Returns metadata, aliases, tags, and lineage (which run created it, which runs
consumed it, linked artifacts) for a specific artifact version.

<when_to_use>
Call this when the user wants to inspect a specific artifact version: its metadata,
who created it, what depends on it, or its file contents.

For comparing two versions side-by-side, use compare_artifact_versions_tool instead.
</when_to_use>

<critical_info>
artifact_name must include a version or alias qualifier:
  - "entity/project/name:v3" (by version number)
  - "entity/project/name:latest" (by alias)
Without a qualifier, the API will default to "latest".
Lineage calls (logged_by, used_by) make additional network requests and may
return null if the source run has been deleted.
</critical_info>

Parameters
----------
artifact_name : str
    Fully qualified artifact name with version or alias
    (e.g., "my-team/my-project/my-model:v3").
type_name : str, optional
    Artifact type hint for disambiguation.
include_files : bool, optional
    Whether to include the file manifest. Default: False.
max_files : int, optional
    Maximum files to include when include_files is True. Default: 50.

Returns
-------
JSON with:
  - artifact: core properties (name, type, version, state, metadata, etc.)
  - lineage: logged_by run, used_by runs, source_artifact, linked_artifacts
  - files: (only when include_files=True) list of {name, size, digest}
"""


def get_artifact_details(
    artifact_name: str,
    type_name: Optional[str] = None,
    include_files: bool = False,
    max_files: int = 50,
) -> str:
    """Get full details for a specific artifact version."""

    api = WandBApiManager.get_api()
    with track_tool_execution(
        "get_artifact_details",
        api.viewer,
        {
            "artifact_name": artifact_name,
            "type_name": type_name,
            "include_files": include_files,
        },
    ) as ctx:
        try:
            artifact = api.artifact(artifact_name, type=type_name)

            result: Dict[str, Any] = {
                "artifact": {
                    "id": getattr(artifact, "id", None),
                    "name": getattr(artifact, "name", None),
                    "type": getattr(artifact, "type", None),
                    "version": getattr(artifact, "version", None),
                    "state": getattr(artifact, "state", None),
                    "description": getattr(artifact, "description", None),
                    "size": getattr(artifact, "size", None),
                    "file_count": getattr(artifact, "file_count", None),
                    "tags": getattr(artifact, "tags", []),
                    "aliases": getattr(artifact, "aliases", []),
                    "metadata": getattr(artifact, "metadata", {}),
                    "created_at": str(getattr(artifact, "created_at", "")),
                    "digest": getattr(artifact, "digest", None),
                    "commit_hash": getattr(artifact, "commit_hash", None),
                },
                "lineage": _build_lineage(artifact),
            }

            if include_files:
                result["files"] = _list_files(artifact, max_files)

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error in get_artifact_details: {e}", exc_info=True)
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "api_error", "message": str(e)[:500]})


# ---------------------------------------------------------------------------
# Tool 3 – compare_artifact_versions
# ---------------------------------------------------------------------------

COMPARE_ARTIFACT_VERSIONS_TOOL_DESCRIPTION = """Compare two artifact versions side-by-side.

Shows metadata diff, alias/tag changes, size delta, file changes, and lineage
differences between two artifact versions.

<when_to_use>
Call this when the user asks "what changed between version X and version Y" of
a model, dataset, or any artifact. Both versions can be from the same collection
or from different collections/projects.
</when_to_use>

<critical_info>
Both artifact names must include version or alias qualifiers.
The diff is directional: artifact_name_a is the "before" and artifact_name_b is
the "after".
File-level diff compares file names and digests. Set include_file_diff=False to
skip file comparison for large artifacts with many files.
For artifacts with more than 1000 files, only the first 1000 files per artifact
are scanned — files beyond that limit will not appear in the diff. The
"truncated" flag in file_diff is set when this occurs.
</critical_info>

Parameters
----------
artifact_name_a : str
    Fully qualified name of the "before" artifact (e.g., "team/proj/model:v3").
artifact_name_b : str
    Fully qualified name of the "after" artifact (e.g., "team/proj/model:v7").
type_name : str, optional
    Artifact type hint for disambiguation.
include_file_diff : bool, optional
    Whether to compute file-level diff. Default: True.
max_file_diff_entries : int, optional
    Maximum file diff entries to report. Default: 50.

Returns
-------
JSON with:
  - metadata_diff: keys added, removed, changed between versions
  - tags_diff / aliases_diff: set differences
  - size_diff / file_count_diff: numeric deltas with percent change
  - file_diff: files added, removed, modified (by digest), unchanged count
  - lineage_diff: which runs created each version
  - digest_match: whether the two versions have identical content digests
"""


def compare_artifact_versions(
    artifact_name_a: str,
    artifact_name_b: str,
    type_name: Optional[str] = None,
    include_file_diff: bool = True,
    max_file_diff_entries: int = 50,
) -> str:
    """Compare two artifact versions side-by-side."""

    api = WandBApiManager.get_api()
    with track_tool_execution(
        "compare_artifact_versions",
        api.viewer,
        {
            "artifact_name_a": artifact_name_a,
            "artifact_name_b": artifact_name_b,
            "type_name": type_name,
        },
    ) as ctx:
        try:
            art_a = api.artifact(artifact_name_a, type=type_name)
            art_b = api.artifact(artifact_name_b, type=type_name)

            meta_a = getattr(art_a, "metadata", {}) or {}
            meta_b = getattr(art_b, "metadata", {}) or {}

            tags_a = set(getattr(art_a, "tags", []) or [])
            tags_b = set(getattr(art_b, "tags", []) or [])

            aliases_a = set(getattr(art_a, "aliases", []) or [])
            aliases_b = set(getattr(art_b, "aliases", []) or [])

            size_a = getattr(art_a, "size", 0) or 0
            size_b = getattr(art_b, "size", 0) or 0

            fc_a = getattr(art_a, "file_count", 0) or 0
            fc_b = getattr(art_b, "file_count", 0) or 0

            logged_by_a = _get_logged_by(art_a)
            logged_by_b = _get_logged_by(art_b)

            result: Dict[str, Any] = {
                "artifact_a": artifact_name_a,
                "artifact_b": artifact_name_b,
                "metadata_diff": _compute_metadata_diff(meta_a, meta_b),
                "tags_diff": {
                    "added": sorted(tags_b - tags_a),
                    "removed": sorted(tags_a - tags_b),
                },
                "aliases_diff": {
                    "added": sorted(aliases_b - aliases_a),
                    "removed": sorted(aliases_a - aliases_b),
                },
                "size_diff": {
                    "a": size_a,
                    "b": size_b,
                    "delta": size_b - size_a,
                    "percent_change": round((size_b - size_a) / size_a * 100, 2) if size_a else None,
                },
                "file_count_diff": {
                    "a": fc_a,
                    "b": fc_b,
                    "delta": fc_b - fc_a,
                },
                "lineage_diff": {
                    "logged_by_a": logged_by_a,
                    "logged_by_b": logged_by_b,
                    "same_source_run": logged_by_a is not None
                    and logged_by_b is not None
                    and (logged_by_a or {}).get("run_id") == (logged_by_b or {}).get("run_id"),
                },
                "digest_match": getattr(art_a, "digest", None) == getattr(art_b, "digest", None),
            }

            if include_file_diff:
                result["file_diff"] = _compute_file_diff(art_a, art_b, max_file_diff_entries)

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error in compare_artifact_versions: {e}", exc_info=True)
            ctx.mark_error(f"{type(e).__name__}: {e}")
            return json.dumps({"error": "api_error", "message": str(e)[:500]})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _serialize_artifact_summary(artifact: Any) -> Dict[str, Any]:
    """Extract core properties from an Artifact into a plain dict."""
    return {
        "version": getattr(artifact, "version", None),
        "name": getattr(artifact, "name", None),
        "aliases": getattr(artifact, "aliases", []),
        "tags": getattr(artifact, "tags", []),
        "state": getattr(artifact, "state", None),
        "size": getattr(artifact, "size", None),
        "file_count": getattr(artifact, "file_count", None),
        "description": getattr(artifact, "description", None),
        "created_at": str(getattr(artifact, "created_at", "")),
        "digest": getattr(artifact, "digest", None),
    }


def _serialize_run_info(run: Any) -> Optional[Dict[str, Any]]:
    """Extract run identity, or None."""
    if run is None:
        return None
    return {
        "run_id": getattr(run, "id", None),
        "run_name": getattr(run, "name", None),
        "project": getattr(run, "project", None),
        "entity": getattr(run, "entity", None),
    }


def _get_logged_by(artifact: Any) -> Optional[Dict[str, Any]]:
    """Safely call artifact.logged_by()."""
    try:
        run = artifact.logged_by()
        return _serialize_run_info(run)
    except Exception:
        logger.debug("logged_by() failed", exc_info=True)
        return None


def _build_lineage(artifact: Any) -> Dict[str, Any]:
    """Build the lineage section for get_artifact_details."""
    logged_by = _get_logged_by(artifact)

    used_by: Optional[List[Dict[str, Any]]] = None
    used_by_truncated = False
    try:
        runs = artifact.used_by()
        if runs is not None:
            used_by = []
            for run in runs:
                if len(used_by) >= MAX_USED_BY:
                    used_by_truncated = True
                    break
                used_by.append(_serialize_run_info(run))
    except Exception:
        logger.debug("used_by() failed", exc_info=True)

    source_artifact = None
    try:
        src = artifact.source_artifact
        if src is not None and src is not artifact:
            source_artifact = getattr(src, "source_qualified_name", None) or getattr(src, "name", None)
    except Exception:
        logger.debug("source_artifact failed", exc_info=True)

    linked: Optional[List[str]] = None
    try:
        linked_arts = artifact.linked_artifacts
        if linked_arts is not None:
            linked = []
            for la in linked_arts:
                name = getattr(la, "source_qualified_name", None) or getattr(la, "name", None)
                if name:
                    linked.append(name)
    except Exception:
        logger.debug("linked_artifacts failed", exc_info=True)

    lineage: Dict[str, Any] = {
        "logged_by": logged_by,
        "used_by": used_by,
        "source_artifact": source_artifact,
        "linked_artifacts": linked,
    }
    if used_by_truncated:
        lineage["used_by_truncated"] = True
    return lineage


def _list_files(artifact: Any, max_files: int) -> List[Dict[str, Any]]:
    """List files in an artifact, capped at max_files."""
    files: List[Dict[str, Any]] = []
    try:
        for f in artifact.files():
            if len(files) >= max_files:
                break
            files.append(
                {
                    "name": getattr(f, "name", None),
                    "size": getattr(f, "size", None),
                    "digest": getattr(f, "digest", None),
                }
            )
    except Exception:
        logger.debug("files() failed", exc_info=True)
    return files


def _compute_metadata_diff(meta_a: Dict, meta_b: Dict) -> Dict[str, Any]:
    """Key-by-key diff between two metadata dicts."""
    keys_a, keys_b = set(meta_a.keys()), set(meta_b.keys())
    added = {k: meta_b[k] for k in sorted(keys_b - keys_a)}
    removed = {k: meta_a[k] for k in sorted(keys_a - keys_b)}
    changed = {}
    for k in sorted(keys_a & keys_b):
        if meta_a[k] != meta_b[k]:
            changed[k] = {"a": meta_a[k], "b": meta_b[k]}
    return {"added": added, "removed": removed, "changed": changed}


MAX_FILES_FOR_DIFF = 1000


def _compute_file_diff(art_a: Any, art_b: Any, max_entries: int) -> Dict[str, Any]:
    """Compute file-level diff between two artifacts by name + digest."""
    files_a: Dict[str, str] = {}
    files_b: Dict[str, str] = {}
    scan_truncated = False

    try:
        for f in art_a.files():
            if len(files_a) >= MAX_FILES_FOR_DIFF:
                scan_truncated = True
                break
            files_a[getattr(f, "name", "")] = getattr(f, "digest", "")
    except Exception:
        logger.debug("files() failed for artifact_a", exc_info=True)

    try:
        for f in art_b.files():
            if len(files_b) >= MAX_FILES_FOR_DIFF:
                scan_truncated = True
                break
            files_b[getattr(f, "name", "")] = getattr(f, "digest", "")
    except Exception:
        logger.debug("files() failed for artifact_b", exc_info=True)

    names_a, names_b = set(files_a.keys()), set(files_b.keys())
    added = sorted(names_b - names_a)
    removed = sorted(names_a - names_b)
    modified = sorted(n for n in (names_a & names_b) if files_a[n] != files_b[n])
    unchanged_count = len(names_a & names_b) - len(modified)

    total_entries = len(added) + len(removed) + len(modified)
    truncated = scan_truncated or total_entries > max_entries

    if truncated:
        added = added[:max_entries]
        remaining = max_entries - len(added)
        removed = removed[: max(remaining, 0)]
        remaining -= len(removed)
        modified = modified[: max(remaining, 0)]

    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "unchanged_count": unchanged_count,
        "truncated": truncated,
    }
