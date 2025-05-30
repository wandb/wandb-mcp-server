"""Models for the MCP query tools."""

from typing import Any, Dict, List, Optional, Union

import wandb

from wandb_mcp_server.utils import get_rich_logger

# Added imports for AST pagination

# Create a logger for this module
logger = get_rich_logger(__name__)


def list_entity_projects(entity: str | None = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch all projects for a specific wandb entity. If no entity is provided, 
    fetches projects for the current user and their teams.

    Args:
        entity (str, optional): The wandb entity (username or team name). If None, 
                               fetches projects for the current user and their teams.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping entity names to lists of project dictionaries.
            Each project dictionary contains:
            - name: Project name
            - entity: Entity name
            - description: Project description
            - visibility: Project visibility (public/private)
            - created_at: Creation timestamp
            - updated_at: Last update timestamp
            - tags: List of project tags
    """
    # Initialize wandb API
    api = wandb.Api()

    # Merge entity and teams into a single list
    if entity is None:
        viewer = api.viewer
        entities = [viewer.entity] + viewer.teams
    else:
        entities = [entity]

    # Get all projects for the entity

    entities_projects = {}
    for entity in entities:
        projects = api.projects(entity)
        
        # Convert projects to a list of dictionaries
        projects_data = []
        for project in projects:
            project_dict = {
                "name": project.name,
                "entity": project.entity,
                "description": getattr(project, "description", None),
                "visibility": getattr(project, "visibility", None),
                "created_at": project.created_at,
                "updated_at": project.updated_at,
                "tags": project.tags,
            }
            projects_data.append(project_dict)

        entities_projects[entity] = projects_data

    return entities_projects


def query_wandb_runs(
    entity: str,
    project: str,
    per_page: int = 50,
    order: str = "-created_at",
    filters: Dict[str, Any] = None,
    search: str = None,
) -> List[Dict[str, Any]]:
    """
    Fetch runs from a specific wandb entity and project with filtering and sorting support.

    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        per_page (int): Number of runs to fetch (default: 50)
        order (str): Sort order (default: "-created_at"). Prefix with "-" for descending order.
                    Examples: "created_at", "-created_at", "name", "-name", "state", "-state"
        filters (Dict[str, Any]): Dictionary of filters to apply. Keys can be:
            - state: "running", "finished", "crashed", "failed", "killed"
            - tags: List of tags to filter by
            - config: Dictionary of config parameters to filter by
            - summary: Dictionary of summary metrics to filter by
        search (str): Search string to filter runs by name or tags

    Returns:
        List[Dict[str, Any]]: List of run dictionaries containing run information
    """
    # Initialize wandb API
    api = wandb.Api()

    # Build query parameters
    query_params = {"per_page": per_page, "order": order}

    # Add filters if provided
    if filters:
        for key, value in filters.items():
            if key in ["state", "tags", "config", "summary"]:
                query_params[key] = value

    # Add search if provided
    if search:
        query_params["search"] = search

    # Get runs from the specified entity and project with filters
    runs = api.runs(f"{entity}/{project}", **query_params)

    # Convert runs to a list of dictionaries
    runs_data = []
    for run in runs:
        run_dict = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "config": run.config,
            "summary": run.summary,
            "created_at": run.created_at,
            "url": run.url,
            "tags": run.tags,
        }
        runs_data.append(run_dict)

    return runs_data


def query_wandb_run_config(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch configuration parameters for a specific run.

    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        run_id (str): The ID of the run to fetch config for

    Returns:
        Dict[str, Any]: Dictionary containing configuration parameters
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return run.config


def query_wandb_run_training_metrics(
    entity: str, project: str, run_id: str
) -> Dict[str, List[Any]]:
    """
    Fetch training metrics history for a specific run.

    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        run_id (str): The ID of the run to fetch metrics for

    Returns:
        Dict[str, List[Any]]: Dictionary mapping metric names to their history
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Get the history of all metrics
    history = run.history()

    # Convert to a more convenient format
    metrics = {}
    for column in history.columns:
        if column not in ["_timestamp", "_runtime", "_step"]:
            metrics[column] = history[column].tolist()

    return metrics


def query_wandb_run_system_metrics(
    entity: str, project: str, run_id: str
) -> Dict[str, List[Any]]:
    """
    Fetch system metrics history for a specific run.

    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        run_id (str): The ID of the run to fetch metrics for

    Returns:
        Dict[str, List[Any]]: Dictionary mapping system metric names to their history
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Get the history of system metrics
    system_metrics = run.history(stream="events")

    # Convert to a more convenient format
    metrics = {}
    for column in system_metrics.columns:
        if column not in ["_timestamp", "_runtime", "_step"]:
            metrics[column] = system_metrics[column].tolist()

    return metrics


def query_wandb_run_summary_metrics(
    entity: str, project: str, run_id: str
) -> Dict[str, Any]:
    """
    Fetch summary metrics for a specific run.

    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        run_id (str): The ID of the run to fetch metrics for

    Returns:
        Dict[str, Any]: Dictionary containing summary metrics
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return run.summary


def query_wandb_artifacts(
    entity: str,
    project: str,
    artifact_name: Optional[str] = None,
    artifact_type: Optional[str] = None,
    version_alias: str = "latest",
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetches details for a specific artifact or lists artifact collections of a specific type.

    If artifact_name is provided, fetches details for that specific artifact.
    If artifact_name is not provided, artifact_type must be provided to list
    collections of that type.

    Args:
        entity (str): The wandb entity (username or team name).
        project (str): The project name.
        artifact_name (Optional[str]): The name of the artifact to fetch (e.g., 'my-dataset').
                                       If None, lists collections based on artifact_type.
        artifact_type (Optional[str]): The type of artifact collection to list.
                                       Required if artifact_name is None.
        version_alias (str): The version or alias for the specific artifact
                             (e.g., 'v1', 'latest'). Defaults to 'latest'.
                             Ignored if artifact_name is None.

    Returns:
        Union[List[Dict[str, Any]], Dict[str, Any]]:
            - Dict[str, Any]: Details of the specified artifact if artifact_name is provided.
            - List[Dict[str, Any]]: List of artifact collections if artifact_name is None
                                     and artifact_type is provided.

    Raises:
        ValueError: If neither artifact_name nor artifact_type is provided,
                    or if artifact_name is None and artifact_type is also None.
        wandb.errors.CommError: If the specified artifact is not found when artifact_name is provided.
    """
    api = wandb.Api()

    if artifact_name:
        # Fetch specific artifact details (logic from get_artifact)
        try:
            artifact = api.artifact(
                name=f"{entity}/{project}/{artifact_name}:{version_alias}"
            )
            artifact_data = {
                "id": artifact.id,
                "name": artifact.name,
                "type": artifact.type,
                "version": artifact.version,
                "aliases": artifact.aliases,
                "state": artifact.state,
                "size": artifact.size,
                "created_at": artifact.created_at,
                "description": artifact.description,
                "metadata": artifact.metadata,
                "digest": artifact.digest,
            }
            return artifact_data
        except wandb.errors.CommError as e:
            # Re-raise to signal artifact not found or other communication issues
            raise e
    elif artifact_type:
        # List artifact collections (logic from list_artifact_collections)
        collections = api.artifact_collections(
            project_name=f"{entity}/{project}", type_name=artifact_type
        )
        collections_data = []
        for collection in collections:
            collections_data.append(
                {
                    "name": collection.name,
                    "type": collection.type,
                    "project": project,  # Include project for clarity
                    "entity": entity,  # Include entity for clarity
                }
            )
        return collections_data
    else:
        raise ValueError("Either 'artifact_name' or 'artifact_type' must be provided.")


def query_wandb_sweeps(
    entity: str, project: str, action: str, sweep_id: Optional[str] = None
) -> Union[List[Dict[str, Any]], Dict[str, Any], None]:
    """
    Manages W&B sweeps: either lists all sweeps in a project OR gets the best run for a specific sweep.

    Use the 'action' parameter to specify the desired operation:
    - Set action='list_sweeps' to list all sweeps in the project. 'sweep_id' is ignored.
    - Set action='get_best_run' to find the best run for a specific sweep. 'sweep_id' is REQUIRED for this action.

    Args:
        entity (str): The wandb entity (username or team name).
        project (str): The project name.
        action (str): The operation to perform. Must be exactly 'list_sweeps' or 'get_best_run'.
        sweep_id (Optional[str]): The unique ID of the sweep. This is REQUIRED only when action='get_best_run'.
                                  It is ignored if action='list_sweeps'.

    Returns:
        Union[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
            - If action='list_sweeps': Returns a list of unique sweeps found in the project. [List[Dict]]
            - If action='get_best_run': Returns details of the best run for the specified sweep_id. [Dict]
                                        Returns None if the sweep exists but has no best run yet. [None]

    Raises:
        ValueError: If 'action' is not 'list_sweeps' or 'get_best_run'.
        ValueError: If action='get_best_run' but 'sweep_id' is not provided.
        wandb.errors.CommError: If a provided 'sweep_id' (when action='get_best_run') is not found or other API errors occur.
    """
    api = wandb.Api()

    if action == "list_sweeps":
        # List all sweeps in the project (logic from original list_wandb_sweeps)
        runs = api.runs(f"{entity}/{project}", include_sweeps=True)
        sweeps_found = {}
        for run in runs:
            if run.sweep and run.sweep.id not in sweeps_found:
                sweep_obj = run.sweep
                sweeps_found[sweep_obj.id] = {
                    "id": sweep_obj.id,
                    "config": sweep_obj.config,
                    "metric": getattr(sweep_obj, "metric", None),
                    "method": getattr(sweep_obj, "method", None),
                    "entity": sweep_obj.entity,
                    "project": sweep_obj.project,
                    "state": sweep_obj.state,
                }
        return list(sweeps_found.values())

    elif action == "get_best_run":
        # Get the best run for a specific sweep (logic from original get_wandb_sweep_best_run)
        if sweep_id is None:
            raise ValueError(
                "The 'sweep_id' argument is required when action is 'get_best_run'."
            )

        try:
            sweep = api.sweep(path=f"{entity}/{project}/{sweep_id}")
            best_run = sweep.best_run()

            if best_run:
                run_dict = {
                    "id": best_run.id,
                    "name": best_run.name,
                    "state": best_run.state,
                    "config": best_run.config,
                    "summary": best_run.summary,
                    "created_at": best_run.created_at,
                    "url": best_run.url,
                    "tags": best_run.tags,
                }
                return run_dict
            else:
                # Sweep exists, but no best run found
                return None
        except wandb.errors.CommError as e:
            # Re-raise if sweep_id itself is invalid or other API error occurs
            raise e
    else:
        # Invalid action specified
        raise ValueError(
            f"Invalid action specified: '{action}'. Must be 'list_sweeps' or 'get_best_run'."
        )


def query_wandb_reports(entity: str, project: str) -> List[Dict[str, Any]]:
    """
    List available W&B Reports within a project.

    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name

    Returns:
        List[Dict[str, Any]]: List of report dictionaries.
    """
    # Note: The public API for listing reports might be less direct.
    # `api.reports` might require entity/project to be set in Api() constructor
    # or might work differently. This is an attempt based on API structure.
    # If this fails, GraphQL might be necessary (see execute_graphql_query).
    try:
        # Initialize API potentially with overrides if needed
        api = wandb.Api(overrides={"entity": entity, "project": project})
        reports = (
            api.reports()
        )  # Assumes this lists reports for the configured entity/project

        reports_data = []
        for report in reports:
            # Attributes depend on the actual Report object structure
            report_data = {
                "id": getattr(report, "id", None),  # Adjust attribute names as needed
                "name": getattr(report, "name", None),
                "title": getattr(
                    report, "title", getattr(report, "display_name", None)
                ),
                "description": getattr(report, "description", None),
                "url": getattr(report, "url", None),
                "created_at": getattr(report, "created_at", None),
                "updated_at": getattr(report, "updated_at", None),
            }
            reports_data.append(report_data)
        return reports_data
    except Exception as e:
        # Consider logging the error
        logger.error(
            f"Error listing reports for {entity}/{project}: {e}. Direct report listing might require GraphQL."
        )
        # Fallback or raise error
        return []  # Return empty list on error for now
