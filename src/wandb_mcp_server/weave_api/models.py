"""
Data models for the Weave API.

This module defines the data structures used across the Weave API client.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class SortDirection(str, Enum):
    """Sort direction options."""

    ASC = "asc"
    DESC = "desc"


class FilterOperator(str, Enum):
    """Operators for query filters."""

    EQUALS = "$eq"
    GREATER_THAN = "$gt"
    GREATER_THAN_EQUAL = "$gte"
    LESS_THAN = "$lt"
    LESS_THAN_EQUAL = "$lte"
    CONTAINS = "$contains"


@dataclass
class TimeRange:
    """Time range for filtering traces."""

    start: Optional[datetime] = None
    end: Optional[datetime] = None


@dataclass
class AttributeFilter:
    """Filter for trace attributes."""

    path: str
    value: Any
    operator: FilterOperator = FilterOperator.EQUALS


@dataclass
class QueryFilter:
    """Query filter parameters."""

    display_name: Optional[str] = None
    op_name: Optional[str] = None
    op_name_contains: Optional[str] = None
    trace_id: Optional[str] = None
    trace_ids: Optional[List[str]] = None
    call_ids: Optional[List[str]] = None
    parent_ids: Optional[List[str]] = None
    status: Optional[str] = None
    time_range: Optional[TimeRange] = None
    latency: Optional[Dict[str, float]] = None
    attributes: Optional[List[AttributeFilter]] = None
    has_exception: Optional[bool] = None
    trace_roots_only: Optional[bool] = None


@dataclass
class QueryParams:
    """Parameters for a Weave trace query."""

    entity_name: str
    project_name: str
    filters: Optional[QueryFilter] = None
    sort_by: str = "started_at"
    sort_direction: SortDirection = SortDirection.DESC
    limit: Optional[int] = None
    offset: int = 0
    include_costs: bool = True
    include_feedback: bool = True
    columns: Optional[List[str]] = None
    expand_columns: Optional[List[str]] = None
    truncate_length: Optional[int] = 200
    return_full_data: bool = False
    metadata_only: bool = False


@dataclass
class TraceCost:
    """Cost information for a model in a trace."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_total_cost: float = 0.0
    completion_tokens_total_cost: float = 0.0
    total_cost: float = 0.0


class TokenCounts(BaseModel):
    """Token count information for traces."""

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    average_tokens_per_trace: Union[int, float] = 0


class TraceMetadata(BaseModel):
    """Metadata about a collection of traces."""

    total_traces: int = 0
    token_counts: Dict[str, Union[int, float]] = Field(default_factory=dict)
    time_range: Dict[str, Optional[datetime]] = Field(default_factory=dict)
    status_summary: Dict[str, int] = Field(default_factory=dict)
    op_distribution: Dict[str, int] = Field(default_factory=dict)


class WeaveTrace(BaseModel):
    """Representation of a Weave trace."""

    id: str
    project_id: str
    op_name: str
    trace_id: str
    started_at: datetime
    display_name: Optional[str] = None
    parent_id: Optional[str] = None
    ended_at: Optional[datetime] = None
    inputs: Optional[Dict[str, Any]] = None
    output: Optional[Any] = None
    exception: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    summary: Dict[str, Any] = Field(default_factory=dict)
    costs: Dict[str, TraceCost] = Field(default_factory=dict)
    feedback: Dict[str, Any] = Field(default_factory=dict)
    status: Optional[str] = None
    latency_ms: Optional[int] = None
    wb_run_id: Optional[str] = None
    wb_user_id: Optional[str] = None
    deleted_at: Optional[datetime] = None

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True


class QueryResult(BaseModel):
    """Result of a Weave trace query."""

    metadata: TraceMetadata
    traces: Optional[List[Union[WeaveTrace, Dict[str, Any]]]] = None

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True
