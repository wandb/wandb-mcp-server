"""Registry filtering must not depend on removed private SDK utilities."""

import copy

import pytest

from wandb_mcp_server.registry_reads import _prefix_filter_names


@pytest.mark.parametrize(
    "source,expected",
    [
        ({"name": "model"}, {"name": "wandb-registry-model"}),
        ({"name": "wandb-registry-model"}, {"name": "wandb-registry-model"}),
        (
            {"name": {"$in": ["model", "dataset"]}},
            {"name": {"$in": ["wandb-registry-model", "wandb-registry-dataset"]}},
        ),
        ({"name": {"$regex": "^model"}}, {"name": {"$regex": "^model"}}),
        (
            {"$or": [{"name": "model"}, {"description": "model"}]},
            {"$or": [{"name": "wandb-registry-model"}, {"description": "model"}]},
        ),
        ({"name": {"$exists": True}}, {"name": {"$exists": True}}),
    ],
)
def test_prefixing_matches_literal_name_contract_without_mutation(source, expected):
    original = copy.deepcopy(source)
    assert _prefix_filter_names(source) == expected
    assert source == original
