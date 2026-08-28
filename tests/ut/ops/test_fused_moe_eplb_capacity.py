from types import SimpleNamespace

import pytest

from vllm_ascend.ops.fused_moe.fused_moe import make_eplb_placement_config, use_multistage_eplb_load


@pytest.mark.parametrize(
    ("dynamic_eplb", "policy_type", "collection_interval", "expected"),
    [
        (True, 2, 600, False),
        (True, 3, 600, True),
        (True, 3, 1, False),
        (False, 3, 600, False),
    ],
)
def test_use_multistage_eplb_load(dynamic_eplb, policy_type, collection_interval, expected):
    assert use_multistage_eplb_load(dynamic_eplb, policy_type, collection_interval) is expected


def test_make_eplb_placement_config_does_not_copy_source():
    source = SimpleNamespace(expert_map_path=None, dynamic_eplb=True, num_redundant_experts=0)

    placement_config = make_eplb_placement_config(source, num_redundant_experts=8)

    assert placement_config.expert_map_path is None
    assert placement_config.dynamic_eplb is True
    assert placement_config.num_redundant_experts == 8
    assert source.num_redundant_experts == 0
