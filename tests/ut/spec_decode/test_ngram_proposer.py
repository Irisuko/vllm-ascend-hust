from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer


def test_constructor_supports_upstream_jit_warmup(monkeypatch):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            prompt_lookup_min=3,
            prompt_lookup_max=5,
            num_speculative_tokens=3,
        ),
        model_config=SimpleNamespace(max_model_len=32),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    monkeypatch.setattr(
        NgramProposer,
        "batch_propose",
        lambda _self, num_requests, *_args: [[] for _ in range(num_requests)],
    )

    proposer = AscendNgramProposer(config, SimpleNamespace(input_batch=None))

    assert proposer.k == 3


def test_propose_forwards_upstream_arguments_and_updates_tokens():
    input_batch = SimpleNamespace(
        req_ids=["request-0"],
        spec_decode_unsupported_reqs=set(),
        max_model_len=8,
    )
    proposer = AscendNgramProposer.__new__(AscendNgramProposer)
    proposer.runner = SimpleNamespace(input_batch=input_batch)
    proposer.batch_propose = MagicMock(return_value=[[21, 22]])
    num_tokens_no_spec = np.array([2], dtype=np.int32)
    token_ids_cpu = np.zeros((1, 8), dtype=np.int32)

    result = proposer.propose(
        3,
        [[11]],
        num_tokens_no_spec,
        token_ids_cpu,
    )

    assert result == [[21, 22]]
    assert token_ids_cpu[0, 2] == 11
    proposer.batch_propose.assert_called_once_with(
        1,
        [0],
        num_tokens_no_spec,
        token_ids_cpu,
        3,
    )
