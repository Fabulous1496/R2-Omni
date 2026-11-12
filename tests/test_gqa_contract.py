import pytest
import torch


@pytest.mark.parametrize("query_heads,kv_heads", [(16, 2), (28, 4), (32, 8)])
def test_qwen_gqa_head_order(query_heads: int, kv_heads: int) -> None:
    """Qwen repeats each KV head contiguously, including GQA ratio seven."""
    ratio = query_heads // kv_heads
    values = torch.arange(kv_heads).view(1, kv_heads, 1, 1)
    repeated = values[:, :, None].expand(1, kv_heads, ratio, 1, 1)
    repeated = repeated.reshape(1, query_heads, 1, 1)
    expected = torch.arange(kv_heads).repeat_interleave(ratio)
    assert torch.equal(repeated.flatten(), expected)
