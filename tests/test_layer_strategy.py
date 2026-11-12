from r2omni.sspa.policy.layer_strategy import LayerStrategyConfig, build_layer_strategy


def test_four_layer_groups_for_qwen_7b() -> None:
    strategy = build_layer_strategy(
        LayerStrategyConfig(mode="block", num_layers=36, block_group_size=4)
    )

    assert [
        layer for layer, item in strategy.items() if item["score_at_this_layer"]
    ] == list(range(0, 36, 4))
    assert strategy[0]["assignment_source_layer"] == 0
    assert strategy[3]["assignment_source_layer"] == 0
    assert strategy[32]["group_end_layer"] == 35
    assert strategy[35]["assignment_source_layer"] == 32
