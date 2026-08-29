import torch

from gemq.inference.moe_block import sort_expert_assignments


def test_sort_expert_assignments_preserves_token_weight_pairs():
    selected = torch.tensor([[2, 0], [1, 2], [0, 1]])
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]])

    experts, tokens, sorted_weights, counts = sort_expert_assignments(
        selected, weights, top_k=2, num_experts=4
    )

    assert experts.tolist() == [0, 0, 1, 1, 2, 2]
    assert tokens.tolist() == [0, 2, 1, 2, 0, 1]
    assert torch.equal(sorted_weights, torch.tensor([0.3, 0.8, 0.4, 0.2, 0.7, 0.6]))
    assert counts == [2, 2, 2, 0]


def test_sort_expert_assignments_is_stable_within_expert():
    selected = torch.tensor([[1, 0], [1, 2], [1, 0]])
    weights = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    experts, tokens, sorted_weights, counts = sort_expert_assignments(
        selected, weights, top_k=2, num_experts=3
    )

    expert_one = experts == 1
    assert tokens[expert_one].tolist() == [0, 1, 2]
    assert sorted_weights[expert_one].tolist() == [0.0, 2.0, 4.0]
    assert counts == [2, 3, 1]
