import torch

from gemq.quantize import summarize_router_updates


def test_router_update_audit_detects_change_without_mean_shift():
    before = {"router": torch.tensor([1.0, 3.0])}
    parameter = torch.nn.Parameter(torch.tensor([2.0, 2.0]))

    audit = summarize_router_updates(before, [("router", parameter)])

    assert parameter.mean() == before["router"].mean()
    assert audit["effective_update"] is True
    assert audit["changed_elements"] == 2
    assert audit["changed_parameter_tensors"] == 1
    assert audit["delta_l2"] > 0
    assert audit["delta_max_abs"] == 1.0


def test_router_update_audit_records_representable_noop():
    before = {"router": torch.tensor([1.0, 3.0])}
    parameter = torch.nn.Parameter(before["router"].clone())

    audit = summarize_router_updates(before, [("router", parameter)])

    assert audit["effective_update"] is False
    assert audit["changed_elements"] == 0
    assert audit["delta_l2"] == 0.0
