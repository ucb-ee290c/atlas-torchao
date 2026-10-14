import copy

import characterize_layers as cl
import pytest
import torch


def test_population_and_bounded_sample_do_not_change_rng():
    torch.manual_seed(17)
    state = torch.random.get_rng_state().clone()
    x = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0, float("inf")])
    record = cl.tensor_summary(x, sample_seed=4, max_samples=3)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert record["population_elements"] == 6
    assert record["finite_elements"] == 5 and record["nonfinite_elements"] == 1
    assert record["zero_elements"] == 1
    assert record["population_finite"]["mean"] == pytest.approx(0.0, abs=1e-14)
    assert record["population_finite"]["rms"] == pytest.approx((34 / 5) ** 0.5)
    assert record["sample"]["elements"] == 3
    assert record["sample"]["quantiles_are_sample_estimates"]
    assert record == cl.tensor_summary(x, sample_seed=4, max_samples=3)


def test_exact_trace_preserves_model_outputs_and_pairs_observations():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
    x = torch.arange(8).reshape(2, 4).float()
    expected = model(x).detach().clone()
    with cl.LayerTrace(model, ["0"], max_samples=3) as first:
        first.begin_observation("episode3/frame4", mode="bf16", seed=12)
        assert torch.equal(model(x), expected)
    with cl.LayerTrace(model, ["0"], max_samples=3) as second:
        second.begin_observation("episode3/frame4", mode="candidate", seed=12)
        assert torch.equal(model(x), expected)
    assert len(first.records) == 2 and not first.handles
    paired = cl.compare_samples(first.records, second.records)
    assert all(row["sample_metrics"]["rmse"] == 0 for row in paired)
    assert all(not row["full_tensor_error_measured"] for row in paired)
    assert not model[0]._forward_hooks and not model[0]._forward_pre_hooks


def test_repeated_calls_are_not_labelled_sampling_iterations():
    model = torch.nn.Sequential(torch.nn.Identity()).eval()
    with cl.LayerTrace(model, ["0"]) as trace:
        trace.begin_observation("observation", mode="baseline", seed=0)
        model(torch.ones(2))
        model(torch.zeros(2))
    assert [row["call_index"] for row in trace.records] == [0, 0, 1, 1]
    assert all(row["sampling_step"] is None for row in trace.records)


def test_missing_duplicate_or_misaligned_pairs_refuse():
    model = torch.nn.Sequential(torch.nn.Identity()).eval()
    with cl.LayerTrace(model, ["0"]) as trace:
        trace.begin_observation("x", mode="bf16", seed=0)
        model(torch.arange(6).float())
    with pytest.raises(ValueError, match="coverage"):
        cl.compare_samples(trace.records, trace.records[:1])
    with pytest.raises(ValueError, match="duplicate"):
        cl.compare_samples(trace.records + trace.records, trace.records)
    changed = copy.deepcopy(trace.records)
    changed[0]["sample"]["indices"] = [100]
    with pytest.raises(ValueError, match="coordinates"):
        cl.compare_samples(trace.records, changed)
    with pytest.raises(ValueError, match="empty"):
        cl.compare_samples([], [])


def test_nonfinite_error_and_invalid_sites_refuse():
    model = torch.nn.Sequential(torch.nn.Identity()).eval()
    with pytest.raises(ValueError, match="missing"):
        cl.LayerTrace(model, ["absent"])
    with pytest.raises(ValueError, match="distinct"):
        cl.LayerTrace(model, ["0", "0"])
    with cl.LayerTrace(model, ["0"]) as trace:
        trace.begin_observation("x", mode="bf16", seed=0)
        model(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="nonfinite"):
        cl.compare_samples(trace.records, trace.records)


def test_record_budget_and_exception_restore_hooks():
    model = torch.nn.Sequential(torch.nn.Identity()).eval()
    with pytest.raises(RuntimeError, match="budget"):
        with cl.LayerTrace(model, ["0"], max_records=1) as trace:
            trace.begin_observation("x", mode="bf16", seed=0)
            model(torch.ones(2))
    assert len(trace.records) == 1 and not trace.handles
    assert not model[0]._forward_hooks and not model[0]._forward_pre_hooks


def test_empty_tensor_is_explicit_not_nan():
    result = cl.tensor_summary(torch.empty(0), sample_seed=1)
    assert result["population_elements"] == 0
    assert result["population_finite"]["mean"] is None
    assert result["sample"]["finite_abs_quantiles"] is None
