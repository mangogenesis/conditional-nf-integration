import torch

from nf.conditional_cubic_1d import ConditionalCubicFlow1D
from nf.vectorized_cubic import batched_cubic_inverse


def test_batched_inverse_matches_scalar_outputs_and_gradients():
    torch.manual_seed(7)
    flows = [ConditionalCubicFlow1D(cond_dim=5, hidden=16, n_bins=8) for _ in range(4)]
    values = torch.rand(4, 13)
    contexts = torch.rand(4, 13, 5)

    expected_u, expected_ld = [], []
    for index, flow in enumerate(flows):
        u, ld = flow.inverse(values[index], contexts[index])
        expected_u.append(u)
        expected_ld.append(ld)
    expected_u = torch.stack(expected_u)
    expected_ld = torch.stack(expected_ld)

    actual_u, actual_ld = batched_cubic_inverse(flows, values, contexts)
    torch.testing.assert_close(actual_u, expected_u, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_ld, expected_ld, rtol=1e-5, atol=1e-6)

    scalar_loss = expected_ld.sum()
    scalar_loss.backward()
    scalar_grads = [flow.net[0].weight.grad.detach().clone() for flow in flows]
    for flow in flows:
        flow.zero_grad(set_to_none=True)
    actual_ld.sum().backward()
    for flow, expected_grad in zip(flows, scalar_grads):
        torch.testing.assert_close(flow.net[0].weight.grad, expected_grad, rtol=2e-5, atol=2e-6)
