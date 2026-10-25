import torch


def kl_loss(log_q, f_vals, eps=1e-12):
    w = f_vals / (f_vals.sum() + eps)
    return -(w.detach() * log_q).sum()


def raw_std_loss(log_q, f_vals, eps=1e-12, max_logw=20.0):
    log_f = torch.log(f_vals + eps)
    log_w = log_f - log_q

    # smooth cap: avoids hard gradient cutoff
    log_w = max_logw * torch.tanh(log_w / max_logw)

    w = torch.exp(log_w)
    w = w / (w.mean().detach() + eps)

    return torch.sqrt(w.var(unbiased=False) + 1e-8)


def kl_raw_std_loss(log_q, f_vals, eta=0.5):
    return kl_loss(log_q, f_vals) + eta * raw_std_loss(log_q, f_vals)


def logw_std_loss(log_q, f_vals, eta=0.1, logq_min=-20.0, eps=1e-12):
    """
    Alternative loss:
        L = L_KL + eta * std(log(1 + w))
    where
        w = f / q
    """
    f_safe = torch.clamp(f_vals, min=eps)
    log_q_safe = torch.clamp(log_q, min=logq_min)

    w = f_safe * torch.exp(-log_q_safe)
    z = torch.log1p(w)

    return kl_loss(log_q, f_vals, eps=eps, logq_min=logq_min) + eta * z.std(unbiased=False)


def cv2_loss(log_q, f_vals, logq_min=-20.0, eps=1e-12):
    """
    CV^2 loss:
        L = E[w^2] / E[w]^2 - 1
    """
    f_safe = torch.clamp(f_vals, min=eps)
    log_q_safe = torch.clamp(log_q, min=logq_min)

    w = f_safe * torch.exp(-log_q_safe)
    m1 = w.mean()
    m2 = (w ** 2).mean()

    return m2 / (m1 ** 2 + eps) - 1.0