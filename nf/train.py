import torch
import time

from nf.losses import kl_loss, raw_std_loss, kl_raw_std_loss


def train_nf(
    model,
    benchmark_fn,
    dim,
    steps=2000,
    batch_size=4096,
    lr=1e-3,
    loss_name="kl",
    device="cpu",
    eta=0.5,
    verbose=True,
    log_every=200,
    tol=None,
    patience=5,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    logged_history = []
    stopped_early = False
    stop_step = steps

    t0 = time.time()

    for step in range(steps):
        x = torch.rand(batch_size, dim, device=device)
        f_vals = benchmark_fn(x)
        f_vals = f_vals / (f_vals.mean() + 1e-12)
        log_q = model.log_prob(x)

        if loss_name == "kl":
            loss = kl_loss(log_q, f_vals)

        elif loss_name == "raw_std_loss":
            loss = raw_std_loss(log_q, f_vals)

        elif loss_name == "kl_raw_std_loss":
            loss = kl_raw_std_loss(log_q, f_vals, eta=eta)

        else:
            raise ValueError(f"Unknown loss_name: {loss_name}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = float(loss.item())
        history.append(loss_val)

        if step % log_every == 0:
            logged_history.append(loss_val)

            if verbose:
                print(f"step {step:4d} | loss = {loss_val:.6f}")

            if tol is not None and len(logged_history) >= 2 * patience:
                previous = logged_history[-2 * patience : -patience]
                recent = logged_history[-patience:]

                previous_mean = sum(previous) / len(previous)
                recent_mean = sum(recent) / len(recent)

                delta = abs(recent_mean - previous_mean)

                if verbose:
                    print(f"          convergence delta = {delta:.3e}")

                if delta < tol:
                    stopped_early = True
                    stop_step = step

                    if verbose:
                        print(
                            f"Early stopping at step {step}: "
                            f"delta={delta:.3e} < tol={tol:.3e}"
                        )

                    break

    runtime = time.time() - t0

    return {
        "model": model,
        "history": history,
        "logged_history": logged_history,
        "runtime": runtime,
        "loss_name": loss_name,
        "dim": dim,
        "steps": steps,
        "actual_steps": stop_step,
        "batch_size": batch_size,
        "lr": lr,
        "eta": eta,
        "tol": tol,
        "patience": patience,
        "stopped_early": stopped_early,
    }


def save_nf_checkpoint(train_out, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": train_out["model"].state_dict(),
            "history": train_out["history"],
            "runtime": train_out["runtime"],
            "loss_name": train_out["loss_name"],
            "dim": train_out["dim"],
            "steps": train_out["steps"],
            "batch_size": train_out["batch_size"],
            "lr": train_out["lr"],
            "eta": train_out["eta"],
        },
        path,
    )

    print(f"Saved checkpoint -> {path}")