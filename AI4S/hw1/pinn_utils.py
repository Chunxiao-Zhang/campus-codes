import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_ic(n: int):
    x = np.random.uniform(-1.0, 1.0, size=(n, 1))
    t = np.zeros((n, 1))
    u = -np.sin(np.pi * x)
    return x, t, u


def sample_bc(n: int):
    t = np.random.uniform(0.0, 1.0, size=(n, 1))
    x_l = -np.ones((n, 1))
    x_r = np.ones((n, 1))
    u_l = np.zeros((n, 1))
    u_r = np.zeros((n, 1))
    return x_l, t, u_l, x_r, t, u_r


def sample_f(n: int):
    x = np.random.uniform(-1.0, 1.0, size=(n, 1))
    t = np.random.uniform(0.0, 1.0, size=(n, 1))
    return x, t


def to_tensor(*arrs, device="cpu"):
    return [torch.tensor(a, dtype=torch.float32, device=device) for a in arrs]


def pde_residual(model: nn.Module, x: torch.Tensor, t: torch.Tensor, nu: float):
    x.requires_grad_(True)
    t.requires_grad_(True)
    xt = torch.cat([x, t], dim=1)
    u = model(xt)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    return u_t + u * u_x - nu * u_xx


def eval_model_on_grid(model: nn.Module, nx=201, nt=201, device="cpu", batch_size: int = 65536):
    xs = np.linspace(-1, 1, nx)
    ts = np.linspace(0, 1, nt)
    xx, tt = np.meshgrid(xs, ts)
    xt = np.stack([xx.reshape(-1), tt.reshape(-1)], axis=1)

    pred_parts = []
    with torch.no_grad():
        xt_t = torch.tensor(xt, dtype=torch.float32, device=device)
        for i in range(0, xt_t.shape[0], batch_size):
            pred_parts.append(model(xt_t[i : i + batch_size]).cpu().numpy())

    pred = np.vstack(pred_parts)
    return xs, ts, pred.reshape(nt, nx)


def format_seconds(seconds: float) -> str:
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"
