import time
from typing import Dict, Tuple

import numpy as np
import torch

from baseline import BaselineMLP
from pinn_utils import (
    eval_model_on_grid,
    format_seconds,
    pde_residual,
    sample_bc,
    sample_f,
    sample_ic,
    set_seed,
    to_tensor,
)
from train_config import TrainConfig


def train_rar(
    cfg: TrainConfig,
    x_ref: np.ndarray = None,
    t_ref: np.ndarray = None,
    u_ref: np.ndarray = None,
) -> Tuple[torch.nn.Module, Dict]:
    """改进方法 2: RAR（残差自适应重采样）."""
    set_seed(cfg.seed)
    model = BaselineMLP().to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {"loss": [], "loss_f": [], "loss_ic": [], "loss_bc": [], "rel_l2": []}
    run_name = cfg.case_name if cfg.case_name else "rar"
    start = time.time()

    # RAR 维护一份配点池，周期性注入 hard points
    x_f_np, t_f_np = sample_f(cfg.n_f)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()

        x_ic, t_ic, u_ic = sample_ic(cfg.n_ic)
        x_l, t_l, u_l, x_r, t_r, u_r = sample_bc(cfg.n_bc)

        if epoch > 1:
            refresh_n = max(1, cfg.n_f // 5)
            x_new, t_new = sample_f(refresh_n)
            x_f_np[-refresh_n:] = x_new
            t_f_np[-refresh_n:] = t_new

        x_ic, t_ic, u_ic = to_tensor(x_ic, t_ic, u_ic, device=cfg.device)
        x_l, t_l, u_l = to_tensor(x_l, t_l, u_l, device=cfg.device)
        x_r, t_r, u_r = to_tensor(x_r, t_r, u_r, device=cfg.device)
        x_f, t_f = to_tensor(x_f_np, t_f_np, device=cfg.device)

        u0_pred = model(torch.cat([x_ic, t_ic], dim=1))
        ul_pred = model(torch.cat([x_l, t_l], dim=1))
        ur_pred = model(torch.cat([x_r, t_r], dim=1))
        f = pde_residual(model, x_f, t_f, cfg.nu)

        loss_ic = torch.mean((u0_pred - u_ic) ** 2)
        loss_bc = torch.mean((ul_pred - u_l) ** 2) + torch.mean((ur_pred - u_r) ** 2)
        loss_f = torch.mean(f**2)
        loss = loss_ic + loss_bc + loss_f

        loss.backward()
        opt.step()

        # RAR: 计算候选池残差，选 top-k 强化训练
        if epoch % cfg.rar_every == 0:
            with torch.no_grad():
                x_pool, t_pool = sample_f(cfg.rar_pool)
                x_pool_t, t_pool_t = to_tensor(x_pool, t_pool, device=cfg.device)

            f_pool = pde_residual(model, x_pool_t, t_pool_t, cfg.nu).detach().abs().squeeze(1)
            topk = min(cfg.rar_topk, f_pool.shape[0])
            idx = torch.topk(f_pool, k=topk).indices.cpu().numpy()
            x_hard = x_pool[idx]
            t_hard = t_pool[idx]

            replace_n = min(topk, x_f_np.shape[0])
            x_f_np[:replace_n] = x_hard[:replace_n]
            t_f_np[:replace_n] = t_hard[:replace_n]

        history["loss"].append(float(loss.item()))
        history["loss_f"].append(float(loss_f.item()))
        history["loss_ic"].append(float(loss_ic.item()))
        history["loss_bc"].append(float(loss_bc.item()))

        if x_ref is not None and t_ref is not None and u_ref is not None and epoch % cfg.log_every == 0:
            _, _, pred = eval_model_on_grid(model, nx=len(x_ref), nt=len(t_ref), device=cfg.device)
            rel_l2 = np.linalg.norm(pred - u_ref) / (np.linalg.norm(u_ref) + 1e-12)
            history["rel_l2"].append((epoch, float(rel_l2)))

        should_print = (
            cfg.verbose
            and (epoch == 1 or epoch % max(1, cfg.print_every) == 0 or epoch == cfg.epochs)
        )
        if should_print:
            elapsed = time.time() - start
            per_epoch = elapsed / epoch
            eta = per_epoch * (cfg.epochs - epoch)
            msg = (
                f"[TRAIN][{run_name}] "
                f"epoch {epoch}/{cfg.epochs} "
                f"loss={loss.item():.3e} "
                f"loss_f={loss_f.item():.3e} "
                f"loss_ic={loss_ic.item():.3e} "
                f"loss_bc={loss_bc.item():.3e} "
                f"elapsed={format_seconds(elapsed)} "
                f"eta={format_seconds(eta)}"
            )
            if history["rel_l2"] and history["rel_l2"][-1][0] == epoch:
                msg += f" rel_l2={history['rel_l2'][-1][1]:.3e}"
            print(msg, flush=True)

    return model, history
