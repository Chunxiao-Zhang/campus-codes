import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List

# 避免默认 matplotlib 配置目录不可写
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplcache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from adaptive_activation import train_adaptive_activation
from baseline import train_baseline
from pinn_utils import eval_model_on_grid
from rar import train_rar
from reference import solve_burgers_fdm
from train_config import TrainConfig


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

METHOD_REGISTRY = {
    "base": train_baseline,
    "ada": train_adaptive_activation,
    "rar": train_rar,
}

METHOD_ALIASES = {
    "baseline": "base",
    "adaptive_activation": "ada",
    "base": "base",
    "ada": "ada",
    "rar": "rar",
}


def resolve_methods(args) -> List[str]:
    """解析方法选择，支持空格分隔、逗号分隔、重复 --method。"""
    raw_items = []
    if args.methods:
        raw_items.extend(args.methods)
    if args.method:
        raw_items.extend(args.method)

    if not raw_items:
        raw_items = ["all"]

    tokens: List[str] = []
    for item in raw_items:
        for part in str(item).split(","):
            p = part.strip()
            if p:
                tokens.append(p.lower())

    if not tokens:
        return list(METHOD_REGISTRY.keys())

    if "all" in tokens:
        return list(METHOD_REGISTRY.keys())

    mapped = [METHOD_ALIASES.get(m, m) for m in tokens]

    invalid = [m for m in mapped if m not in METHOD_REGISTRY]
    if invalid:
        valid = ", ".join(METHOD_REGISTRY.keys())
        bad = ", ".join(sorted(set(invalid)))
        raise ValueError(
            f"Unknown method(s): {bad}. Valid methods: {valid}, all "
            "(aliases: baseline->base, adaptive_activation->ada)"
        )

    # 去重并保持输入顺序
    resolved = []
    seen = set()
    for m in mapped:
        if m not in seen:
            seen.add(m)
            resolved.append(m)
    return resolved


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_curve(history, out_path: Path, key="loss", title="Training Loss"):
    y = history[key]
    x = np.arange(1, len(y) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_field(x, t, u, out_path: Path, title="u(x,t)"):
    xx, tt = np.meshgrid(x, t)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, tt, u, cmap="RdBu_r", linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="u")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u")
    ax.view_init(elev=28, azim=-130)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def run_case(
    method_name: str,
    cfg: TrainConfig,
    x_ref,
    t_ref,
    u_ref,
    out_dir: Path,
    cli_args: dict,
):
    train_fn = METHOD_REGISTRY[method_name]
    model, hist = train_fn(cfg, x_ref, t_ref, u_ref)

    xg, tg, up = eval_model_on_grid(model, nx=len(x_ref), nt=len(t_ref), device=cfg.device)
    rel_l2 = float(np.linalg.norm(up - u_ref) / (np.linalg.norm(u_ref) + 1e-12))
    mse = float(np.mean((up - u_ref) ** 2))

    case_dir = out_dir / method_name
    ensure_dir(case_dir)

    torch.save(model.state_dict(), case_dir / "model.pt")

    with open(case_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"rel_l2": rel_l2, "mse": mse, "nu": cfg.nu, "epochs": cfg.epochs},
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(case_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": method_name,
                "cli_args": cli_args,
                "train_config": asdict(cfg),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(case_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False)

    save_curve(hist, case_dir / "loss.png", key="loss", title=f"{method_name} - total loss")
    save_curve(hist, case_dir / "loss_f.png", key="loss_f", title=f"{method_name} - PDE loss")
    save_field(x_ref, t_ref, u_ref, case_dir / "ref_field.png", title="Reference (FDM)")
    save_field(xg, tg, up, case_dir / "pred_field.png", title=f"{method_name} prediction")
    save_field(xg, tg, np.abs(up - u_ref), case_dir / "error_field.png", title=f"{method_name} |error|")

    return {"name": method_name, "rel_l2": rel_l2, "mse": mse, "dir": str(case_dir)}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help=(
            "选择训练方法，可传一个或多个。"
            "支持: --methods base rar 或 --methods base,rar 或 --methods all"
        ),
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help="可重复传入单个方法，例如 --method baseline --method rar",
    )

    parser.add_argument("--nu", type=float, default=0.01, help="粘性系数分子，程序内部使用 nu/pi")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="outputs")

    parser.add_argument("--n_f", type=int, default=8000)
    parser.add_argument("--n_ic", type=int, default=400)
    parser.add_argument("--n_bc", type=int, default=400)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--quiet", action="store_true")

    # RAR 超参（仅 rar 方法使用）
    parser.add_argument("--rar_every", type=int, default=500)
    parser.add_argument("--rar_pool", type=int, default=5000)
    parser.add_argument("--rar_topk", type=int, default=500)

    # 参考解网格分辨率
    parser.add_argument("--nx_ref", type=int, default=401)
    parser.add_argument("--nt_ref", type=int, default=1201)

    return parser.parse_args()


def main():
    args = parse_args()
    try:
        methods = resolve_methods(args)
    except ValueError as e:
        raise SystemExit(str(e))
    nu = float(args.nu) / np.pi

    cli_args = vars(args).copy()
    cli_args["methods_resolved"] = methods
    cli_args["nu_input"] = float(args.nu)
    cli_args["nu_resolved"] = nu

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    x_ref, t_ref, u_ref = solve_burgers_fdm(nu=nu, nx=args.nx_ref, nt=args.nt_ref)

    print(f"[INFO] methods={methods}")

    rows = []
    total = len(methods)
    for idx, method_name in enumerate(methods, start=1):
        print(f"[INFO] running ({idx}/{total}): {method_name}")

        cfg = TrainConfig(
            nu=nu,
            epochs=args.epochs,
            lr=args.lr,
            n_f=args.n_f,
            n_ic=args.n_ic,
            n_bc=args.n_bc,
            device=args.device,
            seed=args.seed,
            log_every=args.log_every,
            print_every=args.print_every,
            verbose=not args.quiet,
            case_name=method_name,
            rar_every=args.rar_every,
            rar_pool=args.rar_pool,
            rar_topk=args.rar_topk,
        )

        rows.append(run_case(method_name, cfg, x_ref, t_ref, u_ref, out_dir, cli_args))

    rows = sorted(rows, key=lambda x: x["rel_l2"])
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\n=== Summary (sorted by rel_l2) ===")
    for r in rows:
        print(f"{r['name']:<24} rel_l2={r['rel_l2']:.4e} mse={r['mse']:.4e}")
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()
