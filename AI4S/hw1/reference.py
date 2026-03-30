import numpy as np


def initial_condition(x: np.ndarray) -> np.ndarray:
    """Burgers 方程给定初值: u(x,0) = -sin(pi*x)."""
    return -np.sin(np.pi * x)


def solve_burgers_fdm(
    nu: float,
    nx: int = 401,
    nt: int = 1201,
    x_range=(-1.0, 1.0),
    t_range=(0.0, 1.0),
):
    """
    用有限差分法（FDM）求解一维 Burgers 方程:
        u_t + u * u_x = nu * u_xx

    条件:
    - 边界条件: u(-1,t)=u(1,t)=0
    - 初始条件: u(x,0)=-sin(pi x)

    返回:
    - x: 空间网格, shape=(nx,)
    - t: 时间网格, shape=(nt,)
    - u: 数值解, shape=(nt,nx)
    """
    x0, x1 = x_range
    t0, t1 = t_range
    x = np.linspace(x0, x1, nx)
    t = np.linspace(t0, t1, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u = np.zeros((nt, nx), dtype=np.float64)
    u[0] = initial_condition(x)
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    # 显式时间推进:
    # 对流项采用按速度符号切换的一阶迎风差分
    # 扩散项采用二阶中心差分
    for n in range(nt - 1):
        un = u[n].copy()
        ux_upwind = np.zeros_like(un)

        pos = un >= 0
        neg = ~pos

        ux_upwind[pos] = (un[pos] - np.roll(un, 1)[pos]) / dx
        ux_upwind[neg] = (np.roll(un, -1)[neg] - un[neg]) / dx
        ux_upwind[0] = 0.0
        ux_upwind[-1] = 0.0

        uxx = (np.roll(un, -1) - 2.0 * un + np.roll(un, 1)) / (dx * dx)
        uxx[0] = 0.0
        uxx[-1] = 0.0

        u[n + 1] = un - dt * un * ux_upwind + dt * nu * uxx
        u[n + 1, 0] = 0.0
        u[n + 1, -1] = 0.0

    return x, t, u


def bilinear_interpolate_reference(
    x_ref: np.ndarray,
    t_ref: np.ndarray,
    u_ref: np.ndarray,
    xq: np.ndarray,
    tq: np.ndarray,
) -> np.ndarray:
    """在 (x_ref,t_ref) 规则网格上对参考解做双线性插值。"""
    xq = np.asarray(xq)
    tq = np.asarray(tq)

    ix = np.searchsorted(x_ref, xq, side="left")
    it = np.searchsorted(t_ref, tq, side="left")
    ix = np.clip(ix, 1, len(x_ref) - 1)
    it = np.clip(it, 1, len(t_ref) - 1)

    x1 = x_ref[ix - 1]
    x2 = x_ref[ix]
    t1 = t_ref[it - 1]
    t2 = t_ref[it]

    q11 = u_ref[it - 1, ix - 1]
    q12 = u_ref[it, ix - 1]
    q21 = u_ref[it - 1, ix]
    q22 = u_ref[it, ix]

    wx = (xq - x1) / (x2 - x1 + 1e-12)
    wt = (tq - t1) / (t2 - t1 + 1e-12)

    out = (
        q11 * (1 - wx) * (1 - wt)
        + q21 * wx * (1 - wt)
        + q12 * (1 - wx) * wt
        + q22 * wx * wt
    )
    return out
