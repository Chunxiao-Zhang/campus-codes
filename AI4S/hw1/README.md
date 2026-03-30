# PINNs for 1D Burgers Equation

这个目录用于复现实验：在 1D Burgers 方程上对比标准 PINNs 与两种改进方法，并与 FDM 参考解比较误差。

## 快速开始

```bash
cd topic2_pinns
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 训练入口

统一入口：`run_experiments.py`

```bash
python run_experiments.py --methods base ada rar --nu 0.01 --epochs 4000 --device cpu --out outputs_run_1e-2
```

说明：
- `--nu` 传入的是分子值，程序内部实际使用 `nu/pi`。
  - 例如 `--nu 0.01` 实际为 `0.01/pi`
  - `--nu 0.001` 实际为 `0.001/pi`
- `--methods` 可选一个或多个方法：`base`、`ada`、`rar`
- 也支持：`--methods all`

### 常见命令

只跑基线：

```bash
python run_experiments.py --methods base --nu 0.01 --epochs 4000 --device cpu --out outputs_base
```

跑两种方法：

```bash
python run_experiments.py --methods base rar --nu 0.001 --epochs 4000 --device cpu --out outputs_base_rar
```

## 主要参数

- `--methods`：训练方法（`base`/`ada`/`rar`/`all`）
- `--nu`：粘性系数分子（内部按 `nu/pi` 计算）
- `--epochs`：训练轮数
- `--lr`：学习率
- `--n_f`：PDE 配点数
- `--n_ic`：初值点数
- `--n_bc`：边界点数
- `--device`：`cpu` 或 `cuda`
- `--print_every`：日志打印间隔
- `--quiet`：关闭详细训练日志

RAR 专用：
- `--rar_every`
- `--rar_pool`
- `--rar_topk`

参考解网格：
- `--nx_ref`
- `--nt_ref`

## 输出结构

每个方法会生成一个子目录（例如 `outputs_run_1e-2/base/`），包含：

- `metrics.json`：最终误差指标（`rel_l2`、`mse`）
- `history.json`：训练过程损失历史
- `loss.png`：总损失曲线
- `loss_f.png`：PDE 残差损失曲线
- `ref_field.png`：FDM 参考场
- `pred_field.png`：模型预测场
- `error_field.png`：绝对误差场
- `model.pt`：模型参数

汇总文件：
- `summary.json`（按 `rel_l2` 排序）

## 脚本功能说明

- `run_experiments.py`
  - 训练主入口
  - 解析命令行参数
  - 调用指定方法训练
  - 保存指标、模型和可视化结果

- `reference.py`
  - FDM 参考解求解器
  - 提供 `solve_burgers_fdm(...)`
  - 提供双线性插值工具

- `baseline.py`
  - 标准 PINNs（MLP + tanh）实现
  - 定义基线网络 `BaselineMLP`
  - 定义训练函数 `train_baseline(...)`

- `adaptive_activation.py`
  - 改进方法 1：自适应激活 `tanh(alpha z)`
  - 定义训练函数 `train_adaptive_activation(...)`

- `rar.py`
  - 改进方法 2：RAR 残差自适应重采样
  - 定义训练函数 `train_rar(...)`

- `pinn_utils.py`
  - 公共工具函数
  - 包括采样、tensor 转换、PDE 残差、网格评估、随机种子等

- `train_config.py`
  - 统一训练配置 `TrainConfig`
  - 集中管理超参数（含 RAR 参数）

- `requirements.txt`
  - Python 依赖列表

## 当前方法名映射

- `base`：标准 PINNs
- `ada`：自适应激活
- `rar`：残差自适应重采样

兼容别名：
- `baseline` 会映射到 `base`
- `adaptive_activation` 会映射到 `ada`
