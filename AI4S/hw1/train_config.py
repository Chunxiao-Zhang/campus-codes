from dataclasses import dataclass


@dataclass
class TrainConfig:
    nu: float
    epochs: int = 4000
    lr: float = 1e-3
    n_f: int = 8000
    n_ic: int = 400
    n_bc: int = 400
    device: str = "cpu"
    seed: int = 42
    log_every: int = 200
    print_every: int = 50
    verbose: bool = True
    case_name: str = ""

    # RAR 相关配置（仅 rar 方法使用）
    rar_every: int = 500
    rar_pool: int = 5000
    rar_topk: int = 500
