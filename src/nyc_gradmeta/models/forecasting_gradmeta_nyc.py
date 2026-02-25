# Changelog (2026-02-24):
# - Added Bogotá-style 3-stage training (GradMeta -> Adapter -> Together) with stage-aware checkpoints.
# - Moved epi/seed/beta scaling configuration into the calibration NN constructor and removed duplicate scaling.
# - Added private tensor normalization modes and stronger stability/self-check diagnostics.

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".venv/mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nyc_gradmeta.data.seq_dataset import SeqDataset
from nyc_gradmeta.sim.model_utils import (
    CalibNNTwoEncoderThreeOutputs,
    ErrorCorrectionAdapter,
    MetapopulationSEIRMBeta,
    moving_average,
)
from nyc_gradmeta.utils import series_to_supervised


PARAM_ORDER = ("kappa", "symprob", "epsilon", "alpha", "gamma", "delta", "mor")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_population_vector(pop_csv: str, num_patch: int) -> torch.Tensor:
    df = pd.read_csv(pop_csv)
    pop_col = None
    for col in ("population", "Population", "pop", "POPULATION"):
        if col in df.columns:
            pop_col = col
            break
    if pop_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 1:
            pop_col = numeric_cols[0]
        else:
            raise ValueError(
                f"Population CSV must include a population column. Available columns: {df.columns.tolist()}"
            )
    pop = df[pop_col].to_numpy(dtype=np.float32)
    if len(pop) != num_patch:
        raise ValueError(f"Expected {num_patch} population rows, found {len(pop)}.")
    return torch.tensor(pop, dtype=torch.float32)


def load_matrix(csv_path: str, n: int) -> torch.Tensor:
    candidates = []
    for kwargs in ({}, {"index_col": 0}):
        df = pd.read_csv(csv_path, **kwargs)
        mat = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
        candidates.append(mat.shape)
        if mat.shape == (n, n):
            return torch.tensor(mat, dtype=torch.float32)
    raise ValueError(
        f"Expected {n}x{n} numeric matrix at {csv_path}, found candidate shapes {candidates}. "
        "Check header/row-label formatting."
    )


def normalize_private_tensor(private_window: torch.Tensor, private_norm: str) -> torch.Tensor:
    mode = str(private_norm).lower()
    out = private_window.to(torch.float32)
    if mode == "none":
        return out
    if mode == "zscore_time":
        mean = out.mean(dim=1, keepdim=True)
        std = out.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        out = (out - mean) / std
    elif mode == "l2_time":
        out = F.normalize(out, p=2.0, dim=1, eps=1e-8)
    else:
        raise ValueError(f"Unsupported private_norm '{private_norm}'. Expected none|zscore_time|l2_time.")
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def align_private_tensor(
    private_tensor: torch.Tensor | None,
    target_steps: int,
    num_patch: int,
    device: torch.device,
    start_idx: int = 0,
    private_norm: str = "none",
) -> torch.Tensor:
    if private_tensor is None:
        return torch.zeros(num_patch, target_steps, 1, dtype=torch.float32, device=device)

    if private_tensor.dim() == 2:
        private_tensor = private_tensor.unsqueeze(2)
    if private_tensor.dim() != 3:
        raise ValueError(f"private tensor must be [P,T] or [P,T,1], got {tuple(private_tensor.shape)}")

    p, t, c = private_tensor.shape
    if p != num_patch:
        raise ValueError(f"private tensor patch dimension mismatch: expected {num_patch}, got {p}")
    if c != 1:
        raise ValueError(f"private tensor channel dimension must be 1, got {c}")

    start_idx = int(max(0, start_idx))
    if start_idx >= t:
        sliced = torch.zeros(num_patch, target_steps, 1, dtype=torch.float32, device=device)
        return normalize_private_tensor(sliced, private_norm)

    sliced = private_tensor[:, start_idx : start_idx + target_steps, :]
    if sliced.shape[1] < target_steps:
        pad_len = target_steps - sliced.shape[1]
        # Right-align private signal so the most recent private days line up with public train/test window.
        pad = torch.zeros(p, pad_len, 1, dtype=sliced.dtype, device=sliced.device)
        sliced = torch.cat([pad, sliced], dim=1)

    sliced = sliced.to(device=device, dtype=torch.float32)
    # Normalize after slicing so stats match the exact train/eval window.
    return normalize_private_tensor(sliced, private_norm)


def param_model_forward(param_model, private_data, public_X, public_y, device):
    y_np = public_y.detach().cpu().numpy().astype(np.float32)
    n_in, n_out = 4, 1
    supervised = series_to_supervised(y_np.reshape(-1, 1), n_in=n_in, n_out=n_out, dropnan=True)
    if supervised.empty:
        raise ValueError("series_to_supervised returned empty frame; public series is too short.")
    supervised_np = supervised.to_numpy(dtype=np.float32)
    train_X = supervised_np[:, :n_in]
    train_Y = supervised_np[:, -n_out:]

    train_X_t = torch.tensor(train_X, dtype=torch.float32, device=device)
    train_Y_t = torch.tensor(train_Y, dtype=torch.float32, device=device)

    num_patch = private_data.shape[0]
    meta_private = torch.eye(num_patch, device=device)
    meta_public = torch.eye(num_patch, device=device)[0:1, :]

    out, out2, out3, _ = param_model.forward(
        private_data,
        meta_private,
        public_X,
        meta_public,
        train_X_t,
        train_Y_t,
    )
    return out, out2, out3


def enforce_epi_constraints(
    params_weekly: torch.Tensor,
    param_mins: dict[str, float] | None = None,
    param_maxs: dict[str, float] | None = None,
) -> torch.Tensor:
    """
    Ensure biologically valid parameter ranges to prevent simulator blow-ups.
    Clamps to configured min/max bounds when provided (else [0,1]) and enforces
    gamma + mor <= 1 while preserving their ratio.
    """
    if params_weekly is None:
        return params_weekly

    if param_mins is not None and param_maxs is not None:
        lo = torch.tensor([float(param_mins[k]) for k in PARAM_ORDER], device=params_weekly.device, dtype=params_weekly.dtype)
        hi = torch.tensor([float(param_maxs[k]) for k in PARAM_ORDER], device=params_weekly.device, dtype=params_weekly.dtype)
        if torch.any(hi < lo):
            raise ValueError("Invalid epi constraints: param_max must be >= param_min for all parameters.")
        out = torch.max(torch.min(params_weekly, hi), lo)
    else:
        out = params_weekly.clamp(0.0, 1.0)

    gamma_idx = PARAM_ORDER.index("gamma")
    mor_idx = PARAM_ORDER.index("mor")
    gamma = out[:, gamma_idx]
    mor = out[:, mor_idx]

    total = gamma + mor
    too_big = total > 1.0
    if too_big.any():
        scale = total[too_big].clamp(min=1e-8)
        gamma[too_big] = gamma[too_big] / scale
        mor[too_big] = mor[too_big] / scale
        out[:, gamma_idx] = gamma
        out[:, mor_idx] = mor

    return out


def forward_simulator(
    abm,
    params_epi_weekly,
    seed_status,
    adjustment_matrix,
    num_steps,
    enforce_constraints: bool = True,
    param_mins: dict[str, float] | None = None,
    param_maxs: dict[str, float] | None = None,
):
    if enforce_constraints:
        params_epi_weekly = enforce_epi_constraints(
            params_epi_weekly,
            param_mins=param_mins,
            param_maxs=param_maxs,
        )

    preds = []
    max_week_idx = params_epi_weekly.shape[0] - 1

    for t in range(num_steps):
        week_idx = min(t // 7, max_week_idx)
        param_t = params_epi_weekly[week_idx, :]
        _, infections = abm.step(t, param_t, seed_status, adjustment_matrix)
        preds.append(infections)

    preds = torch.stack(preds, dim=0)
    if preds.dim() == 2:
        return preds.sum(dim=1)
    return preds.reshape(preds.shape[0], -1).sum(dim=1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    nz = np.abs(y_true) > 1e-8
    if nz.any():
        mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100.0)
    else:
        mape = float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape}


def save_fit_plot(
    out_path: Path,
    y_true_full: np.ndarray,
    y_pred_full: np.ndarray,
    split_idx: int,
    title: str,
) -> None:
    x = np.arange(len(y_true_full))
    plt.figure(figsize=(12, 5))
    plt.plot(x, y_true_full, label="True cases", color="black", linewidth=1.7)
    plt.plot(x, y_pred_full, label="Predicted cases", color="tab:blue", linewidth=1.7)
    plt.axvline(split_idx - 0.5, color="tab:red", linestyle="--", linewidth=1.5, label="Train/Test split")
    plt.xlabel("Day index")
    plt.ylabel("Daily cases")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def assert_model_outputs(
    params_epi_weekly: torch.Tensor,
    seed_status: torch.Tensor,
    adjustment_matrix: torch.Tensor,
    num_patch: int,
) -> None:
    assert params_epi_weekly.dim() == 2
    assert params_epi_weekly.shape[1] == len(PARAM_ORDER)
    assert seed_status.shape == (num_patch,)
    assert adjustment_matrix.shape == (num_patch, num_patch)
    assert torch.isfinite(params_epi_weekly).all()
    assert torch.isfinite(seed_status).all()
    assert torch.isfinite(adjustment_matrix).all()


def detect_blow_up(preds: torch.Tensor) -> bool:
    if not torch.isfinite(preds).all():
        return True
    max_abs = float(torch.max(torch.abs(preds)).detach().cpu().item())
    return max_abs > 1e7


def maybe_print_default(missing_defaults: list[str], key: str, default_value) -> None:
    missing_defaults.append(f"{key}={default_value}")


def resolve_param_bounds(nyc_cfg: dict, missing_defaults: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    param_ranges = nyc_cfg.get("param_ranges", {})
    param_min_cfg = nyc_cfg.get("param_min")
    param_max_cfg = nyc_cfg.get("param_max")

    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    used_ranges = False
    used_defaults = False

    for key in PARAM_ORDER:
        if isinstance(param_min_cfg, dict) and key in param_min_cfg:
            lo = float(param_min_cfg[key])
        elif isinstance(param_ranges, dict) and key in param_ranges and len(param_ranges[key]) == 2:
            lo = float(param_ranges[key][0])
            used_ranges = True
        else:
            lo = 0.0
            used_defaults = True

        if isinstance(param_max_cfg, dict) and key in param_max_cfg:
            hi = float(param_max_cfg[key])
        elif isinstance(param_ranges, dict) and key in param_ranges and len(param_ranges[key]) == 2:
            hi = float(param_ranges[key][1])
            used_ranges = True
        else:
            hi = 1.0
            used_defaults = True

        mins[key] = lo
        maxs[key] = hi

    if param_min_cfg is None:
        maybe_print_default(missing_defaults, "nyc.param_min", "derived")
    if param_max_cfg is None:
        maybe_print_default(missing_defaults, "nyc.param_max", "derived")
    if used_ranges:
        print("[info] Using nyc.param_ranges fallback for missing nyc.param_min/nyc.param_max entries.")
    if used_defaults:
        print("[info] Missing parameter bounds defaulted to [0,1] for uncovered PARAM_ORDER entries.")

    return mins, maxs


def get_private_stats(private_tensor: torch.Tensor) -> dict:
    return {
        "min": float(private_tensor.min().detach().cpu().item()),
        "max": float(private_tensor.max().detach().cpu().item()),
        "mean": float(private_tensor.mean().detach().cpu().item()),
    }


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nyc.json")
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD used in filenames train/test/private.")
    ap.add_argument("--week", type=int, default=8, help="Training weeks multiplier (train_days = train_days_base * week).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None, help="Backward-compatible base epochs used when stage-specific epochs are not set.")
    ap.add_argument("--epochs_gradmeta", type=int, default=None)
    ap.add_argument("--epochs_adapter", type=int, default=None)
    ap.add_argument("--epochs_together", type=int, default=None)
    ap.add_argument("--stage", choices=["gradmeta", "adapter", "together", "all"], default="all")
    ap.add_argument(
        "--long_train",
        action="store_true",
        help="Scale stage epochs by a long-train multiplier (defaults to 5x or cfg-derived).",
    )
    ap.add_argument("--clip_norm", type=float, default=None, help="Gradient clipping max norm (e.g., 10.0).")
    ap.add_argument("--seed", type=int, default=None, help="Random seed; defaults to config seed.")
    ap.add_argument("--use_adapter", action="store_true", help="Enable optional error-correction adapter.")
    ap.add_argument("--adapter_loss", choices=["mse", "rmse"], default="mse")
    ap.add_argument("--freeze_param_model", dest="freeze_param_model", action="store_true")
    ap.add_argument("--no_freeze_param_model", dest="freeze_param_model", action="store_false")
    ap.set_defaults(freeze_param_model=True)
    ap.add_argument("--self_check", action="store_true", help="Print key tensor shapes and range diagnostics.")
    ap.add_argument("--minimal_outputs", action="store_true", help="Write only run_tag-specific outputs; skip duplicates/plots.")
    ap.add_argument(
        "--no_private",
        action="store_true",
        help="If set, ignore OpenTable private tensor and use zeros [P,T,1].",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    nyc = cfg["nyc"]
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    set_seed(seed)

    missing_defaults: list[str] = []
    param_mins, param_maxs = resolve_param_bounds(nyc, missing_defaults)

    if "seed_min" not in nyc:
        maybe_print_default(missing_defaults, "nyc.seed_min", 0.0)
    if "seed_max" not in nyc:
        maybe_print_default(missing_defaults, "nyc.seed_max", 1.0)
    if "beta_min" not in nyc:
        maybe_print_default(missing_defaults, "nyc.beta_min", 0.0)
    if "beta_max" not in nyc:
        maybe_print_default(missing_defaults, "nyc.beta_max", 1.0)
    if "seed_mode" not in nyc:
        maybe_print_default(missing_defaults, "nyc.seed_mode", "fraction")
    if "private_norm" not in nyc:
        maybe_print_default(missing_defaults, "nyc.private_norm", "zscore_time")
    if "enforce_constraints" not in nyc:
        maybe_print_default(missing_defaults, "nyc.enforce_constraints", True)
    if "epochs_gradmeta" not in nyc:
        maybe_print_default(missing_defaults, "nyc.epochs_gradmeta", "cfg.num_epochs_diff")
    if "epochs_adapter" not in nyc:
        maybe_print_default(missing_defaults, "nyc.epochs_adapter", "cfg.num_epochs_diff")
    if "epochs_together" not in nyc:
        maybe_print_default(missing_defaults, "nyc.epochs_together", "cfg.num_epochs_diff")

    if missing_defaults:
        print("[info] Using default/fallback config values:")
        for msg in sorted(set(missing_defaults)):
            print(f"[info]   - {msg}")

    seed_min = nyc.get("seed_min", 0.0)
    seed_max = nyc.get("seed_max", 1.0)
    beta_min = float(nyc.get("beta_min", 0.0))
    beta_max = float(nyc.get("beta_max", 1.0))
    seed_mode = str(nyc.get("seed_mode", "fraction")).lower()
    private_norm = str(nyc.get("private_norm", "zscore_time")).lower()
    enforce_constraints = bool(nyc.get("enforce_constraints", True))

    if private_norm not in {"none", "zscore_time", "l2_time"}:
        raise ValueError(f"Unsupported nyc.private_norm='{private_norm}'. Expected none|zscore_time|l2_time.")
    if seed_mode not in {"fraction", "count"}:
        raise ValueError(f"Unsupported nyc.seed_mode='{seed_mode}'. Expected fraction|count.")

    device = torch.device(args.device)
    days_head = int(cfg["days_head"])
    test_days = int(nyc["test_days"])
    train_days_base = int(nyc["train_days_base"])
    train_days = train_days_base * int(args.week)
    num_patch = int(nyc["num_patch"])
    online_dir = Path(nyc["paths"]["online_dir"])
    private_dir = Path(nyc["paths"]["private_dir"])
    train_csv = online_dir / f"train_{args.asof}.csv"
    test_csv = online_dir / f"test_{args.asof}.csv"
    private_pt = private_dir / f"opentable_private_lap_{args.asof}.pt"
    out_dir = Path("outputs") / "nyc" / args.asof
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SeqDataset(str(train_csv), nyc["target_name"])
    test_dataset = SeqDataset(str(test_csv), nyc["target_name"])
    train_steps_expected = int(train_dataset.X.shape[0])
    test_steps_expected = int(test_dataset.X.shape[0])
    num_pub_features = int(train_dataset.X.shape[1])
    if test_dataset.X.shape[1] != num_pub_features:
        raise ValueError(
            f"Train/Test feature mismatch: train={train_dataset.X.shape[1]}, test={test_dataset.X.shape[1]}"
        )
    cfg_pub_features = nyc.get("num_pub_features")
    if cfg_pub_features is not None and int(cfg_pub_features) != num_pub_features:
        print(
            f"[info] Overriding config num_pub_features={cfg_pub_features} with "
            f"inferred value from train CSV: {num_pub_features}"
        )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    use_private = not args.no_private
    mode = "public_opentable" if use_private else "public_only"
    run_tag = f"{mode}{'_adapter' if args.use_adapter else ''}"
    private_full = None
    if use_private:
        if not private_pt.exists():
            raise FileNotFoundError(f"Private tensor not found: {private_pt}")
        private_full = torch.load(private_pt, map_location="cpu").to(torch.float32)
    private_start_idx = 0
    if use_private and private_full is not None:
        private_t = private_full.shape[1]
        needed_total = train_steps_expected + test_steps_expected
        # Right-align private history to the combined train+test public window.
        private_start_idx = max(0, int(private_t - needed_total))

    population = load_population_vector(nyc["paths"]["population_csv"], num_patch).to(device)
    migration_matrix = load_matrix(nyc["paths"]["migration_matrix_csv"], num_patch).to(device)
    migration_matrix = migration_matrix.clamp(min=0)
    row_sums = migration_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
    migration_matrix = migration_matrix / row_sums

    sim_params = {
        "num_patch": num_patch,
        "train_days": train_days,
        "test_days": test_days,
        "days_head": days_head,
        "seed_mode": seed_mode,
    }
    abm = MetapopulationSEIRMBeta(sim_params, device, num_patch, migration_matrix, population)

    param_model = CalibNNTwoEncoderThreeOutputs(
        num_patch=num_patch,
        num_pub_features=num_pub_features,
        param_mins=param_mins,
        param_maxs=param_maxs,
        seed_min=seed_min,
        seed_max=seed_max,
        beta_min=beta_min,
        beta_max=beta_max,
    ).to(device)

    error_adapter = None
    if args.use_adapter:
        adapter_hidden_dim = int(nyc.get("adapter_hidden_dim", 64))
        error_adapter = ErrorCorrectionAdapter(hidden_dim=adapter_hidden_dim).to(device)

    lr = float(nyc["learning_rate"])
    weight_decay = float(nyc.get("weight_decay", 0.0))
    adapter_lr = float(nyc.get("adapter_learning_rate", lr))
    together_lr = float(nyc.get("together_learning_rate", lr))

    base_epochs = int(args.epochs) if args.epochs is not None else int(cfg.get("num_epochs_diff", 300))
    epochs_gradmeta = int(args.epochs_gradmeta) if args.epochs_gradmeta is not None else int(nyc.get("epochs_gradmeta", base_epochs))
    epochs_adapter = int(args.epochs_adapter) if args.epochs_adapter is not None else int(nyc.get("epochs_adapter", base_epochs))
    epochs_together = int(args.epochs_together) if args.epochs_together is not None else int(nyc.get("epochs_together", base_epochs))

    if args.long_train:
        long_mult = 5
        if args.epochs is None and "num_epochs_long" in cfg and int(cfg.get("num_epochs_diff", 1)) > 0:
            ratio = int(cfg["num_epochs_long"]) / float(cfg.get("num_epochs_diff", 1))
            long_mult = max(1, int(round(ratio)))
        long_mult = int(nyc.get("long_train_multiplier", long_mult))
        long_mult = max(1, long_mult)
        epochs_gradmeta *= long_mult
        epochs_adapter *= long_mult
        epochs_together *= long_mult
        print(f"[info] --long_train enabled. Multiplying stage epochs by {long_mult}.")

    requested_stages = ["gradmeta", "adapter", "together"] if args.stage == "all" else [args.stage]
    stages_to_run = requested_stages[:]
    if not args.use_adapter:
        if any(s in stages_to_run for s in ("adapter", "together")):
            print("[info] --use_adapter not set; skipping adapter/together stages.")
        stages_to_run = [s for s in stages_to_run if s == "gradmeta"]

    (X_train_b, y_train_b) = next(iter(train_loader))
    X_train = X_train_b.squeeze(0).to(device)
    y_train = y_train_b.squeeze(0).to(device)
    training_num_steps = int(y_train.shape[0])
    y_train_raw_np = y_train.detach().cpu().numpy().astype(np.float64)

    private_train = align_private_tensor(
        private_full,
        training_num_steps,
        num_patch,
        device,
        start_idx=private_start_idx,
        private_norm=private_norm,
    )

    y_np = y_train.detach().cpu().numpy().copy()
    k = min(training_num_steps, train_days)
    if k > 0:
        y_np[:k] = moving_average(y_np[:k], int(cfg["smooth_window"]))
    y_train_s = torch.tensor(y_np, dtype=torch.float32, device=device)

    _, y_test_t = test_dataset[0]
    y_test_np = y_test_t.detach().cpu().numpy().astype(np.float64)
    test_horizon = int(len(y_test_np))
    if test_horizon < 1:
        raise ValueError(f"Test dataset is empty: {test_csv}")
    if test_horizon != days_head:
        print(
            f"[info] Using test horizon from test CSV: {test_horizon} days "
            f"(config days_head={days_head})."
        )
    total_num_steps = training_num_steps + test_horizon

    mse = nn.MSELoss()
    best_loss_gradmeta = float("inf")
    best_loss_adapter = float("inf")
    best_loss_together = float("inf")

    def _warn_blow_up(preds: torch.Tensor, stage_name: str) -> bool:
        if not detect_blow_up(preds):
            return False
        print(
            f"[warn] Blow-up guard triggered in {stage_name}: non-finite or abs(pred)>1e7. "
            "Skipping this optimization step; consider lowering learning_rate."
        )
        return True

    if "gradmeta" in stages_to_run and epochs_gradmeta > 0:
        set_requires_grad(param_model, True)
        if error_adapter is not None:
            set_requires_grad(error_adapter, False)
        opt_gradmeta = torch.optim.Adam(param_model.parameters(), lr=lr, weight_decay=weight_decay)
        pbar = tqdm(range(epochs_gradmeta), desc="Stage1-GradMeta", unit="epoch")

        for _ in pbar:
            param_model.train()
            params_epi_weekly, seed_status, adjustment_matrix = param_model_forward(
                param_model=param_model,
                private_data=private_train,
                public_X=X_train,
                public_y=y_train_s,
                device=device,
            )
            assert_model_outputs(params_epi_weekly, seed_status, adjustment_matrix, num_patch)

            base_preds_train = forward_simulator(
                abm=abm,
                params_epi_weekly=params_epi_weekly,
                seed_status=seed_status,
                adjustment_matrix=adjustment_matrix,
                num_steps=training_num_steps,
                enforce_constraints=enforce_constraints,
                param_mins=param_mins,
                param_maxs=param_maxs,
            )
            if _warn_blow_up(base_preds_train, "Stage1-GradMeta"):
                continue

            loss = torch.sqrt(mse(base_preds_train, y_train_s))
            opt_gradmeta.zero_grad()
            loss.backward()
            if args.clip_norm is not None and args.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(list(param_model.parameters()), args.clip_norm)
            opt_gradmeta.step()

            loss_val = float(loss.detach().cpu().item())
            if loss_val < best_loss_gradmeta:
                best_loss_gradmeta = loss_val
                torch.save(param_model.state_dict(), out_dir / "param_model.pt")
            pbar.set_postfix_str(f"RMSE={best_loss_gradmeta:.4f}")

    if error_adapter is not None and "adapter" in stages_to_run and epochs_adapter > 0:
        if (out_dir / "param_model.pt").exists():
            param_model.load_state_dict(torch.load(out_dir / "param_model.pt", map_location=device))
        else:
            print("[info] Stage2 requested without existing param_model.pt; using current param_model weights.")

        if args.freeze_param_model:
            set_requires_grad(param_model, False)
        param_model.eval()

        set_requires_grad(error_adapter, True)
        opt_adapter = torch.optim.Adam(error_adapter.parameters(), lr=adapter_lr, weight_decay=weight_decay)
        pbar = tqdm(range(epochs_adapter), desc="Stage2-Adapter", unit="epoch")

        for _ in pbar:
            error_adapter.train()
            with torch.no_grad():
                params_epi_weekly, seed_status, adjustment_matrix = param_model_forward(
                    param_model=param_model,
                    private_data=private_train,
                    public_X=X_train,
                    public_y=y_train_s,
                    device=device,
                )
                assert_model_outputs(params_epi_weekly, seed_status, adjustment_matrix, num_patch)

                base_preds_train = forward_simulator(
                    abm=abm,
                    params_epi_weekly=params_epi_weekly,
                    seed_status=seed_status,
                    adjustment_matrix=adjustment_matrix,
                    num_steps=training_num_steps,
                    enforce_constraints=enforce_constraints,
                    param_mins=param_mins,
                    param_maxs=param_maxs,
                )
                residual_target = y_train_s - base_preds_train

            if _warn_blow_up(base_preds_train, "Stage2-Adapter"):
                continue

            residual_pred = error_adapter(base_preds_train)
            if args.adapter_loss == "rmse":
                loss = torch.sqrt(mse(residual_pred, residual_target))
            else:
                loss = mse(residual_pred, residual_target)

            opt_adapter.zero_grad()
            loss.backward()
            if args.clip_norm is not None and args.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(list(error_adapter.parameters()), args.clip_norm)
            opt_adapter.step()

            loss_val = float(loss.detach().cpu().item())
            if loss_val < best_loss_adapter:
                best_loss_adapter = loss_val
                torch.save(error_adapter.state_dict(), out_dir / "error_adapter.pt")
            pbar.set_postfix_str(f"loss={best_loss_adapter:.4f}")

        set_requires_grad(param_model, True)

    if error_adapter is not None and "together" in stages_to_run and epochs_together > 0:
        if (out_dir / "param_model.pt").exists():
            param_model.load_state_dict(torch.load(out_dir / "param_model.pt", map_location=device))
        if (out_dir / "error_adapter.pt").exists():
            error_adapter.load_state_dict(torch.load(out_dir / "error_adapter.pt", map_location=device))
        else:
            print("[info] Stage3 requested without existing error_adapter.pt; using current adapter weights.")

        set_requires_grad(param_model, True)
        set_requires_grad(error_adapter, True)
        opt_together = torch.optim.Adam(
            list(param_model.parameters()) + list(error_adapter.parameters()),
            lr=together_lr,
            weight_decay=weight_decay,
        )
        pbar = tqdm(range(epochs_together), desc="Stage3-Together", unit="epoch")

        for epoch in pbar:
            param_model.train()
            error_adapter.train()

            params_epi_weekly, seed_status, adjustment_matrix = param_model_forward(
                param_model=param_model,
                private_data=private_train,
                public_X=X_train,
                public_y=y_train_s,
                device=device,
            )
            assert_model_outputs(params_epi_weekly, seed_status, adjustment_matrix, num_patch)

            base_preds_train = forward_simulator(
                abm=abm,
                params_epi_weekly=params_epi_weekly,
                seed_status=seed_status,
                adjustment_matrix=adjustment_matrix,
                num_steps=training_num_steps,
                enforce_constraints=enforce_constraints,
                param_mins=param_mins,
                param_maxs=param_maxs,
            )
            residual_pred = error_adapter(base_preds_train)
            preds_train = base_preds_train + residual_pred
            if _warn_blow_up(preds_train, "Stage3-Together"):
                continue

            residual_target = y_train_s - base_preds_train.detach()
            fit_loss = torch.sqrt(mse(preds_train, y_train_s))
            if args.adapter_loss == "rmse":
                aux_loss = torch.sqrt(mse(residual_pred, residual_target))
            else:
                aux_loss = mse(residual_pred, residual_target)

            alpha = float(epoch) / float(max(1, epochs_together))
            loss = (1.0 - alpha) * fit_loss + alpha * aux_loss

            opt_together.zero_grad()
            loss.backward()
            if args.clip_norm is not None and args.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(param_model.parameters()) + list(error_adapter.parameters()),
                    args.clip_norm,
                )
            opt_together.step()

            loss_val = float(loss.detach().cpu().item())
            if loss_val < best_loss_together:
                best_loss_together = loss_val
                torch.save(param_model.state_dict(), out_dir / "param_model.pt")
                torch.save(error_adapter.state_dict(), out_dir / "error_adapter.pt")
            pbar.set_postfix_str(f"loss={best_loss_together:.4f}")

    if args.use_adapter and error_adapter is None:
        print("[info] Adapter requested but unavailable; skipping adapter stages.")

    if (out_dir / "param_model.pt").exists():
        param_model.load_state_dict(torch.load(out_dir / "param_model.pt", map_location=device))
    if error_adapter is not None and (out_dir / "error_adapter.pt").exists():
        error_adapter.load_state_dict(torch.load(out_dir / "error_adapter.pt", map_location=device))

    param_model.eval()
    if error_adapter is not None:
        error_adapter.eval()

    with torch.no_grad():
        params_epi_weekly, seed_status, adjustment_matrix = param_model_forward(
            param_model=param_model,
            private_data=private_train,
            public_X=X_train,
            public_y=y_train_s,
            device=device,
        )
        assert_model_outputs(params_epi_weekly, seed_status, adjustment_matrix, num_patch)

        base_preds_total = forward_simulator(
            abm=abm,
            params_epi_weekly=params_epi_weekly,
            seed_status=seed_status,
            adjustment_matrix=adjustment_matrix,
            num_steps=total_num_steps,
            enforce_constraints=enforce_constraints,
            param_mins=param_mins,
            param_maxs=param_maxs,
        )
        if error_adapter is not None:
            preds_total = base_preds_total + error_adapter(base_preds_total)
        else:
            preds_total = base_preds_total

        forecast = preds_total[-test_horizon:].detach().cpu().numpy()

    preds_total_np = preds_total.detach().cpu().numpy().astype(np.float64)
    y_true_full = np.concatenate([y_train_raw_np, y_test_np], axis=0)
    if len(y_true_full) != len(preds_total_np):
        raise ValueError(
            f"Length mismatch full truth={len(y_true_full)} vs preds={len(preds_total_np)}. "
            "Expected train + test-window alignment."
        )

    train_metrics = compute_metrics(y_train_raw_np, preds_total_np[:training_num_steps])
    test_metrics = compute_metrics(y_test_np, preds_total_np[training_num_steps:])
    metrics = {
        "asof": args.asof,
        "mode": mode,
        "run_tag": run_tag,
        "seed": seed,
        "epochs_gradmeta": epochs_gradmeta,
        "epochs_adapter": epochs_adapter,
        "epochs_together": epochs_together,
        "best_loss_gradmeta": None if not np.isfinite(best_loss_gradmeta) else float(best_loss_gradmeta),
        "best_loss_adapter": None if not np.isfinite(best_loss_adapter) else float(best_loss_adapter),
        "best_loss_together": None if not np.isfinite(best_loss_together) else float(best_loss_together),
        "requested_stage": args.stage,
        "stages_ran": stages_to_run,
        "use_private": use_private,
        "use_adapter": bool(args.use_adapter),
        "adapter_loss": args.adapter_loss,
        "enforce_constraints": bool(enforce_constraints),
        "seed_mode": seed_mode,
        "private_norm": private_norm,
        "train_len": int(training_num_steps),
        "test_len": int(test_horizon),
        "private_start_idx": int(private_start_idx),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    forecast_days = int(test_horizon)
    # Minimal mode: keep only run_tag-specific forecasts; skip legacy duplicates and plots.
    if args.minimal_outputs:
        np.save(out_dir / f"forecast_{forecast_days}d_{run_tag}.npy", forecast)
        pd.DataFrame({"pred_cases": forecast}).to_csv(
            out_dir / f"forecast_{forecast_days}d_{run_tag}.csv", index=False
        )
    else:
        np.save(out_dir / f"forecast_{forecast_days}d.npy", forecast)
        pd.DataFrame({"pred_cases": forecast}).to_csv(out_dir / f"forecast_{forecast_days}d.csv", index=False)
        np.save(out_dir / f"forecast_{forecast_days}d_{run_tag}.npy", forecast)
        pd.DataFrame({"pred_cases": forecast}).to_csv(
            out_dir / f"forecast_{forecast_days}d_{run_tag}.csv", index=False
        )
        # Keep legacy 28-day filenames for backward compatibility.
        if forecast_days == 28:
            np.save(out_dir / "forecast_28d.npy", forecast)
            pd.DataFrame({"pred_cases": forecast}).to_csv(out_dir / "forecast_28d.csv", index=False)
            np.save(out_dir / f"forecast_28d_{run_tag}.npy", forecast)
            pd.DataFrame({"pred_cases": forecast}).to_csv(
                out_dir / f"forecast_28d_{run_tag}.csv", index=False
            )

    fit_df = pd.DataFrame(
        {
            "day_index": np.arange(len(y_true_full), dtype=int),
            "split": np.where(np.arange(len(y_true_full)) < training_num_steps, "train", "test"),
            "true_cases": y_true_full,
            "pred_cases": preds_total_np,
        }
    )
    fit_df.to_csv(out_dir / f"fit_train_test_{run_tag}.csv", index=False)

    if not args.minimal_outputs:
        save_fit_plot(
            out_path=out_dir / f"fit_train_test_{run_tag}.png",
            y_true_full=y_true_full,
            y_pred_full=preds_total_np,
            split_idx=training_num_steps,
            title=f"NYC GradMeta fit ({run_tag}, ASOF={args.asof})",
        )

    with open(out_dir / f"metrics_{run_tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary_row = pd.DataFrame(
        [
            {
                "asof": args.asof,
                "run_tag": run_tag,
                "mode": mode,
                "use_adapter": bool(args.use_adapter),
                "epochs_gradmeta": epochs_gradmeta,
                "epochs_adapter": epochs_adapter,
                "epochs_together": epochs_together,
                "seed": seed,
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_mape": test_metrics["mape"],
            }
        ]
    )
    summary_path = out_dir / "metrics_summary.csv"
    if summary_path.exists():
        prev = pd.read_csv(summary_path)
        prev = prev[~((prev["asof"] == args.asof) & (prev["run_tag"] == run_tag))]
        summary = pd.concat([prev, summary_row], ignore_index=True)
    else:
        summary = summary_row
    summary.to_csv(summary_path, index=False)

    if args.self_check:
        beta_vec = adjustment_matrix.mean(dim=0)
        private_stats = get_private_stats(private_train)
        print(f"X_train shape: {tuple(X_train.shape)}")
        print(f"y_train shape: {tuple(y_train_s.shape)}")
        print(f"private tensor shape: {tuple(private_train.shape)}")
        print(f"private tensor stats after normalization: {private_stats}")
        print(f"params_epi_weekly shape: {tuple(params_epi_weekly.shape)}")
        print(f"seed_status shape: {tuple(seed_status.shape)}")
        print(f"beta_matrix shape: {tuple(adjustment_matrix.shape)}")
        print(f"predictions shape: {tuple(preds_total.shape)}")
        print(f"forecast shape: {tuple(forecast.shape)}")
        print(f"test_horizon: {test_horizon}")
        print(f"run_tag: {run_tag}")

        for i, key in enumerate(PARAM_ORDER):
            pmin = float(params_epi_weekly[:, i].min().detach().cpu().item())
            pmax = float(params_epi_weekly[:, i].max().detach().cpu().item())
            print(f"param[{key}] min/max: {pmin:.6f}/{pmax:.6f}")

        seed_min_v = float(seed_status.min().detach().cpu().item())
        seed_max_v = float(seed_status.max().detach().cpu().item())
        print(f"seed_status min/max: {seed_min_v:.6f}/{seed_max_v:.6f}")
        if seed_mode == "fraction" and seed_max_v <= 1.0:
            implied_infections = seed_status * population * abm.SEED_SCALE
            imp_min = float(implied_infections.min().detach().cpu().item())
            imp_max = float(implied_infections.max().detach().cpu().item())
            print(f"implied initial infections min/max (fraction mode): {imp_min:.6f}/{imp_max:.6f}")
        else:
            print("implied initial infections min/max (count mode): "
                  f"{seed_min_v:.6f}/{seed_max_v:.6f}")

        beta_min_v = float(beta_vec.min().detach().cpu().item())
        beta_max_v = float(beta_vec.max().detach().cpu().item())
        print(f"beta_vec (mean dim=0) min/max: {beta_min_v:.6f}/{beta_max_v:.6f}")

        base_min = float(base_preds_total.min().detach().cpu().item())
        base_max = float(base_preds_total.max().detach().cpu().item())
        preds_min = float(preds_total.min().detach().cpu().item())
        preds_max = float(preds_total.max().detach().cpu().item())
        print(f"base_preds min/max: {base_min:.6f}/{base_max:.6f}")
        print(f"preds min/max: {preds_min:.6f}/{preds_max:.6f}")
        print(f"test metrics: {test_metrics}")

    print("Saved forecast and models to:", out_dir)


if __name__ == "__main__":
    main()
