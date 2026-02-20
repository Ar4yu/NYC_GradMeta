import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Professor modules (must exist in your repo with same APIs)
from nyc_gradmeta.sim.model_utils import (
    CalibNNTwoEncoderThreeOutputs,
    ErrorCorrectionAdapter,
    moving_average,
    MetapopulationSEIRMBeta,
)
from nyc_gradmeta.utils import series_to_supervised
from nyc_gradmeta.data.seq_dataset import SeqDataset


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_population_vector(pop_csv: str, num_patch: int) -> torch.Tensor:
    df = pd.read_csv(pop_csv)
    if "population" not in df.columns:
        raise ValueError("Population CSV must have 'population' column.")
    pop = df["population"].to_numpy(dtype=np.float32)
    if len(pop) != num_patch:
        raise ValueError(f"Expected {num_patch} population rows, found {len(pop)}.")
    return torch.tensor(pop, dtype=torch.float32)


def load_matrix(csv_path: str, n: int) -> torch.Tensor:
    # Contact matrix CSV has row/column headers (age labels)
    # Read with index_col=0 to skip first column (row labels), then read as numeric
    mat = pd.read_csv(csv_path, index_col=0).to_numpy(dtype=np.float32)
    if mat.shape != (n, n):
        raise ValueError(f"Expected {n}x{n} matrix at {csv_path}, found {mat.shape}.")
    return torch.tensor(mat, dtype=torch.float32)


def param_model_forward(param_model, private_data, public_X, public_y, device):
    """
    Mirrors professor logic you described:
      - scale a 1D series and create supervised train_X/train_Y via series_to_supervised
      - call param_model.forward(...) which returns:
           out (epi params), out2 (seed_status), out3 (adjustment/beta matrix), train_Y
    IMPORTANT: because your exact professor signature may differ, you may need to align arguments.
    """
    # public_X: [T, F_pub], public_y: [T]
    # choose one series to run series_to_supervised on (typically the target)
    y_np = public_y.detach().cpu().numpy().astype(np.float32)

    # Example: use last column or y itself; professor used x[:,:,0] scaled.
    # We'll use y (target) for train_X/train_Y generation (n_in=4).
    # If professor used a different series, swap here.
    n_in = 4
    n_out = 1
    supervised = series_to_supervised(y_np.reshape(-1, 1), n_in=n_in, n_out=n_out, dropnan=True)
    supervised = supervised.to_numpy(dtype=np.float32)

    # train_X: [T - n_in - n_out + 1, n_in]
    # train_Y: [T - n_in - n_out + 1, 1]
    train_X = supervised[:, :n_in]
    train_Y = supervised[:, -n_out:]

    train_X_t = torch.tensor(train_X, dtype=torch.float32, device=device)
    train_Y_t = torch.tensor(train_Y, dtype=torch.float32, device=device)

    # private_data expected as [P, T, 1] and meta as [P, P]
    meta_private = torch.eye(private_data.shape[0], device=device)

    # public encoder metadata: emb_model_2 was initialized with dim_metadata=num_patch,
    # so we need metadata with dimension num_patch. For a single public signal (batch=1),
    # pass [1, num_patch] identity-like structure
    num_patch = private_data.shape[0]
    meta_public = torch.eye(num_patch, device=device).unsqueeze(0)  # [1, P, P] -> mean to [1, P]?
    # Actually, looking at emb_model forward(), it concatenates latent_seqs [B, H] with metadata [B, D]
    # So for batch=1, metadata should be [1, D] where D=num_patch
    # Use first row of identity matrix as metadata for the single public signal
    meta_public = torch.eye(num_patch, device=device)[0:1, :]  # [1, P]

    # The professor model likely takes:
    #   forward(x_private, meta_private, x_public, meta_public, train_X, train_Y)
    # But your description said "data, meta2, train_X, train_Y" are passed as public inputs.
    out, out2, out3, _ = param_model.forward(
        private_data, meta_private,
        public_X, meta_public,
        train_X_t, train_Y_t
    )

    # action_value structure matches what your professor code expects downstream
    return out, out2, out3


def forward_simulator(abm, params_epi_weekly, seed_status, adjustment_matrix, num_steps, device):
    """
    Daily stepping. Weekly params via time_step//7.
    Returns: total_cases_pred [T] torch tensor

    Note: When simulating beyond the number of weeks with learned parameters,
    use the last available week's parameters.
    """
    preds = []
    max_week_idx = params_epi_weekly.shape[0] - 1  # Last available week index

    for t in range(num_steps):
        week_idx = min(t // 7, max_week_idx)  # Use last week if beyond available weeks
        param_t = params_epi_weekly[week_idx, :]  # piecewise constant weekly
        deaths, infections = abm.step(t, param_t, seed_status, adjustment_matrix)  # returns (deaths, infections) per patch
        preds.append(infections)

    preds = torch.stack(preds, dim=0)  # [T, P]
    # Sum across patches to citywide
    if preds.dim() == 2:
        total = preds.sum(dim=1)  # [T]
    else:
        # If abm returns [P] per step, stack gives [T, P] anyway; this is just a safeguard
        total = preds.reshape(preds.shape[0], -1).sum(dim=1)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nyc.json")
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD used in filenames train/test/private.")
    ap.add_argument("--week", type=int, default=8, help="Training weeks multiplier (train_days = train_days_base * week).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument(
        "--no_private",
        action="store_true",
        help="If set, ignore OpenTable private tensor and train only on public master data.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    nyc = cfg["nyc"]

    device = torch.device(args.device)

    days_head = int(cfg["days_head"])
    test_days = int(nyc["test_days"])
    train_days_base = int(nyc["train_days_base"])
    train_days = train_days_base * int(args.week)

    # Paths
    online_dir = Path(nyc["paths"]["online_dir"])
    private_dir = Path(nyc["paths"]["private_dir"])

    train_csv = online_dir / f"train_{args.asof}.csv"
    test_csv = online_dir / f"test_{args.asof}.csv"
    private_pt = private_dir / f"opentable_private_lap_{args.asof}.pt"

    # Load datasets (public)
    train_dataset = SeqDataset(str(train_csv), nyc["target_name"])
    test_dataset = SeqDataset(str(test_csv), nyc["target_name"])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    use_private = not args.no_private

    # Load private tensor if requested; otherwise we will create zeros on the fly.
    private_full = None
    if use_private:
        private_full = torch.load(private_pt)  # [P, T_total]
        private_full = private_full.unsqueeze(2).to(device)  # [P, T_total, 1]

    # Load simulator inputs
    num_patch = int(nyc["num_patch"])
    population = load_population_vector(nyc["paths"]["population_csv"], num_patch).to(device)
    migration_matrix = load_matrix(nyc["paths"]["migration_matrix_csv"], num_patch).to(device)

    # Build simulator
    sim_params = {
        "num_patch": num_patch,
        "train_days": train_days,
        "test_days": test_days,
        "days_head": days_head
    }
    abm = MetapopulationSEIRMBeta(sim_params, device, num_patch, migration_matrix, population)

    # Build parameter model (neural net that outputs weekly epi params,
    # seed_status, and beta matrix for the SEIRM simulator)
    param_model = CalibNNTwoEncoderThreeOutputs(
        num_patch=num_patch,
        num_pub_features=int(nyc["num_pub_features"]),
    ).to(device)

    # Error-correction adapter that learns residuals on top of the mechanistic
    # simulator output (NN -> SEIRM -> adapter).
    adapter_hidden_dim = int(nyc.get("adapter_hidden_dim", 64))
    error_adapter = ErrorCorrectionAdapter(hidden_dim=adapter_hidden_dim).to(device)

    lr = float(nyc["learning_rate"])
    opt = torch.optim.Adam(
        list(param_model.parameters()) + list(error_adapter.parameters()),
        lr=lr,
        weight_decay=float(nyc.get("weight_decay", 0.0)),
    )
    mse = nn.MSELoss()

    num_epochs = int(args.epochs) if args.epochs is not None else int(cfg["num_epochs_diff"])

    # ===== Training loop (full-batch) =====
    best_loss = float("inf")
    best_state = None

    for epoch in range(num_epochs):
        param_model.train()
        for (X_train, y_train) in train_loader:
            # X_train: [1, T, F], y_train: [1, T]
            X_train = X_train.squeeze(0).to(device)  # [T, F]
            y_train = y_train.squeeze(0).to(device)  # [T]

            # Slice or build private data to match training length
            training_num_steps = X_train.shape[0]
            if use_private and private_full is not None:
                private_train = private_full[:, :training_num_steps, :]  # [P, T_train, 1]
            else:
                private_train = torch.zeros(num_patch, training_num_steps, 1, device=device)

            # optional: smoothing head like professor (apply on y_train first train_days)
            y_np = y_train.detach().cpu().numpy().copy()
            if train_days > 1 and len(y_np) >= train_days:
                y_np[:train_days] = moving_average(y_np[:train_days], int(cfg["smooth_window"]))
            y_train_s = torch.tensor(y_np, dtype=torch.float32, device=device)

            # training_num_steps = length of training series
            training_num_steps = y_train_s.shape[0]
            total_num_steps = training_num_steps + days_head

            # forward param model
            params_epi_weekly, seed_status, adjustment_matrix = param_model_forward(
                param_model=param_model,
                private_data=private_train,
                public_X=X_train,
                public_y=y_train_s,
                device=device
            )

            # forward simulator for training_num_steps only (match professor training loss)
            base_preds_train = forward_simulator(
                abm=abm,
                params_epi_weekly=params_epi_weekly,
                seed_status=seed_status,
                adjustment_matrix=adjustment_matrix,
                num_steps=training_num_steps,
                device=device
            )  # [T]
            # error-correction adapter on top of mechanistic simulator
            corr_train = error_adapter(base_preds_train)  # [T]
            preds_train = base_preds_train + corr_train   # [T]

            loss = torch.sqrt(mse(preds_train, y_train_s))

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_val = float(loss.detach().cpu().item())
            if loss_val < best_loss:
                best_loss = loss_val
                best_state = {k: v.detach().cpu().clone() for k, v in param_model.state_dict().items()}

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}  train_RMSE={best_loss:.4f}")

    # Restore best
    if best_state is not None:
        param_model.load_state_dict(best_state)

    # ===== Inference: predict last 28 days =====
    param_model.eval()

    # Reload train batch for inference construction
    (X_train, y_train) = next(iter(train_loader))
    X_train = X_train.squeeze(0).to(device)  # [T_train, F]
    y_train = y_train.squeeze(0).to(device)  # [T_train]

    # Slice or build private data to match training length
    training_num_steps = X_train.shape[0]
    if use_private and private_full is not None:
        private_train = private_full[:, :training_num_steps, :]  # [P, T_train, 1]
    else:
        private_train = torch.zeros(num_patch, training_num_steps, 1, device=device)

    y_np = y_train.detach().cpu().numpy().copy()
    if train_days > 1 and len(y_np) >= train_days:
        y_np[:train_days] = moving_average(y_np[:train_days], int(cfg["smooth_window"]))
    y_train_s = torch.tensor(y_np, dtype=torch.float32, device=device)

    training_num_steps = y_train_s.shape[0]
    total_num_steps = training_num_steps + days_head

    with torch.no_grad():
        params_epi_weekly, seed_status, adjustment_matrix = param_model_forward(
            param_model=param_model,
            private_data=private_train,
            public_X=X_train,
            public_y=y_train_s,
            device=device,
        )

        base_preds_total = forward_simulator(
            abm=abm,
            params_epi_weekly=params_epi_weekly,
            seed_status=seed_status,
            adjustment_matrix=adjustment_matrix,
            num_steps=total_num_steps,
            device=device,
        )  # [training_num_steps + days_head]

        corr_total = error_adapter(base_preds_total)  # [T_total]
        preds_total = base_preds_total + corr_total

        forecast = preds_total[-days_head:].detach().cpu().numpy()  # last 28 days

    # Save outputs (forecasts + trained model weights)
    out_dir = Path("outputs") / "nyc" / args.asof
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "forecast_28d.npy", forecast)
    pd.DataFrame({"pred_cases": forecast}).to_csv(out_dir / "forecast_28d.csv", index=False)

    # Save model weights for reproducibility / later analysis
    torch.save(param_model.state_dict(), out_dir / "param_model.pt")
    torch.save(error_adapter.state_dict(), out_dir / "error_adapter.pt")

    print("Saved forecast and models to:", out_dir)


if __name__ == "__main__":
    main()