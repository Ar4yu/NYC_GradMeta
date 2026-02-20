"""
NYC GradMeta: core simulator + sequence embedding modules + calibration network.

This file is intentionally NYC-focused:
- Metapopulation SEIRM(-Beta) simulators
- Attention-based sequence encoders/decoder
- CalibNNTwoEncoderThreeOutputs (private encoder + public encoder -> epi params, seed, beta matrix)
- Moving average utility

It avoids legacy county/flu fetchers and dataset loaders.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SMOOTH_WINDOW = 7


# =========================
# 1) Mechanistic simulators
# =========================

class MetapopulationSEIRM:
    """
    Metapopulation SEIRM with a scalar beta and an "adjustment_matrix" that
    modifies migration/coupling at t=0 (legacy behavior).

    Returns (NEW_DEATHS_TODAY, NEW_INFECTIONS_TODAY) per patch.
    """

    def __init__(self, params, device, num_patches, migration_matrix, num_agents):
        self.device = device
        self.num_patches = int(num_patches)
        self.state = torch.zeros((self.num_patches, 5), device=self.device)
        self.params = params
        self.migration_matrix = migration_matrix.to(self.device)
        self.num_agents = num_agents.to(self.device)

    def init_compartments(self, seed_infection_status=None):
        if seed_infection_status is None:
            seed_infection_status = {}
        initial_infections = torch.zeros((self.num_patches), device=self.device)

        # NOTE: original professor code enumerated over the container; keep same semantics:
        for idx, value in enumerate(seed_infection_status):
            initial_infections[idx] = value

        initial_conditions = torch.zeros((self.num_patches, 5), device=self.device)
        initial_conditions[:, 2] = initial_infections
        initial_conditions[:, 0] = self.num_agents - initial_infections
        self.state = initial_conditions

    def step(self, t, values, seed_status, adjustment_matrix):
        params = {
            "beta": values[0],
            "kappa": values[1],
            "symprob": values[2],
            "epsilon": values[3],
            "alpha": values[4],
            "gamma": values[5],
            "delta": values[6],
            "mor": values[7],
            "seed_status": seed_status.long(),
            "adjustment_matrix": adjustment_matrix,
        }

        if t == 0:
            self.init_compartments(seed_infection_status=params["seed_status"])

            disease = str(self.params.get("disease", "COVID"))
            if "COVID" in disease:
                self.migration_matrix = torch.clip(
                    self.migration_matrix + 0.1 * torch.diag(params["adjustment_matrix"]),
                    0,
                    1,
                )
                self.migration_matrix = self.migration_matrix / self.migration_matrix.sum(dim=1, keepdim=True)
            else:
                self.migration_matrix = torch.clip(
                    self.migration_matrix + params["adjustment_matrix"],
                    1e-29,
                    1,
                )

        N_eff = self.migration_matrix.T @ self.num_agents
        I_eff = self.migration_matrix.T @ self.state[:, 2].clone()
        E_eff = self.migration_matrix.T @ self.state[:, 1].clone()

        beta_j_eff = torch.nan_to_num((I_eff / N_eff) * params["beta"])
        beta_j_eff = beta_j_eff * ((1 - params["kappa"]) * (1 - params["symprob"]) + params["symprob"])

        E_beta_j_eff = torch.nan_to_num((E_eff / N_eff) * params["beta"])
        E_beta_j_eff = E_beta_j_eff * (1 - params["epsilon"])

        inf_force = self.migration_matrix @ (beta_j_eff + E_beta_j_eff)
        new_inf = torch.minimum(inf_force * self.state[:, 0].clone(), self.state[:, 0].clone())

        self.state[:, 0] = self.state[:, 0].clone() - new_inf + params["delta"] * self.state[:, 3].clone()
        self.state[:, 1] = new_inf + (1 - params["alpha"]) * self.state[:, 1].clone()
        self.state[:, 2] = params["alpha"] * self.state[:, 1].clone() + (1 - params["gamma"] - params["mor"]) * self.state[:, 2].clone()
        self.state[:, 3] = params["gamma"] * self.state[:, 2].clone() + (1 - params["delta"]) * self.state[:, 3].clone()
        self.state[:, 4] = params["mor"] * self.state[:, 2].clone()

        NEW_INFECTIONS_TODAY = self.state[:, 2].clone()
        NEW_DEATHS_TODAY = self.state[:, 4].clone()
        return NEW_DEATHS_TODAY, NEW_INFECTIONS_TODAY


class MetapopulationSEIRMBeta:
    """
    Metapopulation SEIRM where the "adjustment_matrix" passed to step() is a beta_matrix.
    Original behavior: beta_matrix.mean(dim=0) produces a per-patch beta vector.

    Returns (NEW_DEATHS_TODAY, NEW_INFECTIONS_TODAY) per patch.
    """

    def __init__(self, params, device, num_patches, migration_matrix, num_agents, seed_infection_status=None):
        self.device = device
        self.num_patches = int(num_patches)
        self.state = torch.zeros((self.num_patches, 5), device=self.device)
        self.params = params
        self.migration_matrix = migration_matrix.to(self.device)
        self.num_agents = num_agents.to(self.device)
        self.seed_infection_status = seed_infection_status if seed_infection_status is not None else {}

    def init_compartments(self, seed_infection_status=None):
        if seed_infection_status is None:
            seed_infection_status = {}
        initial_infections = torch.zeros((self.num_patches), device=self.device)
        for idx, value in enumerate(seed_infection_status):
            initial_infections[idx] = value

        initial_conditions = torch.zeros((self.num_patches, 5), device=self.device)
        initial_conditions[:, 2] = initial_infections
        initial_conditions[:, 0] = self.num_agents - initial_infections
        self.state = initial_conditions

    def step(self, t, values, seed_status, beta_matrix):
        params = {
            "kappa": values[0],
            "symprob": values[1],
            "epsilon": values[2],
            "alpha": values[3],
            "gamma": values[4],
            "delta": values[5],
            "mor": values[6],
            "seed_status": seed_status.long(),
            "beta_matrix": beta_matrix,
        }

        if t == 0:
            self.init_compartments(seed_infection_status=params["seed_status"])

        N_eff = self.migration_matrix.T @ self.num_agents
        I_eff = self.migration_matrix.T @ self.state[:, 2].clone()
        E_eff = self.migration_matrix.T @ self.state[:, 1].clone()

        beta_vec = params["beta_matrix"].mean(dim=0)  # legacy behavior

        beta_j_eff = torch.nan_to_num((I_eff / N_eff) * beta_vec)
        beta_j_eff = beta_j_eff * ((1 - params["kappa"]) * (1 - params["symprob"]) + params["symprob"])

        E_beta_j_eff = torch.nan_to_num((E_eff / N_eff) * beta_vec)
        E_beta_j_eff = E_beta_j_eff * (1 - params["epsilon"])

        inf_force = self.migration_matrix @ (beta_j_eff + E_beta_j_eff)
        new_inf = torch.minimum(inf_force * self.state[:, 0].clone(), self.state[:, 0].clone())

        self.state[:, 0] = self.state[:, 0].clone() - new_inf + params["delta"] * self.state[:, 3].clone()
        self.state[:, 1] = new_inf + (1 - params["alpha"]) * self.state[:, 1].clone()
        self.state[:, 2] = params["alpha"] * self.state[:, 1].clone() + (1 - params["gamma"] - params["mor"]) * self.state[:, 2].clone()
        self.state[:, 3] = params["gamma"] * self.state[:, 2].clone() + (1 - params["delta"]) * self.state[:, 3].clone()
        self.state[:, 4] = params["mor"] * self.state[:, 2].clone()

        NEW_INFECTIONS_TODAY = self.state[:, 2].clone()
        NEW_DEATHS_TODAY = self.state[:, 4].clone()
        return NEW_DEATHS_TODAY, NEW_INFECTIONS_TODAY


# =========================
# 2) Attention modules
# =========================

class TransformerAttn(nn.Module):
    """
    Transformer-like self-attention block used in the professor code.
    Expects seq in shape [SeqLen, Batch, Hidden].
    """

    def __init__(self, dim_in=40, value_dim=40, key_dim=40) -> None:
        super().__init__()
        self.value_layer = nn.Linear(dim_in, value_dim)
        self.query_layer = nn.Linear(dim_in, value_dim)
        self.key_layer = nn.Linear(dim_in, key_dim)

    def forward(self, seq):
        seq_in = seq.transpose(0, 1)  # [B, T, H]
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.softmax(weights, -1)
        return (weights @ keys).transpose(1, 0)  # [T, B, H]

    def forward_mask(self, seq, mask):
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.exp(weights)
        weights = (weights.transpose(1, 2) * mask.transpose(1, 0)).transpose(1, 2)
        weights = weights / (weights.sum(-1, keepdim=True))
        return (weights @ keys).transpose(1, 0) * mask


class EmbedAttenSeq(nn.Module):
    """
    Encoder: GRU + attention + MLP head that can incorporate metadata.
    """

    def __init__(
        self,
        dim_seq_in: int = 5,
        dim_metadata: int = 3,
        rnn_out: int = 40,
        dim_out: int = 50,
        n_layers: int = 1,
        bidirectional: bool = False,
        attn=TransformerAttn,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim_seq_in = dim_seq_in
        self.dim_metadata = dim_metadata
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.attn_layer = attn(self.rnn_out, self.rnn_out, self.rnn_out)

        self.out_layer = nn.Sequential(
            nn.Linear(self.rnn_out + self.dim_metadata, self.dim_out),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.out_layer.apply(init_weights)

    def forward_mask(self, seqs, metadata, mask):
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = self.attn_layer.forward_mask(latent_seqs, mask)
        latent_seqs = latent_seqs.sum(0)
        out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        return out

    def forward(self, seqs, metadata=None):
        latent_seqs, encoder_hidden = self.rnn(seqs)
        latent_seqs = self.attn_layer(latent_seqs).sum(0)
        if metadata is not None:
            out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        else:
            out = latent_seqs
        return out, encoder_hidden


class DecodeSeq(nn.Module):
    """
    Decoder GRU that consumes a time grid (Hi_data) and a context embedding,
    producing a latent sequence used by output heads.
    """

    def __init__(
        self,
        dim_seq_in: int = 1,
        rnn_out: int = 40,
        dim_out: int = 5,
        n_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim_seq_in = dim_seq_in
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.embed_input = nn.Linear(self.dim_seq_in, self.rnn_out)
        self.attn_combine = nn.Linear(2 * self.rnn_out, self.rnn_out)

        self.rnn = nn.GRU(
            input_size=self.rnn_out,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.out_layer = nn.Sequential(
            nn.Linear(self.rnn_out, self.dim_out),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.out_layer.apply(init_weights)
        self.embed_input.apply(init_weights)
        self.attn_combine.apply(init_weights)

    def forward(self, Hi_data, encoder_hidden, context):
        inputs = Hi_data.transpose(1, 0)  # [T, B, 1]

        # Mirror professor handling: use hidden states after index 2
        if self.bidirectional:
            h0 = encoder_hidden[2:]
        else:
            h0 = encoder_hidden[2:].sum(0).unsqueeze(0)

        inputs = self.embed_input(inputs)
        context = context.repeat(inputs.shape[0], 1, 1)
        inputs = torch.cat((inputs, context), 2)
        inputs = self.attn_combine(inputs)

        latent_seqs = self.rnn(inputs, h0)[0]
        latent_seqs = latent_seqs.transpose(1, 0)
        latent_seqs = self.out_layer(latent_seqs)
        return latent_seqs


# ==========================================
# 3) NYC calibration network (Section 4)
# ==========================================

class CalibNNTwoEncoderThreeOutputs(nn.Module):
    """
    NYC version of professor's CalibNNTwoEncoderThreeOutputs.

    Inputs:
      x_private: [P, T, 1]   (e.g., OpenTable per-patch)
      meta_private: [P, P]   (often identity)
      x_public: [T, F] or [1, T, F] depending on caller
      meta_public: [P, P] (kept consistent with your current wrapper; can be identity)
      train_X: [N, n_in] or [N, n_in, 1] (passed-through; only used in legacy LSTM adapter)
      train_Y: [N, 1]        (returned as the 4th output in professor code)

    Outputs:
      out:  [W, 7]    weekly epi params (kappa, symprob, epsilon, alpha, gamma, delta, mor)
      out2: [P]       seed_status vector (scaled to [0, 1] then used as long in sim)
      out3: [P, P]    beta matrix in [0, 1] (used by MetapopulationSEIRMBeta via mean(dim=0))
      train_Y: passed-through for compatibility
    """

    def __init__(
        self,
        num_patch: int,
        num_pub_features: int,
        training_weeks: int = 0,
        hidden_dim: int = 64,
        out_dim: int = 7,
        n_layers: int = 2,
        bidirectional: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.num_patch = int(num_patch)
        self.num_pub_features = int(num_pub_features)
        self.training_weeks = int(training_weeks)  # if 0, we infer from inputs in forward()

        self.device = device  # optional; used only for constructing some tensors
        out_layer_dim = 32

        # Private encoder: expects seqs in [T, P, 1] after transpose in forward
        self.emb_model = EmbedAttenSeq(
            dim_seq_in=1,
            dim_metadata=self.num_patch,   # meta_private is [P,P] row per patch
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        # Public encoder: expects seqs in [T, B, F] after transpose in forward
        # Your current wrapper uses meta_public = I[P], so dim_metadata=num_patch.
        self.emb_model_2 = EmbedAttenSeq(
            dim_seq_in=self.num_pub_features,
            dim_metadata=self.num_patch,
            rnn_out=hidden_dim,
            dim_out=hidden_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
        )

        self.decoder = DecodeSeq(
            dim_seq_in=1,
            rnn_out=hidden_dim,
            dim_out=out_layer_dim,
            n_layers=1,
            bidirectional=True,
        )

        # Output heads (mirror professor structure)
        self.out_layer = nn.Sequential(
            nn.Linear(out_layer_dim, out_layer_dim // 2),
            nn.ReLU(),
            nn.Linear(out_layer_dim // 2, out_dim),
        )

        self.out_layer2 = nn.Sequential(
            nn.Linear(out_layer_dim, out_layer_dim // 2),
            nn.ReLU(),
            nn.Linear(out_layer_dim // 2, self.num_patch),
        )

        self.out_layer3 = nn.Sequential(
            nn.Linear(out_layer_dim, out_layer_dim // 2),
            nn.ReLU(),
            nn.Linear(out_layer_dim // 2, self.num_patch * self.num_patch),
        )

        self.sigmoid = nn.Sigmoid()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.out_layer.apply(init_weights)
        self.out_layer2.apply(init_weights)
        self.out_layer3.apply(init_weights)

    def forward(
        self,
        x_private: torch.Tensor,
        meta_private: torch.Tensor,
        x_public: torch.Tensor,
        meta_public: torch.Tensor,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # ---- normalize shapes to match professor expectations ----
        # x_private: [P, T, 1] -> [T, P, 1]
        if x_private.dim() != 3:
            raise ValueError(f"x_private expected [P,T,1], got {tuple(x_private.shape)}")
        x_priv_seq = x_private.transpose(1, 0)  # [T, P, 1]

        # meta_private: [P,P] (ok)
        if meta_private.dim() != 2:
            raise ValueError(f"meta_private expected [P,P], got {tuple(meta_private.shape)}")

        # x_public: allow [T,F] or [1,T,F] -> make [T,1,F] then transpose to [T,1,F] for GRU
        if x_public.dim() == 2:
            x_pub = x_public.unsqueeze(0)  # [1, T, F]
        elif x_public.dim() == 3:
            x_pub = x_public
        else:
            raise ValueError(f"x_public expected [T,F] or [B,T,F], got {tuple(x_public.shape)}")

        # professor passes x_2.transpose(1,0) => [T,B,F]
        x_pub_seq = x_pub.transpose(1, 0)  # [T, B, F]

        # ---- encoders ----
        x_embeds, encoder_hidden = self.emb_model.forward(x_priv_seq, meta_private)     # x_embeds: [P, H]
        x_embeds_2, encoder_hidden_2 = self.emb_model_2.forward(x_pub_seq, meta_public)  # x_embeds_2: [B, H]

        # ---- concatenate embeddings (mirror professor behavior) ----
        # In professor code:
        #   x_embeds = cat([x_embeds, x_embeds_2.mean(dim=0).unsqueeze(0)], dim=0)
        # Here x_embeds is [P,H], x_embeds_2 is [B,H] -> mean over B gives [H]
        x_embeds_cat = torch.cat([x_embeds, x_embeds_2.mean(dim=0).unsqueeze(0)], dim=0)  # [P+1, H]

        # encoder_hidden: [layers*dir, batch=P, hidden] ; encoder_hidden_2: [layers*dir, batch=B, hidden]
        # professor does:
        #   encoder_hidden = cat([encoder_hidden, encoder_hidden_2.mean(dim=1).unsqueeze(1)], dim=1)
        enc2_mean = encoder_hidden_2.mean(dim=1).unsqueeze(1)  # [layers*dir, 1, hidden]
        encoder_hidden_cat = torch.cat([encoder_hidden, enc2_mean], dim=1)  # [layers*dir, P+1, hidden]

        # ---- time grid for decoder ----
        # professor uses arange(1, training_weeks+WEEKS_AHEAD+1); for NYC we just decode
        # weekly parameters for however many weeks the current training horizon implies.
        # If training_weeks not set, infer from x_public length (T) by ceil(T/7).
        T = x_pub_seq.shape[0]
        inferred_weeks = int(math.ceil(T / 7.0))
        W = self.training_weeks if self.training_weeks > 0 else inferred_weeks

        time_seq = torch.arange(1, W + 1, device=x_private.device).repeat(x_embeds_cat.shape[0], 1).unsqueeze(2)
        Hi_data = (time_seq - time_seq.min()) / (time_seq.max() - time_seq.min() + 1e-8)

        # ---- decode ----
        # decoder expects context in [SeqLenContext, Batch, Hidden] style,
        # but professor passes x_embeds (context) shaped [P+1,H] and decoder repeats it.
        # Our DecodeSeq expects context shaped [B_ctx, 1, H] initially.
        # After this context is transposed in decoder, we need [1, P+1, H] so that
        # repeat(W, 1, 1) gives [W, P+1, H] matching inputs shape [W, P+1, H_embeddings]
        context = x_embeds_cat.unsqueeze(0)  # [1, P+1, H]
        emb = self.decoder(Hi_data, encoder_hidden_cat, context)  # [P+1, W, out_layer_dim]

        # professor: out = out_layer(emb); out = mean(out, dim=0)
        out = self.out_layer(emb)              # [P+1, W, out_dim]
        out = torch.mean(out, dim=0)           # [W, out_dim]
        out = self.sigmoid(out)                # bound to [0,1] (NYC: keep simple; scale outside if needed)

        # professor: emb_mean = mean(emb, dim=0); emb_mean = emb_mean[-1,:]
        emb_mean = torch.mean(emb, dim=0)      # [W, out_layer_dim]
        emb_last = emb_mean[-1, :]             # [out_layer_dim]

        out2 = self.sigmoid(self.out_layer2(emb_last))                          # [P]
        out3 = self.sigmoid(self.out_layer3(emb_last).reshape(self.num_patch, self.num_patch))  # [P,P]

        # return same 4-tuple API as professor
        return out, out2, out3, train_Y


# =========================
# 4) Error-correction adapter
# =========================


class ErrorCorrectionAdapter(nn.Module):
    """
    Lightweight sequence model that learns a residual correction
    on top of the mechanistic SEIRM simulator output.

    Inputs:
      preds: [T] or [B, T] daily city-wide cases from the simulator

    Output:
      corr:  same shape as preds; additive correction term
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.out = nn.Linear(out_dim, 1)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.out.apply(init_weights)

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        # Normalize to [B, T, 1] for GRU
        if preds.dim() == 1:
            x = preds.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            squeeze_batch = True
        elif preds.dim() == 2:
            x = preds.unsqueeze(-1)  # [B, T, 1]
            squeeze_batch = False
        else:
            raise ValueError(f"ErrorCorrectionAdapter expected [T] or [B,T], got {tuple(preds.shape)}")

        h, _ = self.rnn(x)
        corr = self.out(h).squeeze(-1)  # [B, T]

        if squeeze_batch:
            corr = corr.squeeze(0)  # [T]

        return corr


# =========================
# 5) Small utils
# =========================

def moving_average(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().values