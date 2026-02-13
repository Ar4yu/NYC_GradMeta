"""
NYC GradMeta: core simulator + sequence embedding modules.

This file is intentionally *NYC-focused*:
- Keeps metapopulation SEIRM(-Beta) simulator
- Keeps embedding/decoder modules used by forecasting model
- DOES NOT include old county/flu fetchers or dataset loaders
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SMOOTH_WINDOW = 7


class MetapopulationSEIRM:
    """
    Metapopulation SEIRM with a single scalar beta and a learnable adjustment_matrix
    that modifies migration/coupling at t=0 (legacy behavior).
    """
    def __init__(self, params, device, num_patches, migration_matrix, num_agents):
        self.device = device
        self.num_patches = num_patches
        self.state = torch.zeros((num_patches, 5), device=self.device)
        self.params = params
        self.migration_matrix = migration_matrix.to(self.device)
        self.num_agents = num_agents.to(self.device)

    def init_compartments(self, seed_infection_status=None):
        seed_infection_status = seed_infection_status or {}
        initial_infections = torch.zeros((self.num_patches), device=self.device)
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

            # Legacy preprocessing from original codebase:
            # - COVID uses diag(adjustment_matrix) with clipping + row-normalization
            # - else uses additive clip
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
    Metapopulation SEIRM where 'adjustment_matrix' is actually a beta_matrix.
    Original code uses beta_matrix.mean(dim=0) to get a per-patch beta.
    """
    def __init__(self, params, device, num_patches, migration_matrix, num_agents, seed_infection_status=None):
        self.device = device
        self.num_patches = num_patches
        self.state = torch.zeros((num_patches, 5), device=self.device)
        self.params = params
        self.migration_matrix = migration_matrix.to(self.device)
        self.num_agents = num_agents.to(self.device)
        self.seed_infection_status = seed_infection_status or {}

    def init_compartments(self, seed_infection_status=None):
        seed_infection_status = seed_infection_status or {}
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


class TransformerAttn(nn.Module):
    def __init__(self, dim_in=40, value_dim=40, key_dim=40) -> None:
        super().__init__()
        self.value_layer = nn.Linear(dim_in, value_dim)
        self.query_layer = nn.Linear(dim_in, value_dim)
        self.key_layer = nn.Linear(dim_in, key_dim)

    def forward(self, seq):
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.softmax(weights, -1)
        return (weights @ keys).transpose(1, 0)

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
    def __init__(
        self,
        dim_seq_in: int = 5,
        dim_metadata: int = 3,
        rnn_out: int = 40,
        dim_out: int = 50,
        n_layers: int = 1,
        bidirectional: bool = False,
        attn=TransformerAttn,
        dropout=0.0,
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
            # if no metadata, just return latent
            out = latent_seqs
        return out, encoder_hidden


class DecodeSeq(nn.Module):
    def __init__(
        self,
        dim_seq_in: int = 5,
        dim_metadata: int = 3,
        rnn_out: int = 40,
        dim_out: int = 5,
        n_layers: int = 1,
        bidirectional: bool = False,
        dropout=0.0,
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
        inputs = Hi_data.transpose(1, 0)
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


def moving_average(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().values

