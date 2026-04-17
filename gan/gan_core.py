import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import logging
import json
from scipy.stats import ks_2samp
import random

# Reuse logic from generate_gan_data.py but structured for reuse

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, noise_dim=32, dropout=0.1):
        super().__init__()
        self.noise_dim = noise_dim
        channels = input_size + noise_dim
        self.backbone = nn.Sequential(
            ConvBlock(channels, hidden_size, dilation=1),
            ConvBlock(hidden_size, hidden_size, dilation=2),
            ConvBlock(hidden_size, hidden_size, dilation=4),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, history, noise):
        x = torch.cat((history, noise), dim=2).transpose(1, 2)
        hidden = self.backbone(x).transpose(1, 2)
        generated = self.head(hidden[:, -1, :])
        return generated.unsqueeze(1)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.features = nn.Sequential(
            sn(nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2),
            sn(nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=2, dilation=2)),
            nn.LeakyReLU(0.2),
            sn(nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=4, dilation=4)),
            nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            sn(nn.Linear(hidden_size, hidden_size // 2)),
            nn.LeakyReLU(0.2),
            sn(nn.Linear(hidden_size // 2, 1)),
        )

    def forward(self, seq):
        x = seq.transpose(1, 2)
        x = self.features(x)
        x = self.pool(x)
        return self.fc(x)

def compute_moment_loss(fake_next, actual_next):
    fake_values = fake_next.squeeze(1)
    actual_values = actual_next.squeeze(1)
    fake_mean = fake_values.mean(dim=0)
    actual_mean = actual_values.mean(dim=0)
    fake_std = fake_values.std(dim=0, unbiased=False)
    actual_std = actual_values.std(dim=0, unbiased=False)
    mean_loss = nn.functional.l1_loss(fake_mean, actual_mean)
    # Use squared error for std to penalize volatility collapse more aggressively
    std_loss = nn.functional.mse_loss(fake_std, actual_std)
    return mean_loss + std_loss

def compute_drift_loss(history, fake_next, actual_next):
    last_history = history[:, -1, :]
    fake_delta = fake_next.squeeze(1) - last_history
    actual_delta = actual_next.squeeze(1) - last_history
    return nn.functional.l1_loss(fake_delta, actual_delta)

def discriminator_hinge_loss(real_score, fake_score):
    real_loss = torch.relu(1.0 - real_score).mean()
    fake_loss = torch.relu(1.0 + fake_score).mean()
    return real_loss + fake_loss

def make_stationary(df, price_cols, rate_cols):
    stat_df = df.copy()
    for col in price_cols:
        stat_df[col] = np.log(df[col] / df[col].shift(1))
    for col in rate_cols:
        stat_df[col] = df[col].diff()
    return stat_df.dropna().reset_index(drop=True)

def reconstruct_future_rows(last_known_vals, gen_stat, feature_names, price_cols):
    current_vals = last_known_vals.copy()
    rows = []
    for row_stat in gen_stat:
        for idx, col in enumerate(feature_names):
            if col in price_cols:
                current_vals[idx] = current_vals[idx] * np.exp(row_stat[idx])
            else:
                current_vals[idx] = current_vals[idx] + row_stat[idx]
        rows.append(current_vals.copy())
    return np.array(rows)

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
