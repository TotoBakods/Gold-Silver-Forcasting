import torch
import torch.nn as nn
import numpy as np
import random

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.key   = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.value = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.scale = np.sqrt(hidden_dim * 2)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, filters=128, kernel_size=5, n_layers=2, dropout=0.2, use_attention=True):
        super(CNN_BiLSTM, self).__init__()
        self.conv1     = nn.Conv1d(input_dim, filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1       = nn.BatchNorm1d(filters)
        self.relu      = nn.ReLU()
        self.pool      = nn.MaxPool1d(2)
        self.dropout   = nn.Dropout(dropout)
        
        lstm_dropout = dropout if n_layers > 1 else 0
        self.lstm      = nn.LSTM(filters, hidden_dim, n_layers, batch_first=True, bidirectional=True, dropout=lstm_dropout)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(hidden_dim)
            
        self.fc        = nn.Linear(hidden_dim * 2, 64)
        self.out       = nn.Linear(64, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if x.shape[-1] > 1:
            x = self.pool(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        
        if self.use_attention:
            x = self.attention(x)
            
        x = x[:, -1, :] 
        x = self.fc(x)
        x = self.relu(x)
        return self.out(x)

class GANSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key   = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, noise_dim=64):
        super().__init__()
        channels = input_size + noise_dim
        self.backbone = nn.Sequential(
            ConvBlock(channels, hidden_size, dilation=1),
            ConvBlock(hidden_size, hidden_size, dilation=2),
            ConvBlock(hidden_size, hidden_size, dilation=4),
        )
        self.attention = GANSelfAttention(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self, history, noise):
        x = torch.cat((history, noise), dim=2).transpose(1, 2)
        x = self.backbone(x).transpose(1, 2)
        x = self.attention(x)
        generated = self.head(x[:, -1, :])
        return generated.unsqueeze(1)
