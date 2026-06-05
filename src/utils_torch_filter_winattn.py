import torch
import torch.nn as nn
from sliding_window_attention import SlidingWindowAttention


class WindowAttention(nn.Module):
    """
    Fixed-size causal window attention.

    Uses SlidingWindowAttention under the hood. Each query attends to at
    most `window_size` past keys (including itself), with causal padding
    automatically handled by the sliding window mechanism.

    Args:
        d_model:     Input/output dimension (default: 32)
        window_size: Number of past timesteps to attend to (default: 50)
        num_heads:   Number of attention heads (default: 4)
        dropout:     Attention dropout rate (default: 0.1)
    """

    def __init__(self, d_model=32, window_size=50, num_heads=4, dropout=0.1):
        super(WindowAttention, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.attn = SlidingWindowAttention(
            d_model=d_model, window_size=window_size,
            num_heads=num_heads, dropout=dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        return self.attn(x, attn_mask=None)


class MesNet(nn.Module):
    """
    Measurement Covariance Network with Fixed Causal Window Attention.

    Architecture:
        Conv1D(6->32) -> Conv1D(32->32, dil=3)
        -> WindowAttention (fixed window)
        -> FC(32->32) -> FC(32->2) -> Softplus
        -> Measurement Covariance
    """

    def __init__(self, window_size=50, num_heads=4):
        super(MesNet, self).__init__()

        self.beta_measurement = 3 * torch.ones(2).double()
        self.tanh = torch.nn.Tanh()

        self.cov_net = nn.Sequential(
            nn.Conv1d(6, 32, 5),
            nn.ReplicationPad1d(4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(32, 32, 5, dilation=3),
            nn.ReplicationPad1d(4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        ).double()

        self.window_attention = WindowAttention(
            d_model=32, window_size=window_size,
            num_heads=num_heads, dropout=0.1)

        self.fc1 = nn.Sequential(nn.Linear(32, 32), nn.ReLU()).double()
        self.cov_lin = nn.Sequential(nn.Linear(32, 2), nn.Tanh()).double()

        self.cov_lin[0].bias.data[:] /= 100
        self.cov_lin[0].weight.data[:] /= 100
        for layer in self.fc1:
            if isinstance(layer, nn.Linear):
                layer.weight.data[:] /= 10
                if layer.bias is not None:
                    layer.bias.data[:] /= 100

    def forward(self, u, iekf):
        """
        Args:
            u: (B, 6, T) IMU measurements
        Returns:
            measurements_covs: (T', 2)
        """
        y_cov = self.cov_net(u)                              # (B, 32, T')
        y_cov_t = y_cov.transpose(1, 2)                      # (B, T', 32)
        y_attn = self.window_attention(y_cov_t)              # (B, T', 32)
        y_fc = self.fc1(y_attn)                              # (B, T', 32)
        z_cov = self.cov_lin(y_fc)                           # (B, T', 2)
        z_cov_net = self.beta_measurement.to(z_cov.device).unsqueeze(0) * z_cov
        measurements_covs = (iekf.cov0_measurement.unsqueeze(0) * (10 ** z_cov_net)).squeeze(0)
        return measurements_covs
