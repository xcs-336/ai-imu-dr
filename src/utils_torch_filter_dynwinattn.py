import torch
import torch.nn as nn
import torch.nn.functional as F
from sliding_window_attention import SlidingWindowAttention


class WindowPredictor(nn.Module):
    """
    Predicts optimal window size per timestep based on local motion features.

    Args:
        d_model:        Input feature dimension (default: 32)
        min_window:     Minimum window size (default: 10)
        max_window:     Maximum window size (default: 100)
        num_candidates: Number of discrete window size candidates (default: 10)
    """

    def __init__(self, d_model=32, min_window=10, max_window=100, num_candidates=10):
        super(WindowPredictor, self).__init__()
        self.min_window = min_window
        self.max_window = max_window
        self.num_candidates = num_candidates
        self.register_buffer(
            'candidates',
            torch.linspace(min_window, max_window, num_candidates).double())

        self.predictor = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_candidates),
        ).double()
        self._init_weights()

    def _init_weights(self):
        for layer in self.predictor:
            if isinstance(layer, nn.Linear):
                layer.weight.data[:] /= 10
                if layer.bias is not None:
                    layer.bias.data[:] /= 100

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            expected_windows: (B, T) float expected window sizes
            window_probs:     (B, T, num_candidates)
        """
        logits = self.predictor(x)
        window_probs = torch.softmax(logits, dim=-1)
        expected_windows = torch.matmul(window_probs, self.candidates)
        return expected_windows, window_probs


class DynamicWindowAttention(nn.Module):
    """
    Dynamic window attention: each timestep predicts its own window size.

    Uses SlidingWindowAttention with window_size = max_window. The
    WindowPredictor produces a mask that softly (train) or hard (eval)
    suppresses keys beyond the predicted window within the sliding window.

    Training: sigmoid soft suppression → differentiable, gradient flows
    Eval:     hard truncation → exact window behavior

    Args:
        d_model:           Input/output dimension (default: 32)
        min_window:        Minimum window (default: 10)
        max_window:        Maximum window = KV window size (default: 100)
        num_heads:         Number of attention heads (default: 4)
        num_candidates:    Number of window candidates (default: 10)
        dropout:           Attention dropout (default: 0.1)
        temperature:       Sigmoid temperature for soft mask (default: 5.0)
        suppression_scale: Suppression strength for soft mask (default: 10.0)
    """

    def __init__(self, d_model=32, min_window=10, max_window=100,
                 num_heads=4, num_candidates=10, dropout=0.1,
                 temperature=5.0, suppression_scale=10.0):
        super(DynamicWindowAttention, self).__init__()
        self.d_model = d_model
        self.min_window = min_window
        self.max_window = max_window
        self.num_heads = num_heads
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.suppression_scale = suppression_scale

        # Window predictor
        self.window_predictor = WindowPredictor(
            d_model=d_model, min_window=min_window,
            max_window=max_window, num_candidates=num_candidates)

        # Sliding window attention (KV window = max_window)
        self.attn = SlidingWindowAttention(
            d_model=d_model, window_size=max_window,
            num_heads=num_heads, dropout=dropout)

    def _make_dynamic_mask(self, T, expected_windows, device):
        """
        Create dynamic mask within the sliding window.

        Window coordinate system: w = 0 → furthest past key (distance = W-1)
                                   w = W-1 → self (distance = 0)
        distance = W - 1 - w

        Returns:
            mask: (B, T, W) additive mask, 0 = visible, negative = suppressed
        """
        W = self.max_window
        B = expected_windows.shape[0]

        # Distance of each window position from query: (W,)
        distance = W - 1 - torch.arange(W, device=device, dtype=torch.double)

        if self.training:
            # Sigmoid soft mask: differentiable!
            # logit > 0 when distance > window → suppress
            logit = self.temperature * (distance.unsqueeze(0).unsqueeze(0)
                                        - expected_windows.unsqueeze(-1))     # (B, T, W)
            mask = -self.suppression_scale * torch.sigmoid(logit)             # (B, T, W)
        else:
            # Hard binary mask: -inf where distance >= window_size
            mask = torch.where(
                distance.unsqueeze(0).unsqueeze(0) >= expected_windows.unsqueeze(-1),
                float('-inf'), 0.0)                                           # (B, T, W)
        return mask

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            output:           (B, T, d_model)
            expected_windows: (B, T)
        """
        B, T, _ = x.shape

        # Predict window sizes
        x_norm = self.attn.layer_norm(x)
        expected_windows, _ = self.window_predictor(x_norm)                    # (B, T)

        # Create dynamic mask and pass to sliding window attention
        attn_mask = self._make_dynamic_mask(T, expected_windows, x.device)     # (B, T, W)
        output = self.attn(x, attn_mask=attn_mask)

        return output, expected_windows


class MesNet(nn.Module):
    """
    Measurement Covariance Network with Dynamic Window Attention.

    Architecture:
        Conv1D(6->32) -> Conv1D(32->32, dil=3)
        -> DynamicWindowAttention (adaptive window per timestep)
        -> FC(32->32) -> FC(32->2) -> Softplus
        -> Measurement Covariance

    Returns (measurements_covs, expected_windows) — tuple for training.
    """

    def __init__(self, min_window=10, max_window=100, num_heads=4, num_candidates=10,
                 temperature=5.0, suppression_scale=10.0):
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

        self.dynamic_attention = DynamicWindowAttention(
            d_model=32, min_window=min_window, max_window=max_window,
            num_heads=num_heads, num_candidates=num_candidates,
            dropout=0.1, temperature=temperature,
            suppression_scale=suppression_scale)

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
            (measurements_covs, expected_windows): (T', 2), (B, T')
        """
        y_cov = self.cov_net(u)                                                # (B, 32, T')
        y_cov_t = y_cov.transpose(1, 2)                                        # (B, T', 32)
        y_attn, expected_windows = self.dynamic_attention(y_cov_t)             # (B, T', 32), (B, T')
        y_fc = self.fc1(y_attn)                                                # (B, T', 32)
        z_cov = self.cov_lin(y_fc)                                             # (B, T', 2)
        z_cov_net = self.beta_measurement.to(z_cov.device).unsqueeze(0) * z_cov
        measurements_covs = (iekf.cov0_measurement.unsqueeze(0) * (10 ** z_cov_net)).squeeze(0)
        return measurements_covs, expected_windows
