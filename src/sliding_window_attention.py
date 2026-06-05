import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    """
    Memory-efficient sliding window self-attention using tensor.unfold().

    Replaces O(T^2) global attention with O(T*W) sliding window attention.
    K/V tensors are extracted into sliding windows via unfold (zero-copy view).
    The causal constraint is automatically enforced by left-padding with zeros.

    Args:
        d_model:    Input/output dimension (e.g. 32)
        window_size: Number of past timesteps to attend to (e.g. 50 or 100)
        num_heads:  Number of attention heads
        dropout:    Attention dropout rate
    """

    def __init__(self, d_model=32, window_size=50, num_heads=4, dropout=0.1):
        super(SlidingWindowAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model).double()
        self.k_proj = nn.Linear(d_model, d_model).double()
        self.v_proj = nn.Linear(d_model, d_model).double()
        self.out_proj = nn.Linear(d_model, d_model).double()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model).double()

        # Pre-compute causal padding mask (constant, not learned)
        # We compute it lazily in forward because T varies per sequence
        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            proj.weight.data[:] /= 10
            if proj.bias is not None:
                proj.bias.data[:] /= 100

    def _causal_pad_mask(self, T, W, device):
        """
        Returns (1, 1, T, W) boolean mask: True = padding position → mask out.

        For query at position t, keys at w are valid only when:
            w >= W-1-t   (i.e. not padding zeros)

        Equivalently: mask w where  w < W-1-t
        """
        w_idx = torch.arange(W, device=device).unsqueeze(0)             # (1, W)
        t_idx = torch.arange(T, device=device).unsqueeze(-1)            # (T, 1)
        pad_mask = w_idx < (W - 1 - t_idx)                               # (T, W)
        return pad_mask.unsqueeze(0).unsqueeze(0)                       # (1, 1, T, W)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x:         (B, T, d_model)
            attn_mask: None | (B, T, W)  additive attention mask.
                       None → only causal padding mask (fixed window).
                       Given  → added to scores BEFORE softmax.

        Returns:
            output: (B, T, d_model)  with residual connection included
        """
        B, T, D = x.shape
        H, d, W = self.num_heads, self.head_dim, self.window_size

        residual = x
        x_norm = self.layer_norm(x)

        # Project to Q, K, V
        Q = self.q_proj(x_norm).view(B, T, H, d).transpose(1, 2)       # (B, H, T, d)
        K = self.k_proj(x_norm).view(B, T, H, d).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, T, H, d).transpose(1, 2)

        # Left-pad (W-1) zeros so position 0 has a full window
        # F.pad format: (dim-1_left, dim-1_right, dim-2_left, dim-2_right, ...)
        # For (B, H, T, d): pad dim=2 (T) on the left by W-1
        K_padded = F.pad(K, (0, 0, W - 1, 0, 0, 0, 0, 0))            # (B, H, T+W-1, d)
        V_padded = F.pad(V, (0, 0, W - 1, 0, 0, 0, 0, 0))

        # Extract sliding windows via unfold (ZERO COPY VIEW)
        # unfold on dim=2 adds window dimension at the END: (B, H, T, d_head, W)
        K_w = K_padded.unfold(2, W, 1)                                  # (B, H, T, d_head, W)
        V_w = V_padded.unfold(2, W, 1)                                  # (B, H, T, d_head, W)

        # Attention scores: Q[b,h,t,:] · K_w[b,h,t,:,w]  →  (B, H, T, W)
        scale = d ** (-0.5)
        scores = torch.einsum('bhtd,bhtdw->bhtw', Q, K_w) * scale       # (B, H, T, W)

        # Causal padding mask: positions where K_w is padding zero
        pad_mask = self._causal_pad_mask(T, W, x.device)                # (1, 1, T, W)
        scores = scores.masked_fill(pad_mask, float('-inf'))

        # External mask (dynamic window suppression, etc.)
        if attn_mask is not None:
            # attn_mask: (B, T, W) → broadcast over heads: (B, 1, T, W)
            scores = scores + attn_mask.unsqueeze(1)

        attn_weights = self.dropout(torch.softmax(scores, dim=-1))      # (B, H, T, W)
        attn_out = torch.einsum('bhtw,bhtdw->bhtd', attn_weights, V_w)  # (B, H, T, d_head)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(attn_out) + residual                       # residual included
