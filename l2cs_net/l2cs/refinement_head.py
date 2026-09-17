"""
IEF (Iterative Error Feedback) refinement head for L2CS-Net gaze estimation.

Conceptual reference: Carreira et al., "Human Pose Estimation with Iterative
Error Feedback", CVPR 2016 (arXiv:1507.06550).

Two head variants (select via RefinementConfig.head_type):
  "mlp"  — pools the feature map globally then runs an MLP (Step 3, baseline)
  "attn" — cross-attends over the spatial feature map positions (Step 6 upgrade)

Both share the same forward signature:
    forward(feature_map, pitch_logits, yaw_logits) → (delta_pitch, delta_yaw)
so IEFGazeModel and the training loop are unchanged between variants.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wrapper import L2CSWrapper, GazeFeatures


@dataclass
class RefinementConfig:
    num_bins:         int   = 90     # must match the frozen backbone (Gaze360 = 90)
    num_steps:        int   = 3      # refinement iterations after t=0
    max_delta_deg:    float = 10.0   # max angle-mean shift per step (degrees)
    hidden_dim:       int   = 256    # MLP hidden width
    freeze_backbone:  bool  = True   # keep L2CS weights fixed during training
    head_type:        str   = "mlp"  # "mlp" or "attn"
    d_attn:           int   = 256    # attention projection dim (attn head only)
    # Bin calibration for the backbone (Gaze360 defaults)
    bin_width_deg:    float = 4.0    # degrees per bin
    angle_offset_deg: float = 180.0  # angle = idx * bin_width - angle_offset


def _dist_summary(logits: torch.Tensor, bin_width: float, offset: float):
    """Return (mean_deg, var_deg) from raw logits. Shape: (B,) each."""
    probs  = F.softmax(logits, dim=1)
    idx    = torch.arange(logits.size(1), dtype=logits.dtype, device=logits.device)
    angles = idx * bin_width - offset
    mean   = (probs * angles).sum(dim=1)
    var    = (probs * (angles - mean.unsqueeze(1)) ** 2).sum(dim=1)
    return mean, var


def _estimate_vector(pitch_logits, yaw_logits, cfg: RefinementConfig):
    """Compact 4-D estimate summary, normalised to ~unit scale."""
    p_mean, p_var = _dist_summary(pitch_logits, cfg.bin_width_deg, cfg.angle_offset_deg)
    y_mean, y_var = _dist_summary(yaw_logits,   cfg.bin_width_deg, cfg.angle_offset_deg)
    off2 = cfg.angle_offset_deg ** 2
    return torch.stack([
        p_mean / cfg.angle_offset_deg,
        y_mean / cfg.angle_offset_deg,
        p_var  / off2,
        y_var  / off2,
    ], dim=1)   # (B, 4)


class RefinementHeadMLP(nn.Module):
    """MLP head: globally pools the feature map, then runs two FC layers.

    Identical to the original Step-3 head — kept as an ablation baseline.
    Input:  feature_map (B, 2048, H, W), pitch/yaw logits (B, num_bins)
    Output: delta_pitch, delta_yaw  each (B, num_bins), bounded by tanh
    """

    _FEAT_DIM = 2048
    _EST_DIM  = 4

    def __init__(self, cfg: RefinementConfig):
        super().__init__()
        self.cfg = cfg
        in_dim  = self._FEAT_DIM + self._EST_DIM
        out_dim = cfg.num_bins * 2

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, out_dim),
        )
        self._tanh_scale = cfg.max_delta_deg / cfg.bin_width_deg

    def forward(self, feature_map, pitch_logits, yaw_logits):
        # Pool spatial feature map → (B, 2048)
        pooled   = F.adaptive_avg_pool2d(feature_map, 1).view(feature_map.size(0), -1)
        estimate = _estimate_vector(pitch_logits, yaw_logits, self.cfg)

        out = self.mlp(torch.cat([pooled, estimate], dim=1))
        out = torch.tanh(out) * self._tanh_scale
        return out[:, :self.cfg.num_bins], out[:, self.cfg.num_bins:]


# Alias so existing checkpoints (saved with "head" of type RefinementHead) still load.
RefinementHead = RefinementHeadMLP


class RefinementHeadAttn(nn.Module):
    """Spatial cross-attention head.

    The current gaze estimate forms a query; the backbone spatial feature map
    (14×14 positions, 2048-d each) provides keys and values.  Soft attention
    selects which spatial regions to read given the current estimate, so the
    correction can focus on e.g. the eye corners when the estimate is near-correct
    vs the full face when it is far off.

    Architecture:
      query  = Linear(4, d_attn)          from compact estimate (B, d_attn)
      keys   = Conv2d(2048, d_attn, 1)    per spatial position  (B, H*W, d_attn)
      values = same projection as keys
      attn   = softmax(Q K^T / sqrt(d_attn))                    (B, 1, H*W)
      ctx    = attn @ values                                     (B, d_attn)
      out    = MLP([ctx, estimate]) → tanh-bounded delta logits
    """

    _EST_DIM = 4

    def __init__(self, cfg: RefinementConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_attn

        # Project 2048-d spatial features to d_attn for keys and values (shared)
        self.feat_proj  = nn.Conv2d(2048, d, kernel_size=1, bias=False)
        # Project 4-d estimate to d_attn for the query
        self.query_proj = nn.Linear(self._EST_DIM, d, bias=False)

        # MLP on [attended_context, estimate] → delta logits
        in_dim  = d + self._EST_DIM
        out_dim = cfg.num_bins * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, out_dim),
        )
        self._tanh_scale = cfg.max_delta_deg / cfg.bin_width_deg
        self._scale      = math.sqrt(d)

    def forward(self, feature_map, pitch_logits, yaw_logits):
        B = feature_map.size(0)
        cfg = self.cfg

        estimate = _estimate_vector(pitch_logits, yaw_logits, cfg)   # (B, 4)

        # Keys/values: project all H*W spatial positions
        kv = self.feat_proj(feature_map)              # (B, d_attn, H, W)
        kv = kv.view(B, cfg.d_attn, -1).transpose(1, 2)  # (B, H*W, d_attn)

        # Query from current estimate
        q = self.query_proj(estimate).unsqueeze(1)    # (B, 1, d_attn)

        # Scaled dot-product attention
        attn = torch.bmm(q, kv.transpose(1, 2)) / self._scale  # (B, 1, H*W)
        attn = torch.softmax(attn, dim=-1)

        # Attended context vector
        ctx = torch.bmm(attn, kv).squeeze(1)          # (B, d_attn)

        out = self.mlp(torch.cat([ctx, estimate], dim=1))  # (B, num_bins*2)
        out = torch.tanh(out) * self._tanh_scale
        return out[:, :cfg.num_bins], out[:, cfg.num_bins:]


class IEFGazeModel(nn.Module):
    """Full iterative refinement model.

    Wraps a frozen L2CSWrapper (backbone) and either a MLP or attention head.
    Runs the backbone once for t=0, then loops the head for cfg.num_steps steps.

    Returns a list of (pitch_logits, yaw_logits) of length num_steps+1:
      steps[0]  = raw backbone output (t=0, no refinement)
      steps[-1] = final refined prediction
    """

    def __init__(self, wrapper: L2CSWrapper, cfg: RefinementConfig):
        super().__init__()
        self.wrapper = wrapper
        self.cfg     = cfg

        head_type = getattr(cfg, "head_type", "mlp")
        if head_type == "attn":
            self.head = RefinementHeadAttn(cfg)
        else:
            self.head = RefinementHeadMLP(cfg)

        if cfg.freeze_backbone:
            for p in self.wrapper.parameters():
                p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # Keep backbone BN in eval mode so it always uses running statistics.
        if self.cfg.freeze_backbone:
            self.wrapper.eval()
        return self

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            steps: list[(pitch_logits, yaw_logits)], length = num_steps + 1
        """
        with torch.set_grad_enabled(not self.cfg.freeze_backbone):
            feats: GazeFeatures = self.wrapper.forward_with_features(x)

        pitch_logits  = feats.pitch_logits    # (B, num_bins)
        yaw_logits    = feats.yaw_logits      # (B, num_bins)
        feature_map   = feats.feature_map     # (B, 2048, H, W)

        steps = [(pitch_logits, yaw_logits)]

        for _ in range(self.cfg.num_steps):
            delta_pitch, delta_yaw = self.head(feature_map, pitch_logits, yaw_logits)
            pitch_logits = pitch_logits + delta_pitch
            yaw_logits   = yaw_logits   + delta_yaw
            steps.append((pitch_logits, yaw_logits))

        return steps
