from collections import namedtuple

import torch.nn as nn

from .model import L2CS


# Named return type for forward_with_features.
# pitch_logits / yaw_logits: (B, num_bins) raw (pre-softmax) logits
# feature_map:               (B, C, H, W)  spatial output of layer4, before avgpool
#                            e.g. (B, 2048, 14, 14) for ResNet50 + 448×448 input
GazeFeatures = namedtuple("GazeFeatures", ["pitch_logits", "yaw_logits", "feature_map"])


class L2CSWrapper(nn.Module):
    """Wraps a pretrained L2CS model to also expose the spatial feature map
    from layer4 (before global average pooling) alongside the per-axis bin logits.

    The original model's forward() behaviour is completely unchanged.

    Naming note: model.py calls the two FC heads fc_yaw_gaze / fc_pitch_gaze, but
    every caller in the codebase unpacks the return as (pitch, yaw). This wrapper
    follows the caller convention: GazeFeatures.pitch_logits = fc_yaw_gaze output,
    GazeFeatures.yaw_logits = fc_pitch_gaze output.

    Args:
        base_model: a pretrained L2CS instance (or DataParallel-wrapped L2CS).
    """

    def __init__(self, base_model):
        super().__init__()
        # Unwrap DataParallel so we can always access named submodules directly.
        if isinstance(base_model, nn.DataParallel):
            base_model = base_model.module
        self.base = base_model
        # Expose bin count so downstream code doesn't need to inspect the model.
        self.num_bins: int = self.base.fc_yaw_gaze.out_features

    def forward(self, x):
        """Identical to the base L2CS forward — returns (pitch_logits, yaw_logits)."""
        return self.base(x)

    def forward_with_features(self, x) -> GazeFeatures:
        """Full forward pass that also captures the spatial feature map.

        Manually re-runs every layer of the base model so the feature map can
        be intercepted after layer4 and before avgpool. The logits are
        numerically identical to forward().

        Returns:
            GazeFeatures namedtuple:
              .pitch_logits  (B, num_bins)  — raw logits for pitch axis
              .yaw_logits    (B, num_bins)  — raw logits for yaw axis
              .feature_map   (B, C, H, W)  — spatial tensor before pooling
        """
        b = self.base

        h = b.conv1(x)
        h = b.bn1(h)
        h = b.relu(h)
        h = b.maxpool(h)
        h = b.layer1(h)
        h = b.layer2(h)
        h = b.layer3(h)
        h = b.layer4(h)

        feature_map = h  # (B, 2048, H, W) — spatial info discarded by avgpool

        h = b.avgpool(h)
        h = h.view(h.size(0), -1)
        pitch_logits = b.fc_yaw_gaze(h)
        yaw_logits = b.fc_pitch_gaze(h)

        return GazeFeatures(pitch_logits, yaw_logits, feature_map)
