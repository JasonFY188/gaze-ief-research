"""
Quick sanity test for L2CSWrapper (Step 1).

Run with:  python tests/test_wrapper.py
(No GPU required — uses random weights on CPU.)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torchvision

from l2cs.model import L2CS
from l2cs.wrapper import L2CSWrapper, GazeFeatures


def make_model(num_bins=28):
    """ResNet50-based L2CS with random weights."""
    return L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins)


def test_forward_identical_to_base():
    """wrapper.forward() must be bit-for-bit identical to base_model()."""
    model = make_model()
    model.eval()
    wrapper = L2CSWrapper(model)
    wrapper.eval()

    x = torch.randn(2, 3, 448, 448)
    with torch.no_grad():
        base_pitch, base_yaw = model(x)
        wrap_pitch, wrap_yaw = wrapper(x)

    assert torch.allclose(base_pitch, wrap_pitch), "pitch logits differ in forward()"
    assert torch.allclose(base_yaw, wrap_yaw),     "yaw logits differ in forward()"
    print("PASS  wrapper.forward() is identical to base model")


def test_forward_with_features_logits_match():
    """forward_with_features() must return logits identical to base_model()."""
    model = make_model()
    model.eval()
    wrapper = L2CSWrapper(model)
    wrapper.eval()

    x = torch.randn(2, 3, 448, 448)
    with torch.no_grad():
        base_pitch, base_yaw = model(x)
        feats = wrapper.forward_with_features(x)

    assert isinstance(feats, GazeFeatures), "return type is not GazeFeatures"
    assert torch.allclose(base_pitch, feats.pitch_logits), \
        "pitch logits differ in forward_with_features()"
    assert torch.allclose(base_yaw, feats.yaw_logits), \
        "yaw logits differ in forward_with_features()"
    print("PASS  forward_with_features() logits match base model")


def test_feature_map_shape():
    """Feature map must be (B, 2048, 14, 14) for ResNet50 + 448×448 input."""
    model = make_model()
    model.eval()
    wrapper = L2CSWrapper(model)
    wrapper.eval()

    B = 2
    x = torch.randn(B, 3, 448, 448)
    with torch.no_grad():
        feats = wrapper.forward_with_features(x)

    expected = (B, 2048, 14, 14)
    got = tuple(feats.feature_map.shape)
    assert got == expected, f"feature_map shape: expected {expected}, got {got}"
    print(f"PASS  feature_map shape = {got}")


def test_num_bins_attribute():
    """Wrapper must expose num_bins matching the FC layer."""
    for bins in [28, 90]:
        model = make_model(num_bins=bins)
        wrapper = L2CSWrapper(model)
        assert wrapper.num_bins == bins, \
            f"num_bins: expected {bins}, got {wrapper.num_bins}"
    print("PASS  num_bins attribute correct for both 28 and 90")


def test_dataparallel_unwrap():
    """Wrapper must unwrap DataParallel so .base is the raw L2CS."""
    model = make_model()
    dp = nn.DataParallel(model)
    wrapper = L2CSWrapper(dp)
    assert wrapper.base is model, "DataParallel not unwrapped — .base should be L2CS"
    print("PASS  DataParallel unwrapped correctly")


if __name__ == "__main__":
    test_forward_identical_to_base()
    test_forward_with_features_logits_match()
    test_feature_map_shape()
    test_num_bins_attribute()
    test_dataparallel_unwrap()
    print("\nAll tests passed.")
