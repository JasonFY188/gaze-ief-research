"""Smoke tests for Step 6: both head types + old checkpoint compatibility."""
import sys, torch, torchvision
sys.path.insert(0, __file__.replace("\\tests\\test_step6.py", "").replace("/tests/test_step6.py", ""))

from l2cs.model import L2CS
from l2cs.wrapper import L2CSWrapper
from l2cs.refinement_head import RefinementConfig, IEFGazeModel

CKPT = r"C:\Users\keiso\Jason\Dealing with uncertainty from l2cs net\L2CS-Net\output\refinement\best_fold0.pt"

base    = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=90)
wrapper = L2CSWrapper(base)
x       = torch.randn(2, 3, 448, 448)


def test_mlp_head():
    cfg   = RefinementConfig(head_type="mlp", num_steps=3)
    model = IEFGazeModel(wrapper, cfg).eval()
    with torch.no_grad():
        steps = model(x)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert len(steps) == 4
    assert steps[-1][0].shape == (2, 90)
    print(f"PASS  MLP  head: {n:,} trainable params")


def test_attn_head():
    cfg   = RefinementConfig(head_type="attn", num_steps=3, d_attn=256)
    model = IEFGazeModel(wrapper, cfg).eval()
    with torch.no_grad():
        steps = model(x)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert len(steps) == 4
    assert steps[-1][0].shape == (2, 90)
    print(f"PASS  Attn head: {n:,} trainable params")


def test_attn_logits_differ_from_mlp():
    cfg_mlp  = RefinementConfig(head_type="mlp",  num_steps=3)
    cfg_attn = RefinementConfig(head_type="attn", num_steps=3, d_attn=256)
    mlp_m  = IEFGazeModel(wrapper, cfg_mlp).eval()
    attn_m = IEFGazeModel(wrapper, cfg_attn).eval()
    with torch.no_grad():
        mlp_out  = mlp_m(x)[-1][0]
        attn_out = attn_m(x)[-1][0]
    assert not torch.allclose(mlp_out, attn_out), "Attn and MLP produced identical outputs"
    print("PASS  Attn and MLP outputs differ (as expected with different weights)")


def test_old_checkpoint_loads():
    ckpt        = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg         = ckpt["cfg"]
    model       = IEFGazeModel(wrapper, cfg).eval()
    model.head.load_state_dict(ckpt["head_state_dict"])
    with torch.no_grad():
        steps = model(x)
    assert len(steps) == cfg.num_steps + 1
    val_err = ckpt["val_angular_error"]
    print(f"PASS  Old checkpoint loaded OK (saved val_err={val_err:.3f}deg)")


if __name__ == "__main__":
    test_mlp_head()
    test_attn_head()
    test_attn_logits_differ_from_mlp()
    test_old_checkpoint_loads()
    print("\nAll Step 6 tests passed.")
