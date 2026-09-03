"""Count SHARPE MACs on CPU using the current selective-scan behavior."""

import argparse
import sys
import types
import warnings
from pathlib import Path

import torch
from torchprofile import profile_macs

def cpu_cross_scan(
    x,
    in_channel_first=True,
    out_channel_first=True,
    one_by_one=False,
    scans=0,
    force_torch=False,
):
    """Pure-PyTorch version of the channel-last cross2d scan."""
    if one_by_one or in_channel_first or out_channel_first or scans != 0:
        raise ValueError("This profiler expects the channel-last cross2d scan.")

    forward_joint_first = x.flatten(1, 2)
    forward_frame_first = x.transpose(1, 2).flatten(1, 2)
    return torch.stack(
        (
            forward_joint_first,
            forward_frame_first,
            forward_joint_first.flip(dims=(1,)),
            forward_frame_first.flip(dims=(1,)),
        ),
        dim=2,
    )


def cpu_cross_merge(
    y,
    in_channel_first=True,
    out_channel_first=True,
    one_by_one=False,
    scans=0,
    force_torch=False,
):
    """Pure-PyTorch version of the channel-last cross2d merge."""
    if one_by_one or in_channel_first or out_channel_first or scans != 0:
        raise ValueError("This profiler expects the channel-last cross2d merge.")

    batch, frames, joints, directions, channels = y.shape
    y = y.reshape(batch, frames * joints, directions, channels)
    y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=(1,))
    joint_first = y[:, :, 0]
    frame_first = (
        y[:, :, 1]
        .reshape(batch, joints, frames, channels)
        .transpose(1, 2)
        .contiguous()
        .reshape(batch, frames * joints, channels)
    )
    return joint_first + frame_first


# csm_triton.py contains an unguarded @triton.jit decorator and therefore
# cannot be imported on systems without Triton. Install a small CPU module
# with the same two functions before importing the SHARPE model. This changes
# neither the scan order nor the selective-state-space calculation.
cpu_csm_module = types.ModuleType("model.modules.csm_triton")
cpu_csm_module.cross_scan_fn = cpu_cross_scan
cpu_csm_module.cross_merge_fn = cpu_cross_merge
sys.modules["model.modules.csm_triton"] = cpu_csm_module

from utils.learning import load_model
from utils.tools import get_config


def parse_args():
    repository_root = Path(__file__).resolve().parent
    default_config = repository_root / "configs/h36m/SHARPE-tiny.yaml"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to a SHARPE YAML configuration file.",
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()
    config = get_config(str(cli_args.config.resolve()))

    warnings.filterwarnings(
        "ignore",
        message=r'No handlers found: ".*"\. Skipped\.',
        category=UserWarning,
        module=r"torchprofile\.profile",
    )

    model = load_model(config).cpu().eval()
    sample = torch.randn(
        1,
        config.n_frames,
        config.num_joints,
        config.dim_in,
        dtype=torch.float32,
    )

    macs = profile_macs(model, sample)

    print(f"Configuration: {cli_args.config}")
    print(f"Input shape:   {tuple(sample.shape)}")
    print(f"MACs:          {macs:,}")
    print(f"GMACs:         {macs / 1_000_000_000:.6f}")


if __name__ == "__main__":
    main()
