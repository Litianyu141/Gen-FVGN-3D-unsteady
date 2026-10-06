#!/usr/bin/env python3
"""Roll out a data-loss checkpoint on one case and compare it with the labels.

    python src/entries/rollout_with_data_driven.py \\
        --dataset_dir experiments/Cylinder/NeuralOperator/hybrid-dd/h5 \\
        --labels_dir <label store> \\
        --checkpoint <run>/states/<k>.state --eval-split test --dataset_size 30 \\
        --case-idx 0 --n-steps 500 --save-only-last

Outputs go to ``<run>/rollout_results/<case>/``: VTM snapshots with the labels, per-step
error moments (``error_moments.csv``), timing and slice renderings.  Mesh, label and model
flags not listed below are those of training (``src/Utils/get_param.py``).
"""
import argparse
import os
import shutil
import sys

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["PYVISTA_OFF_SCREEN"] = "true"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Pipeline.NeuralOperator.rollout import setup, run_single  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats-path", default=None, help="z-score statistics (default: <labels_dir>/statistics.json)")
    p.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    p.add_argument("--case-idx", type=int, default=0, help="case index within the split")
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--save-only-last", action="store_true", help="write the VTM of the last step only")
    p.add_argument("--no-vtu", action="store_true")
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--snapshot-steps", default=None, help="steps to render, e.g. 50,100,500")
    p.add_argument("--slice-axis", default=None, choices=["x", "y", "z"],
                   help="normal of the rendered slice (default: the thinnest direction)")
    p.add_argument("--slice-pos", type=float, default=None, help="slice position (default: midpoint)")
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--gpu", type=int, default=0)
    args, rest = p.parse_known_args()
    if args.snapshot_steps:
        args.snapshot_steps = sorted(int(s) for s in args.snapshot_steps.split(","))
    sys.argv = [sys.argv[0]] + rest
    return args


def main():
    ctx = setup(parse_args())
    run_single(ctx)
    ctx["pool"].close()
    shutil.rmtree(ctx["state_dir"], ignore_errors=True)


if __name__ == "__main__":
    main()
