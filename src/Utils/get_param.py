"""Command-line flags shared by the training and rollout entries.

The physics and numerics of a case (schemes, loss weights, boundary conditions) are not
flags: they are read from the case's ``fvconfig.toml`` and ``bc/*.toml`` into ``fvconfig``.
"""
import argparse
import json
import os
import sys


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def params():
    parser = argparse.ArgumentParser(description="Train or roll out a network on finite-volume meshes")

    # training
    parser.add_argument("--net", default="TransFVGN_v2", type=str, choices=["TransFVGN_v2", "GNN", "Transolver"],
                        help="network")
    parser.add_argument("--n_epochs", default=1000000, type=int, help="training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="cases per batch and GPU")
    parser.add_argument("--average_sequence_length", default=500, type=int,
                        help="time steps a pool case is advanced, on average, before it is reset")
    parser.add_argument("--dataset_size", default=500, type=int,
                        help="pool slots over all ranks (each rank holds dataset_size / world_size)")
    parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    parser.add_argument("--optimizer_type", default="soap", type=str, choices=["soap", "adam"])
    parser.add_argument("--max_inner_steps", default=5, type=int,
                        help="passes over the pool per epoch; the last writes the predicted states back")

    # checkpoints and resume
    parser.add_argument("--load_date_time", default=None, type=str, help="run (date-time) to load")
    parser.add_argument("--load_index", default=None, type=int, help="checkpoint index of that run")
    parser.add_argument("--resume_state", default=False, type=str2bool,
                        help="also restore the pool states saved with that checkpoint")
    parser.add_argument("--resume_mmap", default=None, type=str,
                        help="data loss, DDP: run directory whose pool states and case assignment to resume")
    parser.add_argument("--resume_optimizer", default=False, type=str2bool,
                        help="data loss, DDP: also restore the optimizer state of RESUME_CHECKPOINT")
    parser.add_argument("--max_snapshots", default=3, type=int, help="pool snapshots kept on disk")

    # network
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--message_passing_num", default=1, type=int)
    parser.add_argument("--attn_processor_num", default=4, type=int)
    parser.add_argument("--num_heads", default=8, type=int, help="physics-attention heads")
    parser.add_argument("--slice_num", default=32, type=int, help="physics-attention slice tokens")
    parser.add_argument("--fourier_num_freqs", default=4, type=int,
                        help="Fourier bands per coordinate of the node positional encoding (0: none)")

    # data
    parser.add_argument("--dataset_dir", default="h5", type=str, help="case directory (mesh h5 and tomls)")
    parser.add_argument("--loss_mode", default="fvm", type=str, choices=["fvm", "data_driven"],
                        help="FVM residual or supervised loss against the labels")
    parser.add_argument("--split", default="all", type=str, choices=["train", "val", "test", "all"],
                        help="meta.json split of a case set; all = one case per h5")
    parser.add_argument("--labels_dir", default=None, type=str,
                        help="label store: labels_manifest.json, one .mmap per case, statistics.json")

    # set while the case is loaded
    parser.set_defaults(fvconfig={}, num_channels=None, phi_num_channels=None)
    return parser.parse_args()


def get_hyperparam(params):
    """Run name, the directory under ``logger/``."""
    return f"net {params.net}; hs {params.hidden_size};"


# Flags that change a module's shape (fourier_num_freqs through the node input width).
ARCH_KEYS = ("net", "hidden_size", "message_passing_num", "attn_processor_num", "num_heads", "slice_num",
             "fourier_num_freqs")


def arch_from_checkpoint(ckpt_path):
    """Architecture flags in ``commandline_args.json`` beside ``ckpt_path``, or None."""
    sidecar = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "commandline_args.json")
    if not os.path.isfile(sidecar):
        return None
    with open(sidecar, "rt") as f:
        saved = json.load(f)
    return {k: saved[k] for k in ARCH_KEYS if saved.get(k) is not None}


def apply_checkpoint_arch(params, ckpt_path, argv=None):
    """Set the architecture flags of ``params`` to the checkpoint's, except those in ``argv``."""
    argv = sys.argv[1:] if argv is None else argv
    explicit = {a.lstrip("-").split("=")[0] for a in argv if a.startswith("--")}
    arch = arch_from_checkpoint(ckpt_path)
    if arch is None:
        print(f"[arch] no commandline_args.json beside {ckpt_path}; using the command line")
        return params
    for k, v in arch.items():
        cur = getattr(params, k, None)
        if cur == v:
            continue
        if k in explicit:
            print(f"[arch] {k}: command line {cur!r} overrides checkpoint {v!r}")
        else:
            print(f"[arch] {k}: {cur!r} -> {v!r} (from checkpoint)")
            setattr(params, k, v)
    return params
