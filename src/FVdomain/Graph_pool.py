"""Case pool management: in-memory single-case pool, mmap-backed training pool, and GPU-resident pool."""

from collections import OrderedDict, deque
import json
import logging
import os
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch_geometric.utils import scatter

from FVdomain.MmapManager import (
    AsyncSnapshotWriter,
    CaseSlot,
    MeshCacheManager,
    SharedMeshStore,
    STATIC_TENSOR_KEYS,
    _atomic_json_write,
)
from FVdomain.SetBC.onthefly.BCbase import VAR_CONFIG
from Post_process.vtk_export import export_vtm
from Utils.utilities import FaceType

logger = logging.getLogger(__name__)


def global_slot_index(local_slot_idx, rank, world_size):
    """Global slot g lives on rank g % world_size at local index g // world_size."""
    return local_slot_idx * world_size + rank


def reset_owner_rank(reset_counter, dataset_size, world_size):
    """Rank that owns global reset number ``reset_counter``."""
    return (reset_counter % dataset_size) % world_size


def case_seed_for_global_slot(global_idx, base=777):
    """Deterministic per-slot parameter seed independent of world size."""
    return base + int(global_idx)


def build_rank_case_list(case_names, total_dataset_size, rank, world_size, seed=42, return_global=False):
    """Repeat cases to ``total_dataset_size``, shuffle deterministically, and stride-slice by rank."""
    if total_dataset_size % world_size != 0:
        raise ValueError(f"dataset_size ({total_dataset_size}) must be divisible by world_size ({world_size})")
    n_cases = len(case_names)
    global_list = list(case_names) * (total_dataset_size // n_cases) + list(case_names[: total_dataset_size % n_cases])
    random.Random(seed).shuffle(global_list)
    return global_list if return_global else global_list[rank::world_size]


def save_case_assignment(save_dir, global_case_list, dataset_size, world_size, seed):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset_size": dataset_size,
        "world_size": world_size,
        "seed": seed,
        "global_case_list": global_case_list,
        "rank_slices": [global_case_list[r::world_size] for r in range(world_size)],
    }
    with open(save_dir / "case_assignment.json", "w") as f:
        json.dump(data, f, indent=2)


def load_case_assignment(save_dir):
    with open(Path(save_dir) / "case_assignment.json") as f:
        return json.load(f)


def validate_resume_params(saved, dataset_size, world_size):
    for key, current in (("dataset_size", dataset_size), ("world_size", world_size)):
        if saved[key] != current:
            raise ValueError(f"{key} mismatch: saved={saved[key]}, current={current}; cannot resume the pool")


def _sampled_parameters_of(mesh_dict):
    pm = mesh_dict.get("parametric_manager")
    if pm is None:
        return None
    try:
        return {
            k: v["current_value"]
            for k, v in pm.get_status().items()
            if isinstance(v.get("current_value"), (int, float))
        }
    except Exception:
        return None


def _strip_static_tensors(mesh_dict):
    return {k: v for k, v in mesh_dict.items() if k not in STATIC_TENSOR_KEYS}


def _build_merged_mesh(slot):
    merged = dict(slot.meta_objects)
    shared = slot._shared_mesh
    if shared is not None and shared.manifest is not None:
        for key in STATIC_TENSOR_KEYS:
            if shared.has_tensor(key):
                merged[key] = shared.get_tensor(key).clone()
    return merged


class HeteroDataPool:
    """Lightweight in-memory pool used for single-case ODIL solves and tests."""

    def __init__(self, params=None, device=None, state_save_dir=None, export_patches=True, rank=0, ddp_mode=False):
        self.params = params
        self.device = device
        self.export_patches = export_patches
        self.state_save_dir = state_save_dir
        self.meta_pool = []
        self.phi_cpd_cell_pool = []
        self.time_step_pool = []
        self.rank = rank
        self.ddp_mode = ddp_mode

    def load_mesh_to_cpu(self, dataset_dir=None):
        from FVdomain.Load_mesh import H5CFDdataset

        dataset_dir = dataset_dir or self.params.dataset_dir
        if dataset_dir.endswith(".h5"):
            valid_h5 = [dataset_dir]
        else:
            valid_h5 = sorted(
                os.path.join(subdir, name)
                for subdir, _, files in os.walk(dataset_dir, followlinks=True)
                for name in files
                if name.endswith(".h5")
            )
        if not valid_h5:
            raise ValueError(f".h5 file not found in dataset directory: {dataset_dir}")

        mesh_dataset = H5CFDdataset(params=self.params, case_list=valid_h5)
        start_idx = 0
        num_channels = phi_num_channels = None
        while len(self.meta_pool) < self.params.dataset_size:
            for i_data in range(len(valid_h5)):
                meta_data, init_phi = mesh_dataset[i_data]
                meta_data["global_idx"] = torch.arange(start_idx, start_idx + init_phi.shape[0])
                if num_channels is None:
                    num_channels = meta_data["num_channels"]
                    phi_num_channels = meta_data["phi_num_channels"]
                elif meta_data["num_channels"] != num_channels:
                    raise ValueError(f"Channel mismatch: {meta_data['num_channels']} != {num_channels}")
                self.meta_pool.append(meta_data)
                self.phi_cpd_cell_pool.append(init_phi)
                start_idx += init_phi.shape[0]
                if len(self.meta_pool) >= self.params.dataset_size:
                    break

        self.phi_cpd_cell_pool = torch.cat(self.phi_cpd_cell_pool, dim=0)
        self.time_step_pool = torch.zeros(len(self.meta_pool), dtype=torch.float32)

        self.params.num_channels = num_channels
        self.params.phi_num_channels = phi_num_channels
        self.params.dataset_size = len(self.meta_pool)
        self.params.fvconfig = self.meta_pool[0]["fvconfig"]
        self.params.fvschemes = self.meta_pool[0]["fvconfig"]["fvSchemes"]
        fvsol = self.params.fvconfig.get("fvSolution", {})
        for key in ("loss_cont", "loss_mom", "loss_energy", "loss_phi_jump", "loss_pRef"):
            if key in fvsol:
                setattr(self.params, key, fvsol[key])
        return self.params


class GPUBatchLoader:
    """Zero-copy GPU batch loader backed by ``MmapDataPool.load_Graph_Batch``."""

    def __init__(self, pool: "MmapDataPool", batch_size: int = 1, shuffle: bool = True):
        self.base_dataset = pool
        self.pool = pool
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.dataset_size = len(pool.slots)
        self.indices = list(range(self.dataset_size))

    def __len__(self):
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.dataset_size = len(self.pool.slots)
        if len(self.indices) != self.dataset_size:
            self.indices = list(range(self.dataset_size))
        indices = self.indices.copy() if self.shuffle else self.indices
        if self.shuffle:
            random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            yield self.pool.load_Graph_Batch(indices[i : i + self.batch_size])


class MmapDataPool:
    """Mmap-backed training pool with an automatic GPU-resident fast path for homogeneous and ragged batches."""

    def __init__(
        self,
        params=None,
        device=None,
        state_save_dir=None,
        export_patches=True,
        rank=0,
        ddp_mode=False,
        gpu_pool=None,
    ):
        self.params = params
        self.device = device
        self.export_patches = export_patches
        self.state_save_dir = state_save_dir
        self.slots: list[CaseSlot] = []
        self.slot_queue: deque[int] = deque()
        self._plot_env = True
        self.reset_env_flag = False
        self.rst_time = 1
        self.rank = rank
        self.ddp_mode = ddp_mode
        self._world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

        self._rank_dir: Path | None = None
        self._mesh_cache: MeshCacheManager | None = None
        self._snapshot_writer = AsyncSnapshotWriter()
        self._snapshots_root: Path | None = None
        self.loss_mode: str = "fvm"
        self._has_labels: bool = False
        self.plot_count = 0

        self._gpu_pool_pref = gpu_pool
        self.gpu_pool_enabled: bool = False
        self.same_mesh: bool = False
        self._single_n_cpd: int | None = None
        self._gpu_dev: torch.device | None = None
        self._gpu_n_cpd: int = 0
        self._gpu_n_ch: int = 0
        self.slot_n_cpd: list[int] = []
        self.slot_cpd_ptr: torch.Tensor | None = None
        self.gpu_phi: torch.Tensor | None = None
        self.gpu_init_phi: torch.Tensor | None = None
        self.gpu_phi_flat: torch.Tensor | None = None
        self.gpu_init_phi_flat: torch.Tensor | None = None
        self._gpu_phi_views: list[torch.Tensor] = []
        self._gpu_init_phi_views: list[torch.Tensor] = []
        self.gpu_theta_pde: torch.Tensor | None = None
        self.gpu_time_step: torch.Tensor | None = None
        self._cpu_time_steps: list[int] = []
        self.gpu_pRef_valid_mask: torch.Tensor | None = None
        self.gpu_pRefCell_local: torch.Tensor | None = None
        self.gpu_pRefValue: torch.Tensor | None = None
        self.gpu_dt: torch.Tensor | None = None
        self._slot_fvpatches: list = []
        self._slot_bcpatches: list = []
        self._slot_case_name_tensors: list[torch.Tensor] = []
        self._slot_mesh_keys: list = []
        self._gpu_unique_meshes: dict = {}
        self._gpu_batch_templates: dict = {}
        self._hetero_batch_templates: OrderedDict = OrderedDict()
        self._bcpatch_batch_cache: OrderedDict = OrderedDict()
        self._mirror_index_cache: dict = {}
        self._last_batch_cpd: torch.Tensor | None = None

    @property
    def meta_pool(self):
        return [slot.meta_objects for slot in self.slots]

    @property
    def time_step_pool(self):
        if self.gpu_pool_enabled:
            return torch.tensor(self._cpu_time_steps, dtype=torch.float32)
        return torch.tensor([slot.time_step for slot in self.slots], dtype=torch.float32)

    def _find_meta_json(self, dataset_dir: str):
        base = Path(dataset_dir).resolve()
        for cand in (
            base / "meta.json",
            base.parent / "openfoam" / "meta.json",
            base.parent.parent / "openfoam" / "meta.json",
            base.parent.parent.parent / "openfoam" / "meta.json",
        ):
            if cand.is_file():
                return cand
        return None

    def _find_labels_manifest(self, dataset_dir: str):
        explicit = getattr(self.params, "labels_dir", None)
        if explicit:
            p = Path(explicit)
            if p.name != "labels_manifest.json":
                p = p / "labels_manifest.json"
            if not p.is_file():
                raise FileNotFoundError(f"--labels_dir {explicit}: no labels_manifest.json found")
            return p
        base = Path(dataset_dir).resolve()
        for cand in (
            base / "data_driven" / "labels" / "labels_manifest.json",
            base.parent / "data_driven" / "labels" / "labels_manifest.json",
            base / "labels" / "labels_manifest.json",
            base.parent / "labels" / "labels_manifest.json",
        ):
            if cand.is_file():
                return cand
        return None

    def _attach_labels_to_slots(self, manifest_path: Path) -> None:
        with open(manifest_path) as f:
            manifest = json.load(f)
        labels_dir = manifest_path.parent
        mmap_by_name = {
            case["case_name"]: {"file": labels_dir / case["mmap_file"], "shape": tuple(case["shape"])}
            for case in manifest["cases"]
        }
        dtype = manifest.get("dtype", "float32")
        t_offset = manifest.get("T_offset")
        unmatched = [
            s._meta.get("case_name", "")
            for s in self.slots
            if s._meta.get("case_name", "") not in mmap_by_name
        ]
        if unmatched:
            raise RuntimeError(f"No labels in {manifest_path} for cases {unmatched[:5]}")
        for slot in self.slots:
            info = mmap_by_name[slot._meta.get("case_name", "")]
            slot.attach_labels(labels_file=info["file"], shape=info["shape"], dtype=dtype)
            if t_offset is not None:
                slot._label_T_offset = float(t_offset)
        print(f"[MmapDataPool] Attached labels to {len(self.slots)}/{len(self.slots)} slots from {manifest_path}")

    def _sync_params(self):
        meta = self.slots[0].meta_objects
        self.params.num_channels = meta["num_channels"]
        self.params.phi_num_channels = meta["phi_num_channels"]
        self.params.dataset_size = len(self.slots)
        self.params.fvconfig = meta["fvconfig"]
        self.params.fvschemes = meta["fvconfig"]["fvSchemes"]
        fvsol = self.params.fvconfig.get("fvSolution", {})
        for key in ("loss_cont", "loss_mom", "loss_energy", "loss_phi_jump", "loss_pRef"):
            if key in fvsol:
                setattr(self.params, key, fvsol[key])

    def load_mesh_to_mmap(
        self,
        dataset_dir=None,
        resume=False,
        resume_from=None,
        split: str = "all",
        loss_mode: str = "fvm",
        total_dataset_size: int | None = None,
        world_size: int = 1,
        param_overrides=None,
    ):
        from FVdomain.Load_mesh import H5CFDdataset

        dataset_dir = dataset_dir or self.params.dataset_dir
        self.loss_mode = loss_mode

        mmap_root = Path(self.state_save_dir) / "mmap_state"
        mmap_root.mkdir(parents=True, exist_ok=True)
        live_root = Path(os.environ["MMAP_LIVE_ROOT"]) if os.environ.get("MMAP_LIVE_ROOT") else mmap_root
        live_root.mkdir(parents=True, exist_ok=True)
        self._rank_dir = live_root / f"rank_{self.rank}"
        self._rank_dir.mkdir(parents=True, exist_ok=True)
        self._mesh_cache = MeshCacheManager(live_root / "mesh_cache")
        self._snapshots_root = mmap_root / "snapshots"
        self._snapshots_root.mkdir(parents=True, exist_ok=True)

        if resume:
            if resume_from:
                old_root = Path(resume_from)
                old_rank_dir = old_root / f"rank_{self.rank}"
                if (old_rank_dir / "slot_0").exists():
                    old_snaps = old_root / "snapshots"
                    return self._resume_from_mmap(old_rank_dir, old_snaps if old_snaps.exists() else None)
            if (self._rank_dir / "slot_0").exists():
                return self._resume_from_mmap(self._rank_dir, self._snapshots_root)

        if dataset_dir.endswith(".h5"):
            valid_h5 = [dataset_dir]
        else:
            valid_h5 = sorted(
                os.path.join(subdir, name)
                for subdir, _, files in os.walk(dataset_dir, followlinks=True)
                for name in files
                if name.endswith(".h5")
            )
        if not valid_h5:
            raise ValueError(f".h5 file not found in dataset directory: {dataset_dir}")

        case_names_from_meta = None
        case_params_from_meta: dict = {}
        case_meshes_from_meta: dict = {}
        if split != "all":
            meta_json_path = self._find_meta_json(dataset_dir)
            if meta_json_path is None:
                raise FileNotFoundError(f"split={split!r} requested but no meta.json found near {dataset_dir}")
            with open(meta_json_path) as f:
                meta = json.load(f)
            case_names_from_meta = list(meta["split"][split])
            case_params_from_meta = meta.get("case_parameters", {}) or {}
            case_meshes_from_meta = meta.get("case_meshes", {}) or {}
            print(
                f"[MmapDataPool] Using meta.json split={split!r}: {len(case_names_from_meta)} cases"
                + (" + case_parameters" if case_params_from_meta else "")
            )

        mesh_dataset = H5CFDdataset(params=self.params, case_list=valid_h5)
        rank_case_list = None
        if case_names_from_meta is not None and total_dataset_size is not None:
            rank_case_list = build_rank_case_list(case_names_from_meta, total_dataset_size, self.rank, world_size)
            if self.rank == 0:
                global_list = build_rank_case_list(
                    case_names_from_meta, total_dataset_size, 0, world_size, return_global=True
                )
                save_case_assignment(self.state_save_dir, global_list, total_dataset_size, world_size, seed=42)

        h5_lookup: dict = {}
        for idx, hp in enumerate(valid_h5):
            p = Path(hp)
            h5_lookup[hp] = idx
            h5_lookup[p.name] = idx
            h5_lookup[p.stem] = idx
            h5_lookup[p.parent.name] = idx
            if p.parent.parent.name:
                h5_lookup[p.parent.parent.name] = idx

        num_channels = None
        while len(self.slots) < self.params.dataset_size:
            if case_names_from_meta is not None:
                iter_cases = rank_case_list if rank_case_list is not None else case_names_from_meta
                shared_by_idx: dict = {}
                for case_name in iter_cases:
                    if len(self.slots) >= self.params.dataset_size:
                        break
                    raw_ovr = case_params_from_meta.get(case_name)
                    per_case_ovr = dict(raw_ovr) if raw_ovr else None
                    mesh_hint = case_meshes_from_meta.get(case_name)
                    if per_case_ovr and "h5_file" in per_case_ovr:
                        mesh_hint = per_case_ovr.pop("h5_file")
                    if len(valid_h5) == 1:
                        target_i = 0
                    elif mesh_hint is not None and str(mesh_hint) in h5_lookup:
                        target_i = h5_lookup[str(mesh_hint)]
                    elif case_name in h5_lookup:
                        target_i = h5_lookup[case_name]
                    else:
                        target_i = len(self.slots) % len(valid_h5)

                    target_h5 = valid_h5[target_i]
                    target_hash = self._mesh_cache._hash_h5(target_h5)
                    if param_overrides:
                        per_case_ovr = {**(per_case_ovr or {}), **param_overrides}
                    if per_case_ovr:
                        mesh_dataset.set_pending_overrides(per_case_ovr)
                    mesh_dataset.set_pending_global_slot_index(
                        global_slot_index(len(self.slots), self.rank, self._world_size)
                    )
                    mesh_dict, init_phi = mesh_dataset[target_i]
                    if num_channels is None:
                        num_channels = mesh_dict["num_channels"]
                    elif mesh_dict["num_channels"] != num_channels:
                        raise ValueError(f"Channel mismatch: {mesh_dict['num_channels']} != {num_channels}")
                    shared_mesh = shared_by_idx.get(target_i)
                    if shared_mesh is None:
                        shared_mesh = self._mesh_cache.get_or_create(target_h5, mesh_dict)
                        shared_by_idx[target_i] = shared_mesh

                    slot_idx = len(self.slots)
                    slot = CaseSlot.create(
                        slot_dir=self._rank_dir / f"slot_{slot_idx}",
                        shared_mesh=shared_mesh,
                        num_cpd_cells=init_phi.shape[0],
                        phi_channels=init_phi.shape[1],
                        init_phi=init_phi,
                        meta_info={
                            "h5_hash": target_hash,
                            "h5_path": target_h5,
                            "case_name": case_name,
                            "case_parameters": per_case_ovr,
                        },
                    )
                    slot.meta_objects = _strip_static_tensors(mesh_dict)
                    slot.meta_objects["case_name"] = case_name
                    if per_case_ovr:
                        slot.meta_objects["case_parameters"] = per_case_ovr
                    sampled = _sampled_parameters_of(mesh_dict)
                    if sampled:
                        slot.set_meta_entry("sampled_parameters", sampled)
                    self.slots.append(slot)
                    self.slot_queue.append(slot_idx)
            else:
                for i_data, h5_path in enumerate(valid_h5):
                    if len(self.slots) >= self.params.dataset_size:
                        break
                    h5_hash = self._mesh_cache._hash_h5(h5_path)
                    if param_overrides:
                        mesh_dataset.set_pending_overrides(dict(param_overrides))
                    mesh_dataset.set_pending_global_slot_index(
                        global_slot_index(len(self.slots), self.rank, self._world_size)
                    )
                    mesh_dict, init_phi = mesh_dataset[i_data]
                    if num_channels is None:
                        num_channels = mesh_dict["num_channels"]
                    elif mesh_dict["num_channels"] != num_channels:
                        raise ValueError(f"Channel mismatch: {mesh_dict['num_channels']} != {num_channels}")
                    shared_mesh = self._mesh_cache.get_or_create(h5_path, mesh_dict)
                    slot_idx = len(self.slots)
                    slot = CaseSlot.create(
                        slot_dir=self._rank_dir / f"slot_{slot_idx}",
                        shared_mesh=shared_mesh,
                        num_cpd_cells=init_phi.shape[0],
                        phi_channels=init_phi.shape[1],
                        init_phi=init_phi,
                        meta_info={
                            "h5_hash": h5_hash,
                            "h5_path": h5_path,
                            "case_name": mesh_dict["case_name"],
                            "case_parameters": None,
                        },
                    )
                    slot.meta_objects = _strip_static_tensors(mesh_dict)
                    sampled = _sampled_parameters_of(mesh_dict)
                    if sampled:
                        slot.set_meta_entry("sampled_parameters", sampled)
                    self.slots.append(slot)
                    self.slot_queue.append(slot_idx)

        if loss_mode == "data_driven":
            manifest = self._find_labels_manifest(dataset_dir)
            if manifest is None:
                raise RuntimeError(
                    f"loss_mode='data_driven' requires labels_manifest.json under {dataset_dir}. "
                    "Run prepare_labels.py first."
                )
            self._attach_labels_to_slots(manifest)
            self._has_labels = True

        self._sync_params()
        self._maybe_init_gpu_pool()
        return self.params

    def _resume_from_mmap(self, old_rank_dir: Path, old_snapshots_root: Path | None = None):
        from FVdomain.Load_mesh import H5CFDdataset

        old_rank_dir = Path(old_rank_dir)
        self._rank_dir.mkdir(parents=True, exist_ok=True)
        slot_idx = 0
        while (old_rank_dir / f"slot_{slot_idx}").exists():
            old_slot_dir = old_rank_dir / f"slot_{slot_idx}"
            new_slot_dir = self._rank_dir / f"slot_{slot_idx}"
            with open(old_slot_dir / "meta.json") as f:
                old_meta = json.load(f)
            h5_hash = old_meta["h5_hash"]

            current_mesh_dir = self._mesh_cache._cache_dir / h5_hash
            if not (current_mesh_dir / "ready.flag").exists():
                old_mesh_dir = old_rank_dir.parent / "mesh_cache" / h5_hash
                if not (old_mesh_dir / "ready.flag").exists():
                    raise RuntimeError(f"Cannot resume: missing static mesh {h5_hash}")
                if self.rank == 0:
                    current_mesh_dir.mkdir(parents=True, exist_ok=True)
                    for fname in ("manifest.json", "static.bin"):
                        src, dst = old_mesh_dir / fname, current_mesh_dir / fname
                        if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
                            tmp = current_mesh_dir / f"{fname}.tmp.{os.getpid()}"
                            shutil.copy2(str(src), str(tmp))
                            os.replace(str(tmp), str(dst))
                    flag_tmp = current_mesh_dir / f"ready.flag.tmp.{os.getpid()}"
                    flag_tmp.touch()
                    os.replace(str(flag_tmp), str(current_mesh_dir / "ready.flag"))

            manifest_p, static_p = current_mesh_dir / "manifest.json", current_mesh_dir / "static.bin"
            for _ in range(3600):
                if (current_mesh_dir / "ready.flag").exists() and manifest_p.exists() and static_p.exists():
                    try:
                        if static_p.stat().st_size == json.loads(manifest_p.read_text())["total_bytes"]:
                            break
                    except (json.JSONDecodeError, KeyError, OSError):
                        pass
                time.sleep(0.5)
            else:
                raise RuntimeError(f"Timed out waiting for shared mesh {h5_hash} during resume")

            shared_mesh = self._mesh_cache._stores.get(h5_hash)
            if shared_mesh is None:
                shared_mesh = SharedMeshStore.open(current_mesh_dir)
                self._mesh_cache._stores[h5_hash] = shared_mesh

            new_slot_dir.mkdir(parents=True, exist_ok=True)
            if old_slot_dir.resolve() != new_slot_dir.resolve():
                shutil.copy2(str(old_slot_dir / "phi.bin"), str(new_slot_dir / "phi.bin"))
                with open(new_slot_dir / "meta.json", "w") as f:
                    json.dump(dict(old_meta), f, indent=2)
            slot = CaseSlot.open(new_slot_dir, shared_mesh)

            temp_ds = H5CFDdataset(params=self.params, case_list=[old_meta["h5_path"]])
            temp_ds.set_pending_global_slot_index(global_slot_index(slot_idx, self.rank, self._world_size))
            ovr = dict(old_meta.get("case_parameters") or {})
            ovr.update(old_meta.get("sampled_parameters") or {})
            if ovr:
                temp_ds.set_pending_overrides(ovr)
            mesh_dict, _ = temp_ds[0]
            pm = mesh_dict.get("parametric_manager")
            orig = (mesh_dict.get("fvconfig") or {}).get("parameter") or {}
            unpin = {
                k: orig[k]
                for k in (old_meta.get("sampled_parameters") or {})
                if k in orig and k not in (old_meta.get("case_parameters") or {})
            }
            if pm is not None and unpin:
                pm.register_from_toml(unpin)
            slot.meta_objects = _strip_static_tensors(mesh_dict)
            for key in ("case_name", "case_parameters"):
                if old_meta.get(key) is not None:
                    slot.meta_objects[key] = old_meta[key]
            self.slots.append(slot)
            self.slot_queue.append(slot_idx)
            slot_idx += 1

        if not self.slots:
            raise RuntimeError(f"No slots found under {old_rank_dir}")

        for snap_root in (old_snapshots_root, self._snapshots_root):
            if snap_root is not None and self._try_restore_snapshot_from(snap_root):
                break

        self.slot_queue = deque(
            sorted(range(len(self.slots)), key=lambda idx: (-int(self.slots[idx].time_step), idx))
        )
        self._sync_params()
        if self.loss_mode == "data_driven":
            manifest = self._find_labels_manifest(self.params.dataset_dir)
            if manifest is None:
                raise RuntimeError("loss_mode='data_driven' requires labels_manifest.json on resume")
            self._attach_labels_to_slots(manifest)
            self._has_labels = True
        self._maybe_init_gpu_pool()
        return self.params

    def _try_restore_snapshot_from(self, snap_root) -> bool:
        snap_root = Path(snap_root)
        if not snap_root.exists():
            return False
        rank_key = f"rank_{self.rank}"
        for snap_dir in sorted(
            [d for d in snap_root.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        ):
            meta_file = snap_dir / f"snapshot_meta_rank_{self.rank}.json"
            if not meta_file.exists():
                meta_file = snap_dir / "snapshot_meta.json"
            rank_dir = snap_dir / rank_key
            if not (meta_file.exists() and rank_dir.exists()):
                continue
            try:
                rank_meta = json.loads(meta_file.read_text()).get(rank_key, {})
            except Exception:
                continue
            arrays = []
            for i, slot in enumerate(self.slots):
                pf = rank_dir / f"slot_{i}_phi.bin"
                arr = np.fromfile(str(pf), dtype=np.float32) if pf.exists() else None
                if arr is None or arr.size != slot._meta["num_cpd_cells"] * slot._meta["phi_channels"]:
                    break
                arrays.append(arr.reshape(slot._meta["num_cpd_cells"], slot._meta["phi_channels"]))
            else:
                for i, (slot, arr) in enumerate(zip(self.slots, arrays)):
                    slot._phi_mmap[:] = arr
                    slot._phi_mmap.flush()
                    if f"slot_{i}" in rank_meta:
                        slot._meta["time_step"] = rank_meta[f"slot_{i}"]["time_step"]
                        _atomic_json_write(slot._slot_dir / "meta.json", slot._meta)
                logger.info("Restored phi from snapshot: %s", snap_dir)
                return True
        return False


    @staticmethod
    def _fvpatch_sig(fvp) -> tuple:
        nb = getattr(fvp, "neighbor_patch_idx", getattr(fvp, "neighbour_patch_idx", None))
        return (
            tuple(fvp.start_idx_face.reshape(-1).tolist()),
            tuple(fvp.end_idx_face.reshape(-1).tolist()),
            tuple(fvp.start_idx_cpd_cell.reshape(-1).tolist()),
            tuple(fvp.end_idx_cpd_cell.reshape(-1).tolist()),
            tuple(nb.reshape(-1).tolist()) if nb is not None else (),
        )

    def _maybe_init_gpu_pool(self):
        self.gpu_pool_enabled = False
        self._gpu_batch_templates.clear()
        self._hetero_batch_templates.clear()
        self._gpu_unique_meshes.clear()
        self._bcpatch_batch_cache.clear()
        self._slot_mesh_keys = []
        self.slot_n_cpd = []
        self.slot_cpd_ptr = None
        self.gpu_phi = self.gpu_init_phi = self.gpu_phi_flat = self.gpu_init_phi_flat = None
        self._gpu_phi_views = []
        self._gpu_init_phi_views = []
        self._last_batch_cpd = None

        if not self.slots:
            self.same_mesh = False
            self._single_n_cpd = None
            return

        s0 = self.slots[0]
        n_cpd0 = int(s0._meta["num_cpd_cells"])
        n_ch0 = int(s0._meta["phi_channels"])
        theta_w0 = int(s0.meta_objects["theta_PDE"].numel())
        h0 = s0._meta["h5_hash"]
        fvp0 = s0.meta_objects["fvpatch"]
        same_mesh = True
        channels_compat = True
        for s in self.slots:
            self.slot_n_cpd.append(int(s._meta["num_cpd_cells"]))
            fvp_s = s.meta_objects["fvpatch"]
            self._slot_mesh_keys.append((s._meta["h5_hash"], id(s._shared_mesh), self._fvpatch_sig(fvp_s)))
            if int(s._meta["phi_channels"]) != n_ch0 or int(s.meta_objects["theta_PDE"].numel()) != theta_w0:
                channels_compat = False
            if same_mesh and (
                int(s._meta["num_cpd_cells"]) != n_cpd0
                or int(s._meta["phi_channels"]) != n_ch0
                or s._meta["h5_hash"] != h0
                or s._shared_mesh is not s0._shared_mesh
                or not torch.equal(fvp_s.start_idx_face, fvp0.start_idx_face)
                or not torch.equal(fvp_s.start_idx_cpd_cell, fvp0.start_idx_cpd_cell)
            ):
                same_mesh = False

        self.same_mesh = same_mesh
        self._single_n_cpd = n_cpd0 if same_mesh else None

        pref = self._gpu_pool_pref
        if pref is None:
            env_gp = os.environ.get("FVGN_GPU_POOL")
            if env_gp is not None:
                pref = env_gp.strip().lower() not in ("0", "false", "no", "off")
            else:
                pref = getattr(self.params, "gpu_pool", True) if self.params is not None else True
        if not pref or not channels_compat or self.loss_mode != "fvm" or self.device is None:
            return
        dev = torch.device(self.device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return

        num_slots = len(self.slots)
        total_cpd = sum(self.slot_n_cpd)
        unique_stores = {id(s._shared_mesh): s._shared_mesh for s in self.slots if s._shared_mesh is not None}
        static_bytes = sum(
            int(st.manifest.get("total_bytes", 0)) for st in unique_stores.values() if st.manifest is not None
        )
        pool_bytes = 2 * total_cpd * n_ch0 * 4 + (0 if same_mesh else static_bytes)
        max_gb = float(os.environ.get("FVGN_GPU_POOL_MAX_GB", "8.0"))
        if pool_bytes >= max_gb * (1024**3):
            return

        self._gpu_dev = dev
        self._gpu_n_cpd = n_cpd0 if same_mesh else 0
        self._gpu_n_ch = n_ch0

        ptr_list = [0]
        for nc in self.slot_n_cpd:
            ptr_list.append(ptr_list[-1] + nc)
        self.slot_cpd_ptr = torch.tensor(ptr_list, dtype=torch.long, device=dev)

        if same_mesh:
            self.gpu_phi = torch.stack([s.phi_tensor.to(torch.float32) for s in self.slots], dim=0).contiguous().to(dev)
            self.gpu_init_phi = torch.stack(
                [s.meta_objects["init_phi_cpd_cell"].to(torch.float32) for s in self.slots], dim=0
            ).contiguous().to(dev)
            self.gpu_phi_flat = self.gpu_phi.view(-1, n_ch0)
            self.gpu_init_phi_flat = self.gpu_init_phi.view(-1, n_ch0)
            self._gpu_phi_views = [self.gpu_phi[i] for i in range(num_slots)]
            self._gpu_init_phi_views = [self.gpu_init_phi[i] for i in range(num_slots)]
        else:
            self.gpu_phi_flat = torch.cat([s.phi_tensor.to(torch.float32) for s in self.slots], dim=0).contiguous().to(dev)
            self.gpu_init_phi_flat = torch.cat(
                [s.meta_objects["init_phi_cpd_cell"].to(torch.float32) for s in self.slots], dim=0
            ).contiguous().to(dev)
            self._gpu_phi_views = [self.gpu_phi_flat[ptr_list[i] : ptr_list[i + 1]] for i in range(num_slots)]
            self._gpu_init_phi_views = [self.gpu_init_phi_flat[ptr_list[i] : ptr_list[i + 1]] for i in range(num_slots)]

        self.gpu_theta_pde = torch.cat(
            [s.meta_objects["theta_PDE"].to(torch.float32).view(1, -1) for s in self.slots], dim=0
        ).contiguous().to(dev)
        self._cpu_time_steps = [int(s.time_step) for s in self.slots]
        self.gpu_time_step = torch.tensor(self._cpu_time_steps, dtype=torch.float32, device=dev)
        self.gpu_pRef_valid_mask = torch.tensor(
            [bool(s.meta_objects["fvconfig"]["fvSolution"]["pRefCell_valid_mask"]) for s in self.slots],
            dtype=torch.bool, device=dev,
        )
        self.gpu_pRefCell_local = torch.tensor(
            [max(0, int(s.meta_objects["fvconfig"]["fvSolution"]["pRefCell"])) for s in self.slots],
            dtype=torch.long, device=dev,
        )
        self.gpu_pRefValue = torch.tensor(
            [float(s.meta_objects["fvconfig"]["fvSolution"]["pRefValue"]) for s in self.slots],
            dtype=torch.float32, device=dev,
        )
        self.gpu_dt = torch.tensor(
            [float(s.meta_objects["fvconfig"]["theta_pde"]["dt"]["val"]) for s in self.slots],
            dtype=torch.float32, device=dev,
        )
        self._slot_fvpatches = [s.meta_objects["fvpatch"] for s in self.slots]
        self._slot_bcpatches = [s.meta_objects["bcpatch"] for s in self.slots]
        self._slot_case_name_tensors = [
            torch.tensor([ord(c) for c in str(s.meta_objects["case_name"])], dtype=torch.long, device=dev)
            for s in self.slots
        ]

        if not same_mesh:
            for s_idx, m_key in enumerate(self._slot_mesh_keys):
                if m_key not in self._gpu_unique_meshes:
                    self._gpu_unique_meshes[m_key] = self._build_gpu_single_mesh_entry(s_idx)

        self.gpu_pool_enabled = True
        bs = int(getattr(self.params, "batch_size", 1)) if self.params is not None else 1
        if same_mesh and 1 <= bs <= num_slots:
            self._get_or_build_gpu_template(bs)

    def _build_gpu_single_mesh_entry(self, slot_idx: int) -> dict:
        from FVdomain.Graph_dataset import HeteroGraphDataset

        if getattr(self, "_template_dataset", None) is None:
            self._template_dataset = HeteroGraphDataset(self)
        g_i, fvp_i, _ = self._template_dataset._get_mmap(int(slot_idx))
        dev = self._gpu_dev
        cyc = g_i["face"].cyclic_face.to(dev)
        if cyc.numel() == 0:
            cyc = cyc.reshape(2, 0)
        b_mat = g_i["cpd_cell"].single_B_cell_to_cell.to(dev)
        if b_mat.ndim == 2:
            b_mat = b_mat.unsqueeze(-1)
        return {
            "num_nodes": int(g_i._num_nodes),
            "num_faces": int(g_i._num_faces),
            "num_cells": int(g_i._num_cells),
            "num_cpd_cells": int(g_i._num_cpd_cells),
            "num_twoway_edges": int(g_i._num_twoway_edges),
            "fvpatch": fvp_i,
            "node_pos": g_i["node"].pos.to(dev).contiguous(),
            "face_pos": g_i["face"].pos.to(dev).contiguous(),
            "face_area": g_i["face"].face_area.to(dev).contiguous(),
            "face_type": g_i["face"].face_type.to(dev).contiguous(),
            "face_normal": g_i["face"].face_normal.to(dev).contiguous(),
            "cyclic_face": cyc.contiguous(),
            "cell_volume": g_i["cell"].cell_volume.to(dev).contiguous(),
            "cells_face_unv": g_i["cell"].cells_face_unv.to(dev).contiguous(),
            "cpd_pos": g_i["cpd_cell"].pos.to(dev).contiguous(),
            "cell_type": g_i["cpd_cell"].cell_type.to(dev).contiguous(),
            "A_cell_to_cell": g_i["cpd_cell"].A_cell_to_cell.to(dev).contiguous(),
            "single_B_cell_to_cell": b_mat.contiguous(),
            "gnn_edge_counts": g_i["cpd_cell"].gnn_edge_counts.to(dev).contiguous(),
            "gnn_edge_indices": g_i["cpd_cell"].gnn_edge_indices.to(dev).contiguous(),
            "gnn_edge_sort_idx": g_i["cpd_cell"].gnn_edge_sort_idx.to(dev).contiguous(),
            "face_node": g_i["face", "contains", "node"].face_node.to(dev).contiguous(),
            "cells_node": g_i["cell", "contains", "node"].cells_node.to(dev).contiguous(),
            "cells_node_ptr": g_i["cell", "contains", "node"].cells_node_ptr.to(dev).contiguous(),
            "cells_face": g_i["cell", "bounded_by", "face"].cells_face.to(dev).contiguous(),
            "cells_face_ptr": g_i["cell", "bounded_by", "face"].cells_face_ptr.to(dev).contiguous(),
            "face_node_ptr": g_i["face", "bounds", "cell"].face_node_ptr.to(dev).contiguous(),
            "neighbor_cpd_cell": g_i["cpd_cell", "neighbors", "cpd_cell"].neighbor_cpd_cell.to(dev).contiguous(),
            "neighbor_cpd_cell_x": g_i["cpd_cell", "wlsq_neighbors", "cpd_cell"].neighbor_cpd_cell_x.to(dev).contiguous(),
        }

    def _get_or_build_gpu_template(self, B: int) -> dict:
        B = int(B)
        cached = self._gpu_batch_templates.get(B)
        if cached is not None:
            return cached
        from torch_geometric.data import Batch as HeteroBatch
        from FVdomain.Graph_dataset import HeteroGraphDataset
        from FVdomain.SetBC.FvPatch import fvPatchBatch

        if getattr(self, "_template_dataset", None) is None:
            self._template_dataset = HeteroGraphDataset(self)
        g_list, fvp_list = [], []
        for _ in range(B):
            g_i, fvp_i, _ = self._template_dataset._get_mmap(0)
            g_list.append(g_i)
            fvp_list.append(fvp_i)

        batch_hd = HeteroBatch.from_data_list(g_list).to(
            self._gpu_dev, exclude_keys=["slot_idx", "graph_index", "global_idx"]
        )
        fvpatch_batch = fvPatchBatch.from_fvpatch_list(fvp_list, g_list)

        node_static = {}
        for ntype in batch_hd.node_types:
            st = batch_hd[ntype]
            if ntype == "cpd_cell":
                skip = {"x", "init_x", "slot_idx", "global_idx", "phi_GT_cpd_cell"}
            elif ntype == "graph":
                skip = {
                    "theta_PDE", "time_step", "dt", "case_name", "graph_index",
                    "pRefCell_valid_mask", "pRefCell", "pRefValue",
                }
            else:
                skip = set()
            node_static[ntype] = {k: v for k, v in st.items() if k not in skip}

        edge_static = {etype: dict(batch_hd[etype].items()) for etype in batch_hd.edge_types}
        cpd_offsets = torch.arange(B, device=self._gpu_dev, dtype=torch.long) * self._gpu_n_cpd
        cpd_ptr_i32 = torch.arange(B + 1, device=self._gpu_dev, dtype=torch.int32) * self._gpu_n_cpd
        node_static["cpd_cell"]["ptr"] = cpd_ptr_i32.to(torch.long)

        tmpl = {
            "B": B,
            "batch_hd_ref": batch_hd,
            "node_static": node_static,
            "edge_static": edge_static,
            "fvpatch_batch": fvpatch_batch,
            "cpd_offsets": cpd_offsets,
            "cpd_ptr_i32": cpd_ptr_i32,
            "mesh_keys": (self._slot_mesh_keys[0],) * B,
        }
        self._gpu_batch_templates[B] = tmpl
        return tmpl

    def _get_or_build_hetero_gpu_template(self, slot_list: list[int]) -> dict:
        from FVdomain.SetBC.FvPatch import fvPatchBatch

        mesh_keys = tuple(self._slot_mesh_keys[s] for s in slot_list)
        cached = self._hetero_batch_templates.get(mesh_keys)
        if cached is not None:
            self._hetero_batch_templates.move_to_end(mesh_keys)
            return cached

        dev = self._gpu_dev
        B = len(slot_list)
        entries = [self._gpu_unique_meshes[mk] for mk in mesh_keys]

        def _prefix(vals):
            out = [0]
            for v in vals:
                out.append(out[-1] + v)
            return out

        n_nodes = [e["num_nodes"] for e in entries]
        n_faces = [e["num_faces"] for e in entries]
        n_cells = [e["num_cells"] for e in entries]
        n_cpds = [e["num_cpd_cells"] for e in entries]
        n_twoway = [e["num_twoway_edges"] for e in entries]
        off_node, off_face, off_cell, off_cpd, off_twoway = (
            _prefix(n_nodes), _prefix(n_faces), _prefix(n_cells), _prefix(n_cpds), _prefix(n_twoway)
        )
        b_ids = torch.arange(B, device=dev, dtype=torch.long)
        cyc_parts = [e["cyclic_face"] + off_face[b] for b, e in enumerate(entries) if e["cyclic_face"].numel() > 0]
        cyclic_cat = torch.cat(cyc_parts, dim=1) if cyc_parts else torch.zeros((2, 0), dtype=torch.long, device=dev)
        cpd_ptr_long = torch.tensor(off_cpd, device=dev, dtype=torch.long)
        cpd_ptr_i32 = cpd_ptr_long.to(torch.int32)

        node_static = {
            "node": {
                "pos": torch.cat([e["node_pos"] for e in entries], dim=0),
                "batch": torch.repeat_interleave(b_ids, torch.tensor(n_nodes, device=dev, dtype=torch.long)),
                "num_nodes": off_node[-1],
            },
            "face": {
                "pos": torch.cat([e["face_pos"] for e in entries], dim=0),
                "face_area": torch.cat([e["face_area"] for e in entries], dim=0),
                "face_type": torch.cat([e["face_type"] for e in entries], dim=0),
                "face_normal": torch.cat([e["face_normal"] for e in entries], dim=0),
                "cyclic_face": cyclic_cat,
                "batch": torch.repeat_interleave(b_ids, torch.tensor(n_faces, device=dev, dtype=torch.long)),
                "num_nodes": off_face[-1],
            },
            "cell": {
                "cell_volume": torch.cat([e["cell_volume"] for e in entries], dim=0),
                "cells_face_unv": torch.cat([e["cells_face_unv"] for e in entries], dim=0),
                "batch": torch.repeat_interleave(b_ids, torch.tensor(n_cells, device=dev, dtype=torch.long)),
                "num_nodes": off_cell[-1],
            },
            "cpd_cell": {
                "pos": torch.cat([e["cpd_pos"] for e in entries], dim=0),
                "cell_type": torch.cat([e["cell_type"] for e in entries], dim=0),
                "A_cell_to_cell": torch.cat([e["A_cell_to_cell"] for e in entries], dim=0),
                "single_B_cell_to_cell": torch.cat([e["single_B_cell_to_cell"] for e in entries], dim=0),
                "gnn_edge_counts": torch.cat([e["gnn_edge_counts"] for e in entries], dim=0),
                "gnn_edge_indices": torch.cat([e["gnn_edge_indices"] + off_cpd[b] for b, e in enumerate(entries)], dim=0),
                "gnn_edge_sort_idx": torch.cat([e["gnn_edge_sort_idx"] + off_twoway[b] for b, e in enumerate(entries)], dim=0),
                "batch": torch.repeat_interleave(b_ids, torch.tensor(n_cpds, device=dev, dtype=torch.long)),
                "ptr": cpd_ptr_long,
                "num_nodes": off_cpd[-1],
            },
            "graph": {"batch": b_ids, "num_nodes": B},
        }
        edge_static = {
            ("face", "contains", "node"): {
                "face_node": torch.cat([e["face_node"] + off_node[b] for b, e in enumerate(entries)], dim=0),
            },
            ("cell", "contains", "node"): {
                "cells_node": torch.cat([e["cells_node"] + off_node[b] for b, e in enumerate(entries)], dim=0),
                "cells_node_ptr": torch.cat([e["cells_node_ptr"] + off_cell[b] for b, e in enumerate(entries)], dim=0),
            },
            ("cell", "bounded_by", "face"): {
                "cells_face": torch.cat([e["cells_face"] + off_face[b] for b, e in enumerate(entries)], dim=0),
                "cells_face_ptr": torch.cat([e["cells_face_ptr"] + off_cell[b] for b, e in enumerate(entries)], dim=0),
            },
            ("face", "bounds", "cell"): {
                "face_node_ptr": torch.cat([e["face_node_ptr"] + off_face[b] for b, e in enumerate(entries)], dim=0),
            },
            ("cpd_cell", "neighbors", "cpd_cell"): {
                "neighbor_cpd_cell": torch.cat([e["neighbor_cpd_cell"] + off_cpd[b] for b, e in enumerate(entries)], dim=1),
            },
            ("cpd_cell", "wlsq_neighbors", "cpd_cell"): {
                "neighbor_cpd_cell_x": torch.cat([e["neighbor_cpd_cell_x"] + off_cpd[b] for b, e in enumerate(entries)], dim=1),
            },
        }
        fvpatch_batch = fvPatchBatch.from_fvpatch_list(
            [e["fvpatch"] for e in entries],
            [SimpleNamespace(_num_faces=e["num_faces"], _num_cpd_cells=e["num_cpd_cells"]) for e in entries],
        )
        tmpl = {
            "B": B,
            "node_static": node_static,
            "edge_static": edge_static,
            "fvpatch_batch": fvpatch_batch,
            "cpd_offsets": cpd_ptr_long[:B].contiguous(),
            "cpd_ptr_i32": cpd_ptr_i32,
            "mesh_keys": mesh_keys,
        }
        if len(self._hetero_batch_templates) >= 64:
            self._hetero_batch_templates.popitem(last=False)
        self._hetero_batch_templates[mesh_keys] = tmpl
        return tmpl

    def load_Graph_Batch(self, slot_indices):
        if torch.is_tensor(slot_indices):
            slot_list = [int(x) for x in slot_indices.reshape(-1).tolist()]
        elif isinstance(slot_indices, int):
            slot_list = [int(slot_indices)]
        else:
            slot_list = [int(x) for x in slot_indices]

        if not self.gpu_pool_enabled:
            from FVdomain.Graph_loader import HeteroGraphLoader

            if getattr(self, "_fallback_loader", None) is None:
                self._fallback_loader = HeteroGraphLoader(base_dataset=self, batch_size=1, shuffle=False)
            fv_graph = self._fallback_loader._build_one_batch_cpu(slot_list)
            if self.device is not None:
                fv_graph = fv_graph.to(self.device, exclude_keys=["slot_idx", "graph_index", "global_idx"])
            mesh_keys = tuple(self._slot_mesh_keys[s] for s in slot_list) if self._slot_mesh_keys else None
            if self.same_mesh or (mesh_keys is not None and len(set(mesh_keys)) == 1):
                fv_graph._same_mesh = True
            if mesh_keys is not None:
                fv_graph._mesh_batch_sig = mesh_keys
            self._last_batch_cpd = fv_graph.graph_cell.batch
            return fv_graph

        from FVdomain.Graph_data import HeteroGraph
        from FVdomain.Graph_dataset import HeteroFVGraph
        from FVdomain.SetBC.onthefly.BCbase import BoundaryConditionBatch

        B = len(slot_list)
        tmpl = self._get_or_build_gpu_template(B) if self.same_mesh else self._get_or_build_hetero_gpu_template(slot_list)

        hd = HeteroGraph()
        for ntype, kv in tmpl["node_static"].items():
            st = hd[ntype]
            for k, v in kv.items():
                st[k] = v
        for etype, kv in tmpl["edge_static"].items():
            st = hd[etype]
            for k, v in kv.items():
                st[k] = v

        idx_cpu = torch.tensor(slot_list, dtype=torch.long)
        idx_gpu = idx_cpu.to(device=self._gpu_dev)

        cpd_st = hd["cpd_cell"]
        if self.same_mesh:
            cpd_st.x = self.gpu_phi.index_select(0, idx_gpu).reshape(B * self._gpu_n_cpd, self._gpu_n_ch)
            cpd_st.init_x = self.gpu_init_phi.index_select(0, idx_gpu).reshape(B * self._gpu_n_cpd, self._gpu_n_ch)
        elif B == 1:
            s0 = slot_list[0]
            cpd_st.x = self._gpu_phi_views[s0].clone()
            cpd_st.init_x = self._gpu_init_phi_views[s0].clone()
        else:
            cpd_st.x = torch.cat([self._gpu_phi_views[s] for s in slot_list], dim=0)
            cpd_st.init_x = torch.cat([self._gpu_init_phi_views[s] for s in slot_list], dim=0)
        cpd_st.slot_idx = idx_cpu

        g_st = hd["graph"]
        g_st.theta_PDE = self.gpu_theta_pde.index_select(0, idx_gpu)
        g_st.time_step = self.gpu_time_step.index_select(0, idx_gpu)
        g_st.dt = self.gpu_dt.index_select(0, idx_gpu)
        g_st.pRefCell_valid_mask = self.gpu_pRef_valid_mask.index_select(0, idx_gpu)
        g_st.pRefCell = self.gpu_pRefCell_local.index_select(0, idx_gpu) + tmpl["cpd_offsets"]
        g_st.pRefValue = self.gpu_pRefValue.index_select(0, idx_gpu)
        g_st.case_name = (
            self._slot_case_name_tensors[slot_list[0]]
            if B == 1
            else torch.cat([self._slot_case_name_tensors[s] for s in slot_list], dim=0)
        )
        g_st.graph_index = idx_cpu

        bc_key = (tuple(slot_list), tuple(id(self._slot_bcpatches[s]) for s in slot_list))
        bcpatch_batch = self._bcpatch_batch_cache.get(bc_key)
        if bcpatch_batch is None:
            bcpatch_batch = BoundaryConditionBatch.from_bcpatch_list(
                [self._slot_bcpatches[s] for s in slot_list],
                [self._slot_fvpatches[s] for s in slot_list],
            )
            if len(self._bcpatch_batch_cache) >= 128:
                self._bcpatch_batch_cache.popitem(last=False)
            self._bcpatch_batch_cache[bc_key] = bcpatch_batch
        else:
            self._bcpatch_batch_cache.move_to_end(bc_key)

        fv_graph = HeteroFVGraph(hd, tmpl["fvpatch_batch"], bcpatch_batch)
        mesh_keys = tmpl["mesh_keys"]
        fv_graph._same_mesh = self.same_mesh or (B == 1) or (len(set(mesh_keys)) == 1)
        fv_graph._mesh_batch_sig = mesh_keys
        fv_graph._cpd_ptr = tmpl["cpd_ptr_i32"]
        self._last_batch_cpd = cpd_st.batch
        return fv_graph

    def make_gpu_loader(self, batch_size: int = 1, shuffle: bool = True) -> GPUBatchLoader:
        return GPUBatchLoader(self, batch_size=batch_size, shuffle=shuffle)

    def get_slot_phi(self, slot_idx: int) -> torch.Tensor:
        s = int(slot_idx)
        return self._gpu_phi_views[s].detach().cpu().clone() if self.gpu_pool_enabled else self.slots[s].phi_tensor.clone()

    def sync_gpu_to_mmap(self):
        if not self.gpu_pool_enabled or self.gpu_phi_flat is None:
            return
        if self.same_mesh and self.gpu_phi is not None:
            phi_all = self.gpu_phi.detach().cpu()
            for i, slot in enumerate(self.slots):
                slot.update_phi(phi_all[i])
                slot._meta["time_step"] = int(self._cpu_time_steps[i])
                _atomic_json_write(slot._slot_dir / "meta.json", slot._meta)
        else:
            phi_flat = self.gpu_phi_flat.detach().cpu()
            off = 0
            for i, slot in enumerate(self.slots):
                nc = self.slot_n_cpd[i]
                slot.update_phi(phi_flat[off : off + nc])
                off += nc
                slot._meta["time_step"] = int(self._cpu_time_steps[i])
                _atomic_json_write(slot._slot_dir / "meta.json", slot._meta)

    def _set_reset_env_flag(self, flag=False, rst_time=1):
        self.reset_env_flag = flag
        self.rst_time = rst_time

    def reset_slot(self, slot_idx: int):
        s = int(slot_idx)
        if self.gpu_pool_enabled:
            self._gpu_phi_views[s].copy_(self._gpu_init_phi_views[s])
            self.gpu_time_step[s].zero_()
            self._cpu_time_steps[s] = 0
            return
        slot = self.slots[s]
        slot.reset_phi(slot.meta_objects["init_phi_cpd_cell"].to(torch.float32))

    def refresh_pool(self, phi_new_cpd_cell, slot_idx):
        if torch.is_tensor(slot_idx):
            slot_list = [int(x) for x in slot_idx.reshape(-1).tolist()]
        elif isinstance(slot_idx, int):
            slot_list = [slot_idx]
        else:
            slot_list = [int(x) for x in slot_idx]

        if self.gpu_pool_enabled:
            B = len(slot_list)
            phi_det = phi_new_cpd_cell.detach()
            if phi_det.device != self._gpu_dev or phi_det.dtype != torch.float32:
                phi_det = phi_det.to(device=self._gpu_dev, dtype=torch.float32)
            if B == 1:
                s = slot_list[0]
                self._gpu_phi_views[s].copy_(phi_det)
                self.gpu_time_step[s] += 1.0
                self._cpu_time_steps[s] += 1
            elif self.same_mesh and self.gpu_phi is not None:
                phi_3d = phi_det.reshape(B, self._gpu_n_cpd, self._gpu_n_ch)
                idx_gpu = torch.tensor(slot_list, dtype=torch.long, device=self._gpu_dev)
                self.gpu_phi.index_copy_(0, idx_gpu, phi_3d)
                self.gpu_time_step.index_add_(0, idx_gpu, torch.ones(B, dtype=torch.float32, device=self._gpu_dev))
                for s in slot_list:
                    self._cpu_time_steps[s] += 1
            else:
                off = 0
                for s in slot_list:
                    nc = self.slot_n_cpd[s]
                    self._gpu_phi_views[s].copy_(phi_det[off : off + nc])
                    off += nc
                    self._cpu_time_steps[s] += 1
                idx_gpu = torch.tensor(slot_list, dtype=torch.long, device=self._gpu_dev)
                self.gpu_time_step.index_add_(0, idx_gpu, torch.ones(B, dtype=torch.float32, device=self._gpu_dev))
            return

        phi_cpu = phi_new_cpd_cell.detach().cpu()
        if len(slot_list) == 1:
            idx = slot_list[0]
            self.slots[idx].update_phi(phi_cpu)
            self.slots[idx].increment_time_step()
            self._auto_reset_if_past_labels(idx)
        else:
            offset = 0
            for s_idx in slot_list:
                n_cpd = self.slots[s_idx]._meta["num_cpd_cells"]
                self.slots[s_idx].update_phi(phi_cpu[offset : offset + n_cpd])
                self.slots[s_idx].increment_time_step()
                self._auto_reset_if_past_labels(s_idx)
                offset += n_cpd

    def _auto_reset_if_past_labels(self, slot_idx):
        slot = self.slots[slot_idx]
        if self.loss_mode == "data_driven" and slot.has_labels and slot.time_step >= slot.n_timesteps:
            slot.reset_phi(slot.meta_objects["init_phi_cpd_cell"].to(torch.float32))


    def _infer_batch_cpd(self, n_total: int, dev: torch.device, batch_cpd_cell=None):
        if batch_cpd_cell is not None:
            return batch_cpd_cell.to(device=dev, dtype=torch.long)
        solver_ref = getattr(self, "_solver_ref", None)
        if solver_ref is not None and getattr(solver_ref, "batch_cpd_cell", None) is not None:
            if int(solver_ref.batch_cpd_cell.numel()) == n_total:
                return solver_ref.batch_cpd_cell.to(device=dev, dtype=torch.long)
        last_b = getattr(self, "_last_batch_cpd", None)
        if last_b is not None and int(last_b.numel()) == n_total:
            return last_b.to(device=dev, dtype=torch.long)
        n_single = getattr(self, "_single_n_cpd", None)
        if getattr(self, "same_mesh", False) and n_single and n_total % n_single == 0:
            B_inf = n_total // n_single
            if B_inf > 1:
                return torch.arange(B_inf, device=dev, dtype=torch.long).repeat_interleave(n_single)
        return None

    def mirror_project(self, phi_new_cpd_cell, cpd_cell_pos, batch_cpd_cell=None):
        plane = self.params.fvconfig.get("fvSolution", {}).get("bufferMirror")
        if plane is None:
            return phi_new_cpd_cell
        axis, origin = int(plane["axis"]), float(plane["origin"])
        dev = phi_new_cpd_cell.device
        n_total = int(phi_new_cpd_cell.shape[0])
        if getattr(self, "_mirror_index_cache", None) is None:
            self._mirror_index_cache = {}

        def _perm(pos_t: torch.Tensor) -> torch.Tensor:
            n_s = int(pos_t.shape[0])
            key_m = (
                n_s, axis, origin,
                float(pos_t[0, 0].item()),
                float(pos_t[n_s // 2, min(1, pos_t.shape[1] - 1)].item()),
                float(pos_t[-1, -1].item()),
            )
            p_dev = self._mirror_index_cache.get((key_m, dev))
            if p_dev is not None:
                return p_dev
            p_cpu = self._mirror_index_cache.get((key_m, torch.device("cpu")))
            if p_cpu is None:
                pos_np = pos_t.detach().cpu().numpy()
                reflected = pos_np.copy()
                reflected[:, axis] = 2.0 * origin - reflected[:, axis]
                dist, idx_np = cKDTree(pos_np).query(reflected)
                if dist.max() > 1e-6:
                    raise ValueError(f"Mesh is not mirror-symmetric about {plane}: {dist.max():.3e}")
                p_cpu = torch.from_numpy(idx_np).to(torch.long)
                self._mirror_index_cache[(key_m, torch.device("cpu"))] = p_cpu
            p_dev = p_cpu.to(dev)
            self._mirror_index_cache[(key_m, dev)] = p_dev
            return p_dev

        n_single = getattr(self, "_single_n_cpd", None)
        batch_cpd = self._infer_batch_cpd(n_total, dev, batch_cpd_cell)
        if getattr(self, "same_mesh", False) and n_single and n_total % n_single == 0:
            B = n_total // n_single
            perm = _perm(cpd_cell_pos[:n_single])
            phi_v = phi_new_cpd_cell.view(B, n_single, -1)
            mir = phi_v[:, perm, :].clone()
            mir[:, :, axis] = -mir[:, :, axis]
            return (0.5 * (phi_v + mir)).reshape_as(phi_new_cpd_cell)
        if batch_cpd is None or int(batch_cpd.max().item()) == 0:
            perm = _perm(cpd_cell_pos)
            mir = phi_new_cpd_cell[perm].clone()
            mir[:, axis] = -mir[:, axis]
            return 0.5 * (phi_new_cpd_cell + mir)
        B = int(batch_cpd.max().item()) + 1
        out = phi_new_cpd_cell.clone()
        for b in range(B):
            mask_b = batch_cpd == b
            perm_b = _perm(cpd_cell_pos[mask_b])
            phi_b = phi_new_cpd_cell[mask_b]
            mir_b = phi_b[perm_b].clone()
            mir_b[:, axis] = -mir_b[:, axis]
            out[mask_b] = 0.5 * (phi_b + mir_b)
        return out

    def gauge_project(
        self,
        phi_new_cpd_cell,
        mask_interior_cell,
        mom_residual=None,
        cell_volume=None,
        cell_pos=None,
        dt_cell=None,
        batch_cpd_cell=None,
    ):
        gauge = self.params.fvconfig.get("fvSolution", {}).get("bufferGauge", False)
        if not gauge:
            return phi_new_cpd_cell
        dev, dtype = phi_new_cpd_cell.device, phi_new_cpd_cell.dtype
        n_total = int(phi_new_cpd_cell.shape[0])
        interior = mask_interior_cell.view(-1).to(dev)
        phi_out = phi_new_cpd_cell.clone()
        batch_cpd = self._infer_batch_cpd(n_total, dev, batch_cpd_cell)
        n_single = getattr(self, "_single_n_cpd", None)
        is_homog_multi = (
            getattr(self, "same_mesh", False)
            and n_single is not None
            and n_total > n_single
            and n_total % n_single == 0
        )

        if not is_homog_multi and (batch_cpd is None or int(batch_cpd.max().item()) == 0):
            bulk_rate = 0.0
            if gauge == "bulk":
                vol = cell_volume.detach().view(-1, 1).to(device=dev, dtype=dtype)
                bulk = (phi_out[interior, :3] * vol).sum(0) / vol.sum()
                phi_out[interior, :3] -= bulk.to(dtype)
                bulk_rate = bulk / dt_cell.detach().view(-1)[0].to(device=dev, dtype=dtype)
            mean_t = self.params.fvconfig.get("fvSolution", {}).get("bufferMeanT")
            if mean_t is not None:
                vol = cell_volume.detach().view(-1).to(device=dev, dtype=dtype)
                phi_out[interior, 4] -= ((phi_out[interior, 4] * vol).sum() / vol.sum() - float(mean_t)).to(dtype)
            if gauge in ("linear", "bulk"):
                vol_sum = cell_volume.detach().to(device=dev, dtype=dtype).sum()
                grad = (mom_residual.detach().to(device=dev, dtype=dtype).reshape(-1, 3).sum(0) / vol_sum) - bulk_rate
                phi_out[:, 3] -= (cell_pos.detach().to(device=dev, dtype=dtype) @ grad).to(dtype)
            phi_out[:, 3] -= phi_out[interior, 3].mean()
            return phi_out

        if is_homog_multi:
            B = n_total // n_single
            int_s = interior[:n_single]
            phi_3d = phi_out.view(B, n_single, -1)
            n_int_s = int(cell_volume.numel()) // B if cell_volume is not None else int(int_s.sum().item())
            bulk_rate = 0.0
            if gauge == "bulk":
                vol_3d = cell_volume.detach().to(device=dev, dtype=dtype).view(B, n_int_s, 1)
                bulk = (phi_3d[:, int_s, :3] * vol_3d).sum(dim=1, keepdim=True) / vol_3d.sum(dim=1, keepdim=True)
                phi_3d[:, int_s, :3] -= bulk
                dt_b = dt_cell.detach().to(device=dev, dtype=dtype).view(B, n_int_s)[:, 0].view(B, 1, 1)
                bulk_rate = bulk / dt_b
            mean_t = self.params.fvconfig.get("fvSolution", {}).get("bufferMeanT")
            if mean_t is not None:
                vol_t = cell_volume.detach().to(device=dev, dtype=dtype).view(B, n_int_s)
                phi_3d[:, int_s, 4] -= (
                    (phi_3d[:, int_s, 4] * vol_t).sum(dim=1, keepdim=True) / vol_t.sum(dim=1, keepdim=True) - float(mean_t)
                )
            if gauge in ("linear", "bulk"):
                vol_s = cell_volume.detach().to(device=dev, dtype=dtype).view(B, n_int_s).sum(dim=1).view(B, 1, 1)
                mom_3d = mom_residual.detach().to(device=dev, dtype=dtype).reshape(B, -1, 3)
                grad = (mom_3d.sum(dim=1, keepdim=True) / vol_s) - bulk_rate
                pos_3d = cell_pos.detach().to(device=dev, dtype=dtype).view(B, n_single, 3)
                phi_3d[:, :, 3] -= (pos_3d * grad).sum(dim=-1)
            phi_3d[:, :, 3] -= phi_3d[:, int_s, 3].mean(dim=1, keepdim=True)
            return phi_out

        B = int(batch_cpd.max().item()) + 1
        batch_int = batch_cpd[interior]
        bulk_rate = 0.0
        if gauge == "bulk":
            vol64 = cell_volume.detach().view(-1, 1).to(device=dev, dtype=torch.float64)
            vol_sum_b = scatter(vol64, batch_int, dim=0, dim_size=B, reduce="sum")
            bulk_b64 = scatter(phi_out[interior, :3].to(torch.float64) * vol64, batch_int, dim=0, dim_size=B, reduce="sum") / vol_sum_b
            phi_out[interior, :3] -= bulk_b64.to(dtype)[batch_int]
            dt_b64 = scatter(dt_cell.detach().view(-1, 1).to(device=dev, dtype=torch.float64), batch_int, dim=0, dim_size=B, reduce="mean")
            bulk_rate = bulk_b64 / dt_b64
        mean_t = self.params.fvconfig.get("fvSolution", {}).get("bufferMeanT")
        if mean_t is not None:
            vol64 = cell_volume.detach().view(-1).to(device=dev, dtype=torch.float64)
            vol_sum_b = scatter(vol64, batch_int, dim=0, dim_size=B, reduce="sum")
            t_shift = scatter(phi_out[interior, 4].to(torch.float64) * vol64, batch_int, dim=0, dim_size=B, reduce="sum") / vol_sum_b - float(mean_t)
            phi_out[interior, 4] -= t_shift.to(dtype)[batch_int]
        if gauge in ("linear", "bulk"):
            vol64 = cell_volume.detach().view(-1, 1).to(device=dev, dtype=torch.float64)
            vol_sum_b = scatter(vol64, batch_int, dim=0, dim_size=B, reduce="sum")
            mom64 = mom_residual.detach().to(device=dev, dtype=torch.float64).reshape(-1, 3)
            grad_b = ((scatter(mom64, batch_int, dim=0, dim_size=B, reduce="sum") / vol_sum_b) - bulk_rate).to(dtype)
            pos_t = cell_pos.detach().to(device=dev, dtype=dtype)
            phi_out[:, 3] -= (pos_t * grad_b[batch_cpd]).sum(dim=-1)
        p_mean_b = scatter(phi_out[interior, 3].to(torch.float64), batch_int, dim=0, dim_size=B, reduce="mean").to(dtype)
        phi_out[:, 3] -= p_mean_b[batch_cpd]
        return phi_out

    def _update_re_from_patch_velocity(self, mesh, phi_cpd_cell):
        theta_pde = mesh.get("fvconfig", {}).get("theta_pde", {})
        re_cfg = theta_pde.get("Re_config")
        if not isinstance(re_cfg, dict) or "vel" not in re_cfg:
            return
        vel_spec = re_cfg["vel"]
        pname = vel_spec.split(".", 1)[1] if vel_spec.startswith("boundaryfield.") else vel_spec
        pinfo = mesh["patch_dict"][pname]
        sf, ef = int(pinfo["start_idx_face"]), int(pinfo["end_idx_face"])
        sc, ec = int(pinfo["start_idx_cpd_cell"]), int(pinfo["end_idx_cpd_cell"])
        pa = mesh["face|face_area"][sf:ef]
        if pa.dim() == 1:
            pa = pa.unsqueeze(-1)
        v_dot_n = (phi_cpd_cell[sc:ec, 0:3] * mesh["face|face_normal"][sf:ef]).sum(dim=1, keepdim=True)
        tot_a = pa.sum().item()
        v_mag = abs((v_dot_n * pa).sum().item() / tot_a) if tot_a > 0 else 0.0
        rho_v = theta_pde.get("_resolved_rho", theta_pde["rho"]["val"] if isinstance(theta_pde["rho"], dict) else theta_pde["rho"])
        mu_v = theta_pde.get("_resolved_mu", theta_pde["mu"]["val"] if isinstance(theta_pde["mu"], dict) else theta_pde["mu"])
        theta_pde["Re"] = rho_v * v_mag * theta_pde["L"] / mu_v

    def _export_mesh_visualization(self, mesh, phi_cpd_cell, mask_interior_cell):
        fields, col = {}, 0
        for var in mesh["boundary_info"]:
            fields[var] = phi_cpd_cell[:, col : col + VAR_CONFIG[var]]
            col += VAR_CONFIG[var]
        block = self.plot_count // 50 * 50
        name = (
            f"NO.{self.plot_count}_{mesh['case_name']}_Re={mesh['fvconfig']['theta_pde']['Re']:.4f}"
            f"_dt={mesh['fvconfig']['theta_pde']['dt']['val']:.3f}"
        )
        export_vtm(
            os.path.join(self.state_save_dir, "training_results", f"{block}-{block + 50}", name, f"{name}.vtm"),
            mesh,
            mask_interior_cell,
            fields,
        )

    def reset_env(self, plot=False):
        if not self.slot_queue:
            return
        oldest_idx = self.slot_queue.popleft()
        slot = self.slots[oldest_idx]
        old_mesh = slot.meta_objects

        if self.gpu_pool_enabled and not plot:
            pm = old_mesh.get("parametric_manager")
            has_dyn = pm is not None and any(
                st.get("value_type") not in ("constant", "uniform")
                for st in pm.get_status().values()
            )
            if not has_dyn:
                self.reset_slot(oldest_idx)
                self.slot_queue.append(oldest_idx)
                return

        merged_mesh = _build_merged_mesh(slot)
        if plot:
            plot_phi = self.get_slot_phi(oldest_idx)
            mask_int = (slot.get_mesh_tensor("cpd|cell_type").clone().long() == FaceType.NORMAL).squeeze()
            self._update_re_from_patch_velocity(merged_mesh, plot_phi)
            self._export_mesh_visualization(merged_mesh, plot_phi, mask_int)
            self.plot_count += 1
            self._plot_env = False

        if self.loss_mode == "data_driven":
            slot.reset_phi(slot.meta_objects["init_phi_cpd_cell"].to(torch.float32))
        else:
            new_mesh, init_phi = merged_mesh["transform"](merged_mesh, self.params)
            if self.gpu_pool_enabled:
                init_gpu = init_phi.to(device=self._gpu_dev, dtype=torch.float32)
                self._gpu_init_phi_views[oldest_idx].copy_(init_gpu)
                self._gpu_phi_views[oldest_idx].copy_(init_gpu)
                self.gpu_theta_pde[oldest_idx].copy_(new_mesh["theta_PDE"].to(device=self._gpu_dev, dtype=torch.float32).view(-1))
                self.gpu_dt[oldest_idx] = float(new_mesh["fvconfig"]["theta_pde"]["dt"]["val"])
                self.gpu_pRef_valid_mask[oldest_idx] = bool(new_mesh["fvconfig"]["fvSolution"]["pRefCell_valid_mask"])
                self.gpu_pRefCell_local[oldest_idx] = max(0, int(new_mesh["fvconfig"]["fvSolution"]["pRefCell"]))
                self.gpu_pRefValue[oldest_idx] = float(new_mesh["fvconfig"]["fvSolution"]["pRefValue"])
                self.gpu_time_step[oldest_idx].zero_()
                self._cpu_time_steps[oldest_idx] = 0
                self._slot_fvpatches[oldest_idx] = new_mesh["fvpatch"]
                self._slot_bcpatches[oldest_idx] = new_mesh["bcpatch"]
                slot.meta_objects = _strip_static_tensors(new_mesh)
                sampled = _sampled_parameters_of(new_mesh)
                if sampled:
                    slot._meta["sampled_parameters"] = sampled
            else:
                slot.reset_phi(init_phi)
                slot.meta_objects = _strip_static_tensors(new_mesh)
                sampled = _sampled_parameters_of(new_mesh)
                if sampled:
                    slot.set_meta_entry("sampled_parameters", sampled)
        self.slot_queue.append(oldest_idx)

    def save_snapshot(self, checkpoint_idx, epoch):
        if self.gpu_pool_enabled:
            self.sync_gpu_to_mmap()
        self._snapshot_writer.snapshot(
            self.slots,
            self._snapshots_root / f"epoch_{epoch}",
            self.rank,
            {"epoch": epoch, "checkpoint_idx": checkpoint_idx},
        )
        if self.rank == 0:
            self._snapshot_writer.cleanup(self._snapshots_root, getattr(self.params, "max_snapshots", 3))

    def close(self):
        if self._snapshot_writer is not None:
            self._snapshot_writer.wait()


HeteroDataPool.mirror_project = MmapDataPool.mirror_project
HeteroDataPool.gauge_project = MmapDataPool.gauge_project
HeteroDataPool._infer_batch_cpd = MmapDataPool._infer_batch_cpd
