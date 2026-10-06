"""The pool of cases a training run cycles through, with its state kept in memory-mapped
files so a run can be resumed."""

import json
import logging
import os
import random
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from FVdomain.MmapManager import (
    AsyncSnapshotWriter, CaseSlot, MeshCacheManager, SharedMeshStore, STATIC_TENSOR_KEYS, _atomic_json_write,
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
    """Seed of a slot's parameter draws, keyed on the global slot index so a slot draws the
    same cases at any number of ranks."""
    return base + int(global_idx)


def build_rank_case_list(case_names, total_dataset_size, rank, world_size, seed=42, return_global=False):
    """Repeat the split's cases to ``total_dataset_size`` slots, shuffle them, and return
    rank ``rank``'s strided share (or the whole list)."""
    if total_dataset_size % world_size != 0:
        raise ValueError(f"dataset_size ({total_dataset_size}) must be divisible by world_size ({world_size})")
    n_cases = len(case_names)
    global_list = list(case_names) * (total_dataset_size // n_cases) + list(case_names[:total_dataset_size % n_cases])
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
    """``{name: value}`` of the numeric parameters drawn for this case."""
    return {k: v for k, v in mesh_dict["parametric_manager"].current_values().items()
            if isinstance(v, (int, float))}


def _strip_static_tensors(mesh_dict):
    return {k: v for k, v in mesh_dict.items() if k not in STATIC_TENSOR_KEYS}


def _build_merged_mesh(slot):
    """A slot's mesh dictionary with the shared static tensors put back."""
    merged = dict(slot.meta_objects)
    shared = slot._shared_mesh
    if shared is not None and shared.manifest is not None:
        for key in STATIC_TENSOR_KEYS:
            if shared.has_tensor(key):
                merged[key] = shared.get_tensor(key).clone()
    return merged


class MmapDataPool:
    """The training slots of one rank.  Each slot holds a case's current state in its own
    ``phi.bin``; the static mesh tensors are shared through ``MeshCacheManager``.  A slot is
    reset to a newly drawn case in first-in-first-out order."""

    def __init__(self, params=None, device=None, state_save_dir=None, rank=0):
        self.params = params
        self.device = device
        self.state_save_dir = state_save_dir
        self.slots: list[CaseSlot] = []
        self.slot_queue: deque[int] = deque()
        self._plot_env = True
        self.reset_env_flag = False
        self.rst_time = 1
        self.rank = rank
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._rank_dir: Path | None = None
        self._mesh_cache: MeshCacheManager | None = None
        self._snapshot_writer = AsyncSnapshotWriter()
        self._snapshots_root: Path | None = None
        self.loss_mode: str = "fvm"
        self._has_labels: bool = False
        self.plot_count = 0

    @property
    def meta_pool(self):
        return [slot.meta_objects for slot in self.slots]

    def _labels_manifest(self) -> Path:
        p = Path(self.params.labels_dir)
        if p.name != "labels_manifest.json":
            p = p / "labels_manifest.json"
        if not p.is_file():
            raise FileNotFoundError(f"--labels_dir {self.params.labels_dir}: no labels_manifest.json there")
        return p

    def _attach_labels_to_slots(self, manifest_path: Path) -> None:
        """Attach each slot's label trajectory (``labels_manifest.json`` + ``.mmap``) by case name."""
        with open(manifest_path) as f:
            manifest = json.load(f)
        labels_dir = manifest_path.parent
        mmap_by_name = {case["case_name"]: {"file": labels_dir / case["mmap_file"], "shape": tuple(case["shape"])}
                        for case in manifest["cases"]}
        dtype = manifest.get("dtype", "float32")
        unmatched = [slot._meta.get("case_name", "") for slot in self.slots
                     if slot._meta.get("case_name", "") not in mmap_by_name]
        if unmatched:
            raise RuntimeError(f"no labels in {manifest_path} for cases {unmatched[:5]}")
        for slot in self.slots:
            info = mmap_by_name[slot._meta.get("case_name", "")]
            slot.attach_labels(labels_file=info["file"], shape=info["shape"], dtype=dtype)
        print(f"[MmapDataPool] Attached labels to {len(self.slots)}/{len(self.slots)} slots from {manifest_path}")

    def _sync_params(self):
        meta = self.slots[0].meta_objects
        self.params.num_channels = meta["num_channels"]
        self.params.phi_num_channels = meta["phi_num_channels"]
        self.params.dataset_size = len(self.slots)
        self.params.fvconfig = meta["fvconfig"]
        self.params.fvschemes = meta["fvconfig"]["fvSchemes"]
        # the loss weights of fvconfig.toml override the command line
        fvsolution = self.params.fvconfig.get("fvSolution", {})
        for key in ["loss_cont", "loss_mom", "loss_energy", "loss_pRef"]:
            if key in fvsolution:
                setattr(self.params, key, fvsolution[key])

    def load_mesh_to_mmap(self, dataset_dir=None, resume=False, resume_from=None, split: str = "all",
                          loss_mode: str = "fvm", total_dataset_size: int | None = None, world_size: int = 1,
                          param_overrides=None):
        """Fill ``params.dataset_size`` slots from the mesh files under ``dataset_dir``.

        With a ``split`` of ``meta.json`` the slots cycle through that split's cases, each with
        its ``case_parameters``; otherwise every slot draws its parameters from ``bc/*.toml``.
        ``resume`` reopens the slots of an earlier run instead.
        """
        from FVdomain.Load_mesh import H5CFDdataset

        dataset_dir = dataset_dir or self.params.dataset_dir
        self.loss_mode = loss_mode
        mmap_root = Path(self.state_save_dir) / "mmap_state"
        mmap_root.mkdir(parents=True, exist_ok=True)
        self._rank_dir = mmap_root / f"rank_{self.rank}"
        self._rank_dir.mkdir(parents=True, exist_ok=True)
        self._mesh_cache = MeshCacheManager(mmap_root / "mesh_cache")
        self._snapshots_root = mmap_root / "snapshots"
        self._snapshots_root.mkdir(parents=True, exist_ok=True)

        if resume:
            if resume_from:
                old_mmap_root = Path(resume_from)
                old_rank_dir = old_mmap_root / f"rank_{self.rank}"
                if (old_rank_dir / "slot_0").exists():
                    logger.info("Resuming from old mmap state at %s", old_mmap_root)
                    old_snapshots_root = old_mmap_root / "snapshots"
                    return self._resume_from_mmap(
                        old_rank_dir, old_snapshots_root if old_snapshots_root.exists() else None)
                logger.warning("resume_from %s has no rank_%d/slot_0; trying the current run", old_mmap_root, self.rank)
            if (self._rank_dir / "slot_0").exists():
                return self._resume_from_mmap(self._rank_dir, self._snapshots_root)

        if dataset_dir.endswith(".h5"):
            valid_h5file_paths = [dataset_dir]
        else:
            valid_h5file_paths = [os.path.join(subdir, name)
                                  for subdir, _, files in os.walk(dataset_dir, followlinks=True)
                                  for name in files if name.endswith(".h5")]
        if not valid_h5file_paths:
            raise ValueError(f".h5 file not found in dataset directory: {dataset_dir}")
        logger.info("Loading dataset to mmap (rank %d)", self.rank)

        case_names_from_meta = None
        case_parameters_from_meta: dict = {}
        if split != "all":
            meta_json_path = Path(dataset_dir) / "meta.json"
            with open(meta_json_path) as f:
                meta = json.load(f)
            case_names_from_meta = list(meta["split"][split])
            case_parameters_from_meta = meta.get("case_parameters", {}) or {}
            print(f"[MmapDataPool] Using meta.json split='{split}': {len(case_names_from_meta)} cases"
                  + (" + case_parameters" if case_parameters_from_meta else ""))

        mesh_dataset = H5CFDdataset(params=self.params, case_list=valid_h5file_paths)
        rank_case_list = None
        if case_names_from_meta is not None and total_dataset_size is not None:
            rank_case_list = build_rank_case_list(case_names_from_meta, total_dataset_size, self.rank, world_size)
            if self.rank == 0:
                global_list = build_rank_case_list(
                    case_names_from_meta, total_dataset_size, 0, world_size, return_global=True)
                save_case_assignment(self.state_save_dir, global_list, total_dataset_size, world_size, seed=42)

        num_channels = None
        while len(self.slots) < self.params.dataset_size:
            for i_data, h5_path in enumerate(valid_h5file_paths):
                h5_hash = self._mesh_cache._hash_h5(h5_path)
                if case_names_from_meta is not None:
                    iter_cases = rank_case_list if rank_case_list is not None else case_names_from_meta
                    jobs = []
                    for case_name in iter_cases:
                        per_case = case_parameters_from_meta.get(case_name)
                        if param_overrides:
                            per_case = {**(per_case or {}), **param_overrides}
                        jobs.append((case_name, per_case))
                else:
                    jobs = [(None, dict(param_overrides) if param_overrides else None)]

                shared_mesh = None
                for case_name, overrides in jobs:
                    if len(self.slots) >= self.params.dataset_size:
                        break
                    if overrides:
                        mesh_dataset.set_pending_overrides(overrides)
                    mesh_dataset.set_pending_global_slot_index(
                        global_slot_index(len(self.slots), self.rank, self._world_size))
                    mesh_dict, init_phi = mesh_dataset[i_data]
                    if num_channels is None:
                        num_channels = mesh_dict["num_channels"]
                    elif mesh_dict["num_channels"] != num_channels:
                        raise ValueError(f"Number of channels mismatch: {mesh_dict['num_channels']} != {num_channels}")
                    if shared_mesh is None or case_name is None:
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
                            "case_name": case_name if case_name is not None else mesh_dict["case_name"],
                            "case_parameters": overrides if case_name is not None else None,
                        },
                    )
                    slot.meta_objects = _strip_static_tensors(mesh_dict)
                    if case_name is not None:
                        slot.meta_objects["case_name"] = case_name
                        if overrides:
                            slot.meta_objects["case_parameters"] = overrides
                    sampled = _sampled_parameters_of(mesh_dict)
                    if sampled:
                        slot.set_meta_entry("sampled_parameters", sampled)
                    self.slots.append(slot)
                    self.slot_queue.append(slot_idx)
                if len(self.slots) >= self.params.dataset_size:
                    break

        logger.info("Successfully loaded %d cases to mmap (rank %d)", len(self.slots), self.rank)
        if loss_mode == "data_driven":
            self._attach_labels_to_slots(self._labels_manifest())
            self._has_labels = True
        self._sync_params()
        logger.info("fvconfig: %s", self.params.fvconfig)
        return self.params

    def _resume_from_mmap(self, old_rank_dir: Path, old_snapshots_root: Path | None = None):
        """Reopen the slots of an earlier run: copy its shared mesh and ``phi.bin`` files,
        rebuild each case with the parameters it had drawn, and restore the newest snapshot."""
        from FVdomain.Load_mesh import H5CFDdataset

        old_rank_dir = Path(old_rank_dir)
        slot_idx = 0
        while (old_rank_dir / f"slot_{slot_idx}").exists():
            old_slot_dir = old_rank_dir / f"slot_{slot_idx}"
            new_slot_dir = self._rank_dir / f"slot_{slot_idx}"
            with open(old_slot_dir / "meta.json") as f:
                old_meta = json.load(f)
            h5_hash = old_meta["h5_hash"]

            current_mesh_dir = self._mesh_cache._cache_dir / h5_hash
            if not (current_mesh_dir / "ready.flag").exists():
                old_mesh_cache_dir = old_rank_dir.parent / "mesh_cache" / h5_hash
                if not (old_mesh_cache_dir / "ready.flag").exists():
                    raise RuntimeError(f"cannot resume: no static mesh {h5_hash} in the current or the old mesh cache")
                if self.rank == 0:
                    current_mesh_dir.mkdir(parents=True, exist_ok=True)
                    for fname in ["manifest.json", "static.bin"]:
                        src, dst = old_mesh_cache_dir / fname, current_mesh_dir / fname
                        if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
                            tmp = current_mesh_dir / f"{fname}.tmp.{os.getpid()}"
                            shutil.copy2(str(src), str(tmp))
                            os.replace(str(tmp), str(dst))
                    flag_tmp = current_mesh_dir / f"ready.flag.tmp.{os.getpid()}"
                    flag_tmp.touch()
                    os.replace(str(flag_tmp), str(current_mesh_dir / "ready.flag"))

            # the other ranks wait for rank 0's copy
            manifest_path, static_path = current_mesh_dir / "manifest.json", current_mesh_dir / "static.bin"
            for _ in range(3600):
                if (current_mesh_dir / "ready.flag").exists() and manifest_path.exists() and static_path.exists():
                    try:
                        if static_path.stat().st_size == json.load(open(manifest_path))["total_bytes"]:
                            break
                    except (json.JSONDecodeError, KeyError, OSError):
                        pass
                time.sleep(0.5)
            else:
                raise RuntimeError(f"timed out waiting for shared mesh {h5_hash} in {current_mesh_dir}")
            shared_mesh = SharedMeshStore.open(current_mesh_dir)
            self._mesh_cache._stores[h5_hash] = shared_mesh

            new_slot_dir.mkdir(parents=True, exist_ok=True)
            if old_slot_dir.resolve() != new_slot_dir.resolve():
                shutil.copy2(str(old_slot_dir / "phi.bin"), str(new_slot_dir / "phi.bin"))
                with open(new_slot_dir / "meta.json", "w") as f:
                    json.dump(dict(old_meta), f, indent=2)
            slot = CaseSlot.open(new_slot_dir, shared_mesh)

            # rebuild the case with the parameters it had; the drawn ones are released again
            # at its next reset
            temp_dataset = H5CFDdataset(params=self.params, case_list=[old_meta["h5_path"]])
            temp_dataset.set_pending_global_slot_index(global_slot_index(slot_idx, self.rank, self._world_size))
            overrides = dict(old_meta.get("case_parameters") or {})
            overrides.update(old_meta.get("sampled_parameters") or {})
            if overrides:
                temp_dataset.set_pending_overrides(overrides)
            mesh_dict, _ = temp_dataset[0]
            original = (mesh_dict.get("fvconfig") or {}).get("parameter") or {}
            unpin = {k: original[k] for k in (old_meta.get("sampled_parameters") or {})
                     if k in original and k not in (old_meta.get("case_parameters") or {})}
            if unpin:
                mesh_dict["parametric_manager"].register_from_toml(unpin)
            slot.meta_objects = _strip_static_tensors(mesh_dict)
            for key in ("case_name", "case_parameters"):
                if old_meta.get(key) is not None:
                    slot.meta_objects[key] = old_meta[key]
            self.slots.append(slot)
            self.slot_queue.append(slot_idx)
            slot_idx += 1

        if not self.slots:
            raise RuntimeError(f"No slots found under {old_rank_dir}")
        logger.info("Resumed %d slots from mmap (rank %d)", len(self.slots), self.rank)
        for snap_root in [old_snapshots_root, self._snapshots_root]:
            if snap_root is not None and self._try_restore_snapshot_from(snap_root):
                break
        self._sync_params()
        if self.loss_mode == "data_driven":
            self._attach_labels_to_slots(self._labels_manifest())
            self._has_labels = True
        return self.params

    def _try_restore_snapshot_from(self, snap_root) -> bool:
        """Load the newest complete snapshot under ``snap_root`` into the slots."""
        snap_root = Path(snap_root)
        if not snap_root.exists():
            return False
        rank_key = f"rank_{self.rank}"
        for snap_dir in sorted([d for d in snap_root.iterdir() if d.is_dir()],
                               key=lambda d: d.stat().st_mtime, reverse=True):
            meta_file = snap_dir / f"snapshot_meta_rank_{self.rank}.json"
            rank_dir = snap_dir / rank_key
            if not (meta_file.exists() and rank_dir.exists()):
                continue
            rank_meta = json.loads(meta_file.read_text()).get(rank_key, {})
            arrays = []
            for i, slot in enumerate(self.slots):
                phi_file = rank_dir / f"slot_{i}_phi.bin"
                phi_np = np.fromfile(str(phi_file), dtype=np.float32) if phi_file.exists() else None
                if phi_np is None or phi_np.size != slot._meta["num_cpd_cells"] * slot._meta["phi_channels"]:
                    break
                arrays.append(phi_np.reshape(slot._meta["num_cpd_cells"], slot._meta["phi_channels"]))
            else:
                for i, (slot, phi_np) in enumerate(zip(self.slots, arrays)):
                    slot._phi_mmap[:] = phi_np
                    slot._phi_mmap.flush()
                    if f"slot_{i}" in rank_meta:
                        slot._meta["time_step"] = rank_meta[f"slot_{i}"]["time_step"]
                        _atomic_json_write(slot._slot_dir / "meta.json", slot._meta)
                logger.info("Restored phi from snapshot: %s", snap_dir)
                return True
        return False

    def _set_reset_env_flag(self, flag=False, rst_time=1):
        self.reset_env_flag = flag
        self.rst_time = rst_time

    def refresh_pool(self, phi_new_cpd_cell, slot_idx):
        """Write back the new state of the batched slots and advance their time step."""
        if slot_idx.dim() == 0:
            slot_idx = slot_idx.unsqueeze(0)
        if len(slot_idx.unique()) == 1:
            idx = slot_idx.unique().item()
            self.slots[idx].update_phi(phi_new_cpd_cell)
            self.slots[idx].increment_time_step()
            self._auto_reset_if_past_labels(idx)
            return
        offset = 0
        for s_idx in slot_idx.tolist():
            n_cpd = self.slots[s_idx]._meta["num_cpd_cells"]
            self.slots[s_idx].update_phi(phi_new_cpd_cell[offset:offset + n_cpd])
            self.slots[s_idx].increment_time_step()
            self._auto_reset_if_past_labels(s_idx)
            offset += n_cpd

    def _auto_reset_if_past_labels(self, slot_idx):
        """Data loss: restart a slot from its initial state when its labels run out."""
        slot = self.slots[slot_idx]
        if self.loss_mode == "data_driven" and slot.has_labels and slot.time_step >= slot.n_timesteps:
            slot.reset_phi(slot.meta_objects["init_phi_cpd_cell"].to(torch.float32))

    def _update_re_from_patch_velocity(self, mesh, phi_cpd_cell):
        """With ``Re = {vel = "boundaryfield.<patch>"}``, recompute Re from the mean normal
        velocity on that patch (used in output names)."""
        theta_pde = mesh.get("fvconfig", {}).get("theta_pde", {})
        re_config = theta_pde.get("Re_config", None)
        if not isinstance(re_config, dict) or "vel" not in re_config:
            return
        vel_spec = re_config["vel"]
        patch_name = vel_spec.split(".", 1)[1] if vel_spec.startswith("boundaryfield.") else vel_spec
        patch_info = mesh["patch_dict"][patch_name]
        s_f, e_f = int(patch_info["start_idx_face"]), int(patch_info["end_idx_face"])
        s_c, e_c = int(patch_info["start_idx_cpd_cell"]), int(patch_info["end_idx_cpd_cell"])
        patch_face_area = mesh["face|face_area"][s_f:e_f]
        if patch_face_area.dim() == 1:
            patch_face_area = patch_face_area.unsqueeze(-1)
        v_dot_n = (phi_cpd_cell[s_c:e_c, 0:3] * mesh["face|face_normal"][s_f:e_f]).sum(dim=1, keepdim=True)
        flux_sum = (v_dot_n * patch_face_area).sum().item()
        total_area = patch_face_area.sum().item()
        velocity_magnitude = abs(flux_sum / total_area) if total_area > 0 else 0.0
        Re = theta_pde["rho"]["val"] * velocity_magnitude * theta_pde["L"] / theta_pde["mu"]["val"]
        mesh["fvconfig"]["theta_pde"]["Re"] = Re
        logger.info(f"Updated Re from patch '{patch_name}': Re = {Re:.2f} (V_avg={velocity_magnitude:.4f})")

    def _export_mesh_visualization(self, mesh, phi_cpd_cell, mask_interior_cell):
        """Write the state of a case leaving the pool to
        ``training_results/<50-block>/NO.<k>_<case>_Re=<Re>_dt=<dt>/...vtm``."""
        fields, col = {}, 0
        for var in mesh["boundary_info"]:
            fields[var] = phi_cpd_cell[:, col:col + VAR_CONFIG[var]]
            col += VAR_CONFIG[var]
        block = self.plot_count // 50 * 50
        name = (f"NO.{self.plot_count}_{mesh['case_name']}_Re={mesh['fvconfig']['theta_pde']['Re']:.4f}"
                f"_dt={mesh['fvconfig']['theta_pde']['dt']['val']:.3f}")
        export_vtm(os.path.join(self.state_save_dir, "training_results", f"{block}-{block + 50}", name, f"{name}.vtm"),
                   mesh, mask_interior_cell, fields)

    def reset_env(self, plot=False):
        """Reset the oldest slot to a newly drawn case (optionally exporting its state first)
        and move it to the back of the queue."""
        if not self.slot_queue:
            return
        oldest_idx = self.slot_queue.popleft()
        slot = self.slots[oldest_idx]
        logger.info("[reset_env] rank=%d, slot=%d, plot=%s, plot_count=%d", self.rank, oldest_idx, plot, self.plot_count)
        merged_mesh = _build_merged_mesh(slot)

        if plot:
            plot_phi_cpd_cell = slot.phi_tensor.clone()
            mask_interior_cell = (slot.get_mesh_tensor("cpd|cell_type").clone().long() == FaceType.NORMAL).squeeze()
            self._update_re_from_patch_velocity(merged_mesh, plot_phi_cpd_cell)
            self._export_mesh_visualization(merged_mesh, plot_phi_cpd_cell, mask_interior_cell)
            self.plot_count += 1
            self._plot_env = False

        if self.loss_mode == "data_driven":
            slot.reset_phi(slot.meta_objects["init_phi_cpd_cell"].to(torch.float32))
        else:
            new_mesh, init_phi_cpd_cell = merged_mesh["transform"](merged_mesh, self.params)
            slot.reset_phi(init_phi_cpd_cell)
            slot.meta_objects = _strip_static_tensors(new_mesh)
            sampled = _sampled_parameters_of(new_mesh)
            if sampled:
                slot.set_meta_entry("sampled_parameters", sampled)
        self.slot_queue.append(oldest_idx)

    def save_snapshot(self, checkpoint_idx, epoch):
        """Write the slots' states asynchronously, beside a model checkpoint."""
        self._snapshot_writer.snapshot(
            self.slots, self._snapshots_root / f"epoch_{epoch}", self.rank,
            {"epoch": epoch, "checkpoint_idx": checkpoint_idx})
        if self.rank == 0:
            self._snapshot_writer.cleanup(self._snapshots_root, getattr(self.params, "max_snapshots", 3))

    def close(self):
        """Wait for pending snapshot writes."""
        if self._snapshot_writer is not None:
            self._snapshot_writer.wait()
