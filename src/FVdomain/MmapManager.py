"""Memory-mapped storage of the training pool: shared static meshes, per-case state files
and asynchronous snapshots."""

import json
import hashlib
import logging
import os
import shutil
import threading
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

STATIC_TENSOR_KEYS = [
    # Geometry
    "node|node_pos", "face|face_pos", "face|face_area", "face|face_normal",
    "face|face_type", "cell|cell_pos", "cell|cell_volume", "cpd|cell_pos",
    "cpd|cell_type",
    # Connectivity (training-critical)
    "cpd|neighbor_cell", "cpd|neighbor_cell_x", "cyclic_face",
    "cells_node", "cells_node_ptr", "cells_face", "cells_face_ptr",
    "cells_face_normal", "face_node", "face_node_ptr",
    # WLSQ
    "A_cell_to_cell", "single_B_cell_to_cell",
    # GNN CSR
    "gnn|edge_counts", "gnn|edge_indices", "gnn|edge_sort_idx",
    # H5-loaded static tensors used only for VTK export / auxiliary paths.
    # They are never mutated after load and must be shared, not per-slot.
    "cells_face_area",
    "cpd|neighbor_cell_non_cyclic",
    "edge_node",
    "face_edge", "face_edge_ptr",
    "pv_cells_node", "pv_cells_type",
]

_TORCH_TO_NP_DTYPE = {
    torch.float32: np.float32, torch.float64: np.float64,
    torch.int32: np.int32, torch.int64: np.int64,
}

_STR_TO_NP_DTYPE = {
    "float32": np.float32, "float64": np.float64,
    "int32": np.int32, "int64": np.int64,
}


class SharedMeshStore:
    """Read-only mmap store for static mesh tensors packed into a single binary."""

    def __init__(self):
        self._mmap: Optional[np.memmap] = None
        self.manifest: Optional[dict] = None
        self.mesh_dir: Optional[Path] = None

    @classmethod
    def create(cls, mesh_dir: Path, mesh_dict: dict, static_keys: list) -> "SharedMeshStore":
        mesh_dir = Path(mesh_dir)
        mesh_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"tensors": {}}
        offset = 0
        layout = []
        for key in static_keys:
            if key not in mesh_dict or not torch.is_tensor(mesh_dict[key]):
                continue
            tensor = mesh_dict[key]
            nbytes = tensor.numel() * tensor.element_size()
            aligned_offset = (offset + 63) & ~63
            manifest["tensors"][key] = {
                "offset": aligned_offset,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "nbytes": nbytes,
            }
            layout.append((key, aligned_offset, nbytes, tensor))
            offset = aligned_offset + nbytes
        total_bytes = (offset + 63) & ~63
        manifest["total_bytes"] = total_bytes
        bin_path = mesh_dir / "static.bin"
        mm = np.memmap(str(bin_path), dtype="uint8", mode="w+", shape=(total_bytes,))
        for key, off, nbytes, tensor in layout:
            data = tensor.contiguous().numpy()
            flat = np.frombuffer(data.tobytes(), dtype="uint8")
            mm[off: off + nbytes] = flat
        mm.flush()
        del mm
        with open(mesh_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        (mesh_dir / "ready.flag").touch()
        store = cls()
        store.mesh_dir = mesh_dir
        store.manifest = manifest
        store._mmap = np.memmap(str(bin_path), dtype="uint8", mode="r", shape=(total_bytes,))
        return store

    @classmethod
    def open(cls, mesh_dir: Path) -> "SharedMeshStore":
        mesh_dir = Path(mesh_dir)
        with open(mesh_dir / "manifest.json") as f:
            manifest = json.load(f)
        store = cls()
        store.mesh_dir = mesh_dir
        store.manifest = manifest
        store._mmap = np.memmap(
            str(mesh_dir / "static.bin"), dtype="uint8", mode="r",
            shape=(manifest["total_bytes"],),
        )
        return store

    def get_tensor(self, key: str) -> torch.Tensor:
        info = self.manifest["tensors"][key]
        np_dtype = _STR_TO_NP_DTYPE[info["dtype"]]
        arr = np.ndarray(shape=info["shape"], dtype=np_dtype, buffer=self._mmap, offset=info["offset"])
        return torch.from_numpy(arr)

    def has_tensor(self, key: str) -> bool:
        return key in self.manifest["tensors"]



def _atomic_json_write(path: Path, data: dict):
    path = Path(path)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(str(tmp_path), str(path))


class CaseSlot:
    """Per-case read/write mmap for evolving phi state."""

    def __init__(self):
        self._phi_mmap: Optional[np.memmap] = None
        self._meta: Optional[dict] = None
        self._slot_dir: Optional[Path] = None
        self._shared_mesh: Optional[SharedMeshStore] = None
        self.meta_objects: dict = {}
        # Labels memmap (read-only, used for data-driven training).
        self._labels_mmap: Optional[np.memmap] = None
        self._labels_shape: Optional[Tuple[int, int, int]] = None

    @classmethod
    def create(
        cls,
        slot_dir: Path,
        shared_mesh: SharedMeshStore,
        num_cpd_cells: int,
        phi_channels: int,
        init_phi: torch.Tensor,
        meta_info: dict,
    ) -> "CaseSlot":
        slot_dir = Path(slot_dir)
        slot_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "h5_hash": meta_info["h5_hash"],
            "h5_path": meta_info["h5_path"],
            "case_name": meta_info["case_name"],
            "case_parameters": meta_info.get("case_parameters"),
            "time_step": 0,
            "num_cpd_cells": num_cpd_cells,
            "phi_channels": phi_channels,
        }
        _atomic_json_write(slot_dir / "meta.json", meta)
        phi_bin = slot_dir / "phi.bin"
        mm = np.memmap(
            str(phi_bin), dtype=np.float32, mode="w+",
            shape=(num_cpd_cells, phi_channels),
        )
        phi_np = init_phi.contiguous().numpy().astype(np.float32)
        mm[:] = phi_np
        mm.flush()
        obj = cls()
        obj._slot_dir = slot_dir
        obj._shared_mesh = shared_mesh
        obj._meta = meta
        obj._phi_mmap = mm
        return obj

    @classmethod
    def open(cls, slot_dir: Path, shared_mesh: SharedMeshStore) -> "CaseSlot":
        slot_dir = Path(slot_dir)
        with open(slot_dir / "meta.json") as f:
            meta = json.load(f)
        phi_bin = slot_dir / "phi.bin"
        mm = np.memmap(
            str(phi_bin), dtype=np.float32, mode="r+",
            shape=(meta["num_cpd_cells"], meta["phi_channels"]),
        )
        obj = cls()
        obj._slot_dir = slot_dir
        obj._shared_mesh = shared_mesh
        obj._meta = meta
        obj._phi_mmap = mm
        return obj

    @property
    def phi_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self._phi_mmap)

    @property
    def time_step(self) -> int:
        return self._meta["time_step"]

    def update_phi(self, new_phi: torch.Tensor):
        """Write new phi values to the mmap in-place."""
        phi_np = new_phi.contiguous().numpy().astype(np.float32)
        self._phi_mmap[:] = phi_np
        self._phi_mmap.flush()

    def increment_time_step(self):
        """Atomically increment time_step in meta.json."""
        self._meta["time_step"] += 1
        _atomic_json_write(self._slot_dir / "meta.json", self._meta)

    def set_meta_entry(self, key, value):
        """Persist one extra meta.json entry (e.g. the slot's SAMPLED boundary parameters, so a
        resume can re-impose them instead of redrawing).
        """
        self._meta[key] = value
        _atomic_json_write(self._slot_dir / "meta.json", self._meta)

    def reset_phi(self, init_phi: torch.Tensor):
        """Overwrite phi and reset time_step to 0."""
        phi_np = init_phi.contiguous().numpy().astype(np.float32)
        self._phi_mmap[:] = phi_np
        self._phi_mmap.flush()
        self._meta["time_step"] = 0
        _atomic_json_write(self._slot_dir / "meta.json", self._meta)


    def get_mesh_tensor(self, key: str) -> torch.Tensor:
        """Delegate to the shared mesh store."""
        return self._shared_mesh.get_tensor(key)


    def attach_labels(
        self,
        labels_file,
        shape: Tuple[int, int, int],
        dtype: str = "float32",
    ) -> None:
        """Attach a read-only labels memmap to this slot."""
        self._labels_mmap = np.memmap(
            str(labels_file), dtype=dtype, mode="r", shape=shape
        )
        self._labels_shape = tuple(shape)

    def get_label_at(self, t_idx: int) -> np.ndarray:
        """Return a zero-copy memmap view of the label at timestep t_idx."""
        return self._labels_mmap[t_idx]

    @property
    def has_labels(self) -> bool:
        """True if a labels memmap has been attached."""
        return self._labels_mmap is not None

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps in the attached labels memmap; 0 if none."""
        if self._labels_shape is None:
            return 0
        return self._labels_shape[0]


class MeshCacheManager:
    """Cross-rank safe mesh cache."""

    def __init__(self, cache_dir: Path):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._stores: Dict[str, SharedMeshStore] = {}

    def _hash_h5(self, h5_path: str) -> str:
        """MD5 of (file_size_bytes + first 4096 bytes), return 8 hex chars."""
        p = Path(h5_path)
        file_size = p.stat().st_size
        with open(h5_path, "rb") as f:
            header = f.read(4096)
        md5 = hashlib.md5()
        md5.update(str(file_size).encode())
        md5.update(header)
        return md5.hexdigest()[:8]

    def _acquire_creation_lock(self, mesh_dir: Path) -> bool:
        """Try to create .creating.lock exclusively. Returns True on success."""
        lock_path = mesh_dir / ".creating.lock"
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            return False

    def _wait_for_ready(self, mesh_dir: Path, timeout: float = 120.0):
        """Poll for ready.flag, raising TimeoutError if it doesn't appear."""
        deadline = _time.time() + timeout
        while _time.time() < deadline:
            if (mesh_dir / "ready.flag").exists():
                return
            _time.sleep(0.1)
        raise TimeoutError(
            f"SharedMeshStore at {mesh_dir} not ready after {timeout}s"
        )

    def get_or_create(self, h5_path: str, mesh_dict: dict) -> SharedMeshStore:
        """Return (or create) the SharedMeshStore for the given H5 file."""
        h5_hash = self._hash_h5(h5_path)
        if h5_hash in self._stores:
            return self._stores[h5_hash]

        mesh_dir = self._cache_dir / h5_hash
        mesh_dir.mkdir(parents=True, exist_ok=True)

        # If ready.flag already exists, just open
        if (mesh_dir / "ready.flag").exists():
            store = SharedMeshStore.open(mesh_dir)
            self._stores[h5_hash] = store
            return store

        # Stale lock detection: remove locks older than 60s
        lock_path = mesh_dir / ".creating.lock"
        if lock_path.exists():
            lock_age = _time.time() - lock_path.stat().st_mtime
            if lock_age > 60.0:
                logger.warning("Removing stale lock at %s (age %.1fs)", lock_path, lock_age)
                lock_path.unlink(missing_ok=True)

        # Try to become the creator
        if self._acquire_creation_lock(mesh_dir):
            try:
                store = SharedMeshStore.create(mesh_dir, mesh_dict, STATIC_TENSOR_KEYS)
            finally:
                lock_path.unlink(missing_ok=True)
            self._stores[h5_hash] = store
            return store
        else:
            # Another process is creating it — wait for ready.flag
            self._wait_for_ready(mesh_dir)
            store = SharedMeshStore.open(mesh_dir)
            self._stores[h5_hash] = store
            return store


class AsyncSnapshotWriter:
    """Background phi snapshot for checkpoint consistency."""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None

    def snapshot(
        self,
        slots: List["CaseSlot"],
        snapshot_dir: Path,
        rank: int,
        extra_meta: dict,
    ):
        """Clone phi data now (main thread), write to disk in background."""
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        self._thread = None

        snapshot_dir = Path(snapshot_dir)
        # Clone all phi data immediately in the calling thread
        phi_copies = [slot.phi_tensor.clone() for slot in slots]
        time_steps = [slot.time_step for slot in slots]

        def _write():
            rank_dir = snapshot_dir / f"rank_{rank}"
            rank_dir.mkdir(parents=True, exist_ok=True)
            for i, phi in enumerate(phi_copies):
                phi_file = rank_dir / f"slot_{i}_phi.bin"
                phi.numpy().astype(np.float32).tofile(str(phi_file))

            rank_meta = {
                f"slot_{i}": {"time_step": ts}
                for i, ts in enumerate(time_steps)
            }
            meta = dict(extra_meta)
            meta[f"rank_{rank}"] = rank_meta
            _atomic_json_write(
                snapshot_dir / f"snapshot_meta_rank_{rank}.json", meta
            )

        self._thread = threading.Thread(target=_write, daemon=True)
        self._thread.start()

    def wait(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def cleanup(self, snapshots_root: Path, max_keep: int):
        """Delete oldest snapshot dirs, keeping the newest max_keep."""
        snapshots_root = Path(snapshots_root)
        dirs = [d for d in snapshots_root.iterdir() if d.is_dir()]
        if len(dirs) <= max_keep:
            return
        # Sort by mtime ascending — oldest first
        dirs.sort(key=lambda d: d.stat().st_mtime)
        to_delete = dirs[:len(dirs) - max_keep]
        for d in to_delete:
            shutil.rmtree(str(d), ignore_errors=True)
            logger.debug("Removed old snapshot dir: %s", d)
