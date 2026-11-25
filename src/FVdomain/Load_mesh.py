"""Dataset and mesh loader for H5 cases, boundary conditions, WLSQ moments, and GNN CSR tables."""

import copy
import importlib
import importlib.util
import logging
import os
from typing import Optional

import h5py
import numpy as np
import tomlkit
import torch
from torch.utils.data import Dataset

from FVdomain.Graph_pool import case_seed_for_global_slot
from FVdomain.SetBC.FvPatch import fvPatch
from FVdomain.SetBC.onthefly.BCbase import BCType, SOLVER_VARS, VAR_CONFIG, VARType
from FVdomain.SetBC.onthefly.Parafunction import Functionbase, ParametricManager
from FVdomain.SetBC.onthefly.setCyclic import prepare_cyclic_boundary_geometry
from FVsolver.FVdiscretization.FVgrad import compute_normal_matrix
from Utils.toml_utils import toml_to_native
from Utils.utilities import FaceType

logger = logging.getLogger(__name__)


def validate_patch_table(
    patch_dict: dict,
    num_cells: int,
    num_faces: int,
    num_cpd_cells: int,
    patch_table_path: str = "<patchinfo.toml>",
    h5_path: str = "<mesh.h5>",
) -> None:
    """Verify ``patchinfo.toml`` counts and boundary spans match the loaded H5 mesh."""
    num_boundary_faces = num_cpd_cells - num_cells
    num_internal_faces = num_faces - num_boundary_faces
    problems: list[str] = []

    meta = patch_dict.get("metadata", {}) or {}
    for key, mesh_val in (
        ("num_cells", num_cells),
        ("num_faces", num_faces),
        ("num_internal_faces", num_internal_faces),
    ):
        if key in meta and int(meta[key]) != int(mesh_val):
            problems.append(f"[metadata] {key}={int(meta[key])}, mesh has {mesh_val}")

    patches = patch_dict.get("patch", {}) or {}
    if not patches:
        problems.append("no [patch.*] sections found")

    spans_face, spans_cpd = [], []
    for name, patch in patches.items():
        try:
            fs, fe = int(patch["start_idx_face"]), int(patch["end_idx_face"])
            cs, ce = int(patch["start_idx_cpd_cell"]), int(patch["end_idx_cpd_cell"])
            n_faces_patch = int(patch["n_faces"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"patch '{name}' is missing face/cpd_cell index ranges")
            continue
        if fe - fs != n_faces_patch:
            problems.append(f"patch '{name}': face span {fe - fs} != n_faces {n_faces_patch}")
        if ce - cs != n_faces_patch:
            problems.append(f"patch '{name}': ghost span {ce - cs} != n_faces {n_faces_patch}")
        spans_face.append((fs, fe, name))
        spans_cpd.append((cs, ce, name))

    for label, spans, lo, hi in (
        ("boundary face", spans_face, num_internal_faces, num_faces),
        ("ghost cell", spans_cpd, num_cells, num_cpd_cells),
    ):
        cursor = lo
        for start, end, name in sorted(spans):
            if start != cursor:
                problems.append(
                    f"{label} spans do not tile [{lo}, {hi}): patch '{name}' starts at {start}, expected {cursor}"
                )
                break
            cursor = end
        else:
            if cursor != hi:
                problems.append(f"{label} spans cover [{lo}, {cursor}) instead of [{lo}, {hi})")

    if problems:
        detail = "".join(f"  - {p}\n" for p in problems)
        raise ValueError(
            f"Patch table does not match mesh.\n"
            f"  table: {patch_table_path}\n"
            f"  mesh:  {h5_path}\n"
            f"{detail}"
        )


class CFDdatasetBase:
    def __init__(self):
        self.params = None

    def _get_bc_class(self, bc_type: str, var_name: str, sidecar_dir: str = None):
        """Resolve a BC class: ``<sidecar_dir>/bc/set<var>.py`` first, then built-in modules."""
        class_name = bc_type[0].upper() + bc_type[1:].lower()

        if sidecar_dir is not None:
            custom_bc_path = os.path.join(sidecar_dir, "bc", f"set{var_name}.py")
            if os.path.exists(custom_bc_path):
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"custom_bc_{var_name}_{os.path.basename(sidecar_dir)}",
                        custom_bc_path,
                    )
                    custom_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(custom_module)
                    bc_class = getattr(custom_module, class_name, None)
                    if bc_class is not None:
                        logger.info("Loaded custom BC '%s' from %s", class_name, custom_bc_path)
                        return bc_class
                except Exception as exc:
                    logger.warning("Failed to load custom BC from %s: %s", custom_bc_path, exc)

        var_to_module = {
            "U": "FVdomain.SetBC.onthefly.setU",
            "T": "FVdomain.SetBC.onthefly.setT",
            "p": "FVdomain.SetBC.onthefly.setP",
            "p_rgh": "FVdomain.SetBC.onthefly.setP_rgh",
        }
        module_name = var_to_module.get(var_name)
        if module_name is None:
            raise ValueError(f"Unknown variable {var_name!r}; supported: {list(var_to_module)}")
        module = importlib.import_module(module_name)
        bc_class = getattr(module, class_name, None)
        if bc_class is None:
            raise AttributeError(
                f"Boundary condition '{class_name}' not found in {module_name}. "
                f"Available: {[n for n in dir(module) if not n.startswith('_')]}"
            )
        return bc_class


class CFDmesh(CFDdatasetBase):
    """Transform a raw H5 mesh dict into an initialized case with BCs, WLSQ moments, and GNN CSR."""

    _case_counter = 0

    def __init__(self, params):
        super().__init__()
        self.params = params

    def init_env(self, mesh):
        patch_dict = mesh["patch_dict"]
        fvpatch = mesh["fvpatch"]
        fvconfig = mesh["fvconfig"]
        num_cpd_cell = mesh["cpd|cell_pos"].shape[0]
        parametric_manager = mesh.get("parametric_manager")

        if parametric_manager is not None and "boundary_info_raw" in mesh:
            parametric_manager.reset_all()
            boundary_info = copy.deepcopy(mesh["boundary_info_raw"])
        else:
            boundary_info = copy.deepcopy(mesh["boundary_info"])

        phi_cpd_cell = []
        bcpatch = {
            BCType.FIRST: [],
            BCType.MIXED: [],
            BCType.CYCLIC: [],
            BCType.COUPLED: [],
            BCType.EMPTY: [],
            "patch_idx_offset": 0,
        }
        current_col = 0

        init_face_area = mesh.get("face|face_area")
        if init_face_area is not None:
            init_face_area = init_face_area.to(torch.float32).view(-1, 1)
            init_face_type = mesh.get("face|face_type")
            init_node_pos = mesh.get("node|node_pos")
            if (
                init_face_type is not None
                and init_node_pos is not None
                and (init_face_type.view(-1) == FaceType.EMPTY).any()
            ):
                zt = torch.abs(init_node_pos[:, 2].max() - init_node_pos[:, 2].min())
                if zt > 1e-12:
                    init_face_area = init_face_area / zt
        init_unv_face = mesh.get("face|face_normal")
        if init_unv_face is not None:
            init_unv_face = init_unv_face.to(torch.float32).view(-1, 3)

        cpd_cell_pos = mesh.get("cpd|cell_pos")
        for var, bc_info in boundary_info.items():
            bcpatch[var] = []
            num_channels = VAR_CONFIG.get(var)
            if num_channels is None:
                raise ValueError(f"Unknown field {var!r}; supported: {list(VAR_CONFIG)}")

            internal_field_info = bc_info.get("internalField")
            if internal_field_info is None:
                raise ValueError(f"Missing internalField for variable {var!r}")
            resolved_internal = (
                parametric_manager.resolve_value(internal_field_info)
                if parametric_manager is not None
                else internal_field_info
            )
            var_cpd_cell = Functionbase(
                value_type=resolved_internal["value_type"],
                value=resolved_internal["val"],
                num_element=num_cpd_cell,
                channel_dim=num_channels,
            ).eval(device="cpu", cpd_cell_pos=cpd_cell_pos, fvconfig=fvconfig).clone()

            sidecar_dir = mesh.get("sidecar_dir")
            for patch_name, patch_bc in bc_info["boundaryfield"].items():
                bc_type = patch_bc.get("type")
                if bc_type is None:
                    raise ValueError(f"Missing BC type for patch {patch_name!r} in variable {var!r}")
                patch_idx = patch_dict[patch_name]["patch_idx"]
                bc_class = self._get_bc_class(bc_type, var, sidecar_dir=sidecar_dir)

                patch_boundary_info = {k: v for k, v in patch_bc.items() if k != "type"}
                patch_boundary_info["fvconfig"] = fvconfig
                patch_boundary_info["patch_dict"] = patch_dict

                bc_instance = bc_class(
                    patch_name=patch_name,
                    patch_idx=patch_idx,
                    boundary_info=patch_boundary_info,
                    parametric_manager=parametric_manager,
                )
                bcpatch[bc_instance.bcType].append(bc_instance)
                var_cpd_cell = bc_instance.init_field(
                    var_cpd_cell,
                    fvpatch,
                    neighbor_cpd_cell=mesh["cpd|neighbor_cell"],
                    fvconfig=fvconfig,
                    cpd_cell_pos=cpd_cell_pos,
                    face_area=init_face_area,
                    unv_face=init_unv_face,
                )
                bcpatch[var].append(bc_instance)

            phi_cpd_cell.append(var_cpd_cell)
            current_col += num_channels

        phi_cpd_cell = torch.cat(phi_cpd_cell, dim=-1)

        warm_phi_interior = mesh.get("warm|phi_interior")
        if warm_phi_interior is not None:
            mask_int = mesh["cpd|cell_type"].reshape(-1) == FaceType.NORMAL
            warm_t = warm_phi_interior.to(device=phi_cpd_cell.device, dtype=phi_cpd_cell.dtype)
            expected = (int(mask_int.sum()), phi_cpd_cell.shape[-1])
            if tuple(warm_t.shape) != expected:
                raise ValueError(f"warm|phi_interior shape {tuple(warm_t.shape)} != expected {expected}")
            phi_cpd_cell = phi_cpd_cell.clone()
            phi_cpd_cell[mask_int] = warm_t

        mesh["bcpatch"] = bcpatch
        mesh["phi_num_channels"] = current_col
        mesh["num_channels"] = current_col * 2 + len(FaceType)
        mesh["init_phi_cpd_cell"] = phi_cpd_cell.clone()
        return self._auto_detect_pRef_need(mesh, bcpatch, fvconfig), phi_cpd_cell

    def _auto_detect_pRef_need(self, mesh, bcpatch, fvconfig):
        """Enable mean-pressure gauge constraint only when pressure has no Dirichlet patch."""
        mesh["fvconfig"]["fvSolution"]["pRefCell_valid_mask"] = False
        solver_type = str(fvconfig["solver"]["type"]).lower()
        if solver_type not in SOLVER_VARS:
            return mesh

        has_dirichlet_p = any(
            bc.varType in (VARType.PRESSURE, VARType.PRESSURE_RGH)
            for bc in bcpatch.get(BCType.FIRST, [])
        )
        if not has_dirichlet_p and mesh["fvconfig"]["fvSolution"].get("pRefCell", -1) >= 0:
            mesh["fvconfig"]["fvSolution"]["pRefCell_valid_mask"] = True
        return mesh

    def _resolve_param_value(self, mesh, value, default=None, key=None):
        if value is None:
            return default
        pm = mesh.get("parametric_manager")
        if isinstance(value, str):
            if pm is not None:
                return pm.resolve_value(value)
            raise ValueError(f"theta_pde reference {value!r} requires a parametric_manager")
        if isinstance(value, dict):
            vt = value.get("value_type", "constant")
            if vt in ("constant", "uniform"):
                return value.get("val", default)
            if pm is not None:
                synth = f"__theta_pde_{key}" if key else f"__theta_pde_{id(value)}"
                if not pm.has_parameter(synth):
                    pm.register_from_toml({synth: value})
                return pm.get_value(synth)
            raise ValueError(f"theta_pde entry with value_type={vt!r} requires a parametric_manager")
        return value

    @staticmethod
    def _estimate_inlet_velocity(mesh):
        re_cfg = (mesh.get("fvconfig", {}).get("theta_pde", {}) or {}).get("Re_config")
        init_phi = mesh.get("init_phi_cpd_cell")
        patch_dict = mesh.get("patch_dict")
        if (
            isinstance(re_cfg, dict)
            and "vel" in re_cfg
            and init_phi is not None
            and isinstance(patch_dict, dict)
            and init_phi.shape[-1] >= 3
        ):
            vel_spec = str(re_cfg["vel"])
            pname = vel_spec.split(".", 1)[1] if vel_spec.startswith("boundaryfield.") else vel_spec
            pinfo = patch_dict.get(pname)
            if pinfo is not None:
                sc, ec = int(pinfo["start_idx_cpd_cell"]), int(pinfo["end_idx_cpd_cell"])
                sf, ef = int(pinfo["start_idx_face"]), int(pinfo["end_idx_face"])
                u_patch = init_phi[sc:ec, 0:3].to(torch.float32)
                f_area = mesh.get("face|face_area")
                f_norm = mesh.get("face|face_normal")
                if f_area is not None and f_norm is not None and ec > sc:
                    pa = f_area[sf:ef].to(torch.float32).reshape(-1, 1)
                    pn = f_norm[sf:ef].to(torch.float32).reshape(-1, 3)
                    tot_a = float(pa.sum().item())
                    if tot_a > 0.0:
                        v_avg = abs(float(((u_patch * pn).sum(dim=1, keepdim=True) * pa).sum().item()) / tot_a)
                        if v_avg > 0.0:
                            return v_avg
                elif ec > sc:
                    v_avg = float(torch.norm(u_patch, dim=1).mean().item())
                    if v_avg > 0.0:
                        return v_avg

        pm = mesh.get("parametric_manager")
        if pm is not None:
            try:
                v = pm.get_value("inlet_velocity")
                if isinstance(v, (int, float)):
                    return abs(float(v))
                if isinstance(v, (list, tuple)):
                    return float(sum(x**2 for x in v) ** 0.5)
            except (KeyError, ValueError):
                pass

        u_bc = mesh.get("boundary_info", {}).get("U", {})
        for patch_cfg in u_bc.values():
            if not isinstance(patch_cfg, dict) or patch_cfg.get("type") != "fixedValue":
                continue
            val = patch_cfg.get("value", {})
            val = val.get("val") if isinstance(val, dict) else val
            if isinstance(val, (list, tuple)):
                mag = float(sum(x**2 for x in val if isinstance(x, (int, float))) ** 0.5)
                if mag > 0:
                    return mag
            elif isinstance(val, (int, float)) and abs(val) > 0:
                return abs(float(val))
        return 1.0

    def set_theta_PDE(self, mesh):
        fvconfig = mesh["fvconfig"]
        solver_type = str(fvconfig["solver"]["type"]).lower()

        theta_pde_raw = fvconfig.get("theta_pde_raw")
        if theta_pde_raw is None:
            theta_pde_raw = copy.deepcopy(fvconfig["theta_pde"])
            for k in ("Re", "Re_config", "Ra", "_resolved_mu", "_resolved_rho"):
                theta_pde_raw.pop(k, None)
            fvconfig["theta_pde_raw"] = theta_pde_raw
        theta_pde = fvconfig["theta_pde"]
        theta_pde.clear()
        theta_pde.update(copy.deepcopy(theta_pde_raw))

        rho = self._resolve_param_value(mesh, theta_pde.get("rho"), default=1.0, key="rho")
        mu = self._resolve_param_value(mesh, theta_pde.get("mu"), default=1e-3, key="mu")
        dt = self._resolve_param_value(mesh, theta_pde.get("dt"), default=0.01, key="dt")
        L = self._resolve_param_value(mesh, theta_pde.get("L"), default=1.0, key="L")
        source = self._resolve_param_value(mesh, theta_pde.get("source"), default=0.0, key="source")
        theta_pde["_resolved_mu"] = mu
        theta_pde["_resolved_rho"] = rho

        re_config = theta_pde.get("Re")
        if isinstance(re_config, dict) and "vel" in re_config:
            theta_pde["Re_config"] = re_config
            u_char = self._estimate_inlet_velocity(mesh)
            Re = rho * u_char * L / mu if mu != 0 else float("inf")
        elif isinstance(re_config, (int, float)):
            Re = re_config
        else:
            u_char = self._estimate_inlet_velocity(mesh)
            Re = rho * u_char * L / mu if mu != 0 else float("inf")
        theta_pde["Re"] = Re

        unsteady_coeff = 1.0 if theta_pde["unsteady"] else 0.0
        nu_coeff = mu / rho if rho != 0 else mu
        source_term = source / rho if rho != 0 else source

        if solver_type in ("picoext", "picocmpt"):
            grad_p_coeff = 1.0 / rho if rho != 0 else 1.0
            theta_PDE = torch.tensor(
                [unsteady_coeff, 1.0, 1.0, grad_p_coeff, nu_coeff, source_term, dt],
                dtype=torch.float32,
            ).view(1, -1)
        elif solver_type in ("buoyantboussinesqpicoext", "buoyantboussinesqpicocmpt"):
            beta = self._resolve_param_value(mesh, theta_pde.get("beta"), default=0.0, key="beta")
            TRef = self._resolve_param_value(mesh, theta_pde.get("TRef"), default=300.0, key="TRef")
            Pr = self._resolve_param_value(mesh, theta_pde.get("Pr"), default=0.71, key="Pr")
            Cp = self._resolve_param_value(mesh, theta_pde.get("Cp"), default=1005.0, key="Cp")
            g = self._resolve_param_value(mesh, theta_pde.get("g"), default=[0.0, 0.0, -9.81], key="g")
            heat_source = self._resolve_param_value(mesh, theta_pde.get("heat_source"), default=0.0, key="heat_source")
            alpha_cfg = self._resolve_param_value(mesh, theta_pde.get("alpha"), default=None, key="alpha")
            hRef = self._resolve_param_value(mesh, theta_pde.get("hRef"), default=0.0, key="hRef")

            alpha = (nu_coeff / Pr if Pr != 0 else 0.0) if alpha_cfg is None else alpha_cfg
            g_mag = (g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5
            theta_pde["Ra"] = g_mag * beta * L**3 / (nu_coeff * alpha) if beta != 0 and alpha != 0 else 0.0
            theta_pde["Pr"] = Pr
            heat_source_term = heat_source / (rho * Cp) if rho != 0 else heat_source

            theta_PDE = torch.tensor(
                [
                    unsteady_coeff, 1.0, 1.0, 1.0, nu_coeff, source_term, dt,
                    alpha, heat_source_term, TRef, beta, g[0], g[1], g[2], hRef,
                ],
                dtype=torch.float32,
            ).view(1, -1)
        else:
            raise NotImplementedError(f"Unsupported solver type {solver_type!r}")

        mesh["num_channels"] = mesh.get("num_channels", 0) + theta_PDE.shape[1]
        mesh["theta_PDE"] = theta_PDE
        return mesh

    @staticmethod
    def _update_gnn_csr_format(mesh, neighbor_cpd_cell, num_cpd_cells):
        num_edges = neighbor_cpd_cell.shape[1]
        twoway = torch.cat([neighbor_cpd_cell, neighbor_cpd_cell.flip(0)], dim=1)
        sources, receivers = twoway[0], twoway[1]
        sort_idx_raw = receivers.argsort(stable=True)
        sorted_sources = sources[sort_idx_raw]
        sorted_receivers = receivers[sort_idx_raw]
        is_fwd = sort_idx_raw < num_edges
        sort_idx = torch.where(is_fwd, 2 * sort_idx_raw, 2 * (sort_idx_raw - num_edges) + 1)
        mesh["gnn|edge_counts"] = torch.bincount(sorted_receivers, minlength=num_cpd_cells)
        mesh["gnn|edge_indices"] = sorted_sources
        mesh["gnn|edge_sort_idx"] = sort_idx
        return mesh

    @staticmethod
    def calc_WLSQ_A_B_normal_matrix(mesh):
        if "A_cell_to_cell" in mesh:
            return mesh
        cpd_cell_pos = mesh["cpd|cell_pos"]
        neighbor_cpd_cell = mesh["cpd|neighbor_cell"].long()
        neighbor_cpd_cell_x = mesh["cpd|neighbor_cell_x"].long()
        cell_type = mesh["cpd|cell_type"].long().squeeze()

        khops = int((mesh.get("fvconfig") or {}).get("stencil", {}).get("khops", 1))
        if khops == 1:
            neighbor_cpd_cell_x = torch.unique(neighbor_cpd_cell.sort(0)[0], dim=1)

        mask_empty = cell_type == FaceType.EMPTY
        keep = ~(mask_empty[neighbor_cpd_cell_x[0]] | mask_empty[neighbor_cpd_cell_x[1]])
        neighbor_cpd_cell_x = neighbor_cpd_cell_x[:, keep]
        mesh["cpd|neighbor_cell_x"] = neighbor_cpd_cell_x

        A_mat, two_way_B = compute_normal_matrix(cpd_cell_pos, neighbor_cpd_cell_x)
        mesh["A_cell_to_cell"] = A_mat.to(torch.float32)
        mesh["single_B_cell_to_cell"] = torch.chunk(two_way_B, 2, dim=0)[0].to(torch.float32)
        return mesh

    def __call__(self, mesh, params=None):
        mesh = prepare_cyclic_boundary_geometry(mesh)
        mesh, phi_cpd_cell = self.init_env(mesh)

        fourier_num_freqs = int(getattr(params, "fourier_num_freqs", 0)) if params else 0
        mesh["num_channels"] += 6 * fourier_num_freqs if fourier_num_freqs > 0 else 0
        mesh["fourier_num_freqs"] = fourier_num_freqs

        mesh = self.set_theta_PDE(mesh)
        mesh = self.calc_WLSQ_A_B_normal_matrix(mesh)
        mesh = self._update_gnn_csr_format(mesh, mesh["cpd|neighbor_cell"], mesh["cpd|cell_pos"].shape[0])
        return mesh, phi_cpd_cell


class H5CFDdataset(Dataset):
    def __init__(self, params, case_list):
        super().__init__()
        self.case_list = case_list
        self.params = params
        self._pending_param_overrides: Optional[dict] = None
        self._pending_global_slot_index: Optional[int] = None

    def set_pending_overrides(self, overrides: Optional[dict]) -> None:
        self._pending_param_overrides = overrides

    def set_pending_global_slot_index(self, global_idx: Optional[int]) -> None:
        self._pending_global_slot_index = global_idx

    def __getitem__(self, index):
        h5_path = self.case_list[index]
        data_dir = os.path.dirname(h5_path)
        case_path = os.path.dirname(data_dir)
        sidecar_dir = data_dir
        if not os.path.exists(os.path.join(data_dir, "patchinfo.toml")):
            legacy_dir = os.path.join(case_path, "h5")
            if os.path.exists(os.path.join(legacy_dir, "patchinfo.toml")):
                sidecar_dir = legacy_dir

        with h5py.File(h5_path, "r") as h5_file:
            if "case_name" not in h5_file.attrs:
                raise ValueError(f"H5 file {h5_path} is missing root attribute 'case_name'")
            case_name = h5_file.attrs["case_name"]
            if isinstance(case_name, bytes):
                case_name = case_name.decode("utf-8")
            if not isinstance(case_name, str) or not case_name.strip():
                raise ValueError(f"H5 file {h5_path} has invalid case_name={case_name!r}")

            mesh = {
                "case_name": case_name,
                "case_path": case_path,
                "sidecar_dir": sidecar_dir,
            }
            transform = CFDmesh(self.params)
            mesh["transform"] = transform

            patch_dict_path = os.path.join(sidecar_dir, "patchinfo.toml")
            if not os.path.exists(patch_dict_path):
                raise FileNotFoundError(f"Patch table not found at {patch_dict_path}")
            with open(patch_dict_path, "r") as f:
                patch_dict = toml_to_native(tomlkit.load(f))
            validate_patch_table(
                patch_dict,
                num_cells=h5_file["cell|cell_pos"].shape[0],
                num_faces=h5_file["face|face_pos"].shape[0],
                num_cpd_cells=h5_file["cpd|cell_pos"].shape[0],
                patch_table_path=patch_dict_path,
                h5_path=h5_path,
            )
            mesh["fvpatch"] = fvPatch().register(patch_dict["patch"])
            mesh["patch_dict"] = mesh["fvpatch"].patch_dict

            fvconfig_path = os.path.join(sidecar_dir, "fvconfig.toml")
            with open(fvconfig_path, "r") as f:
                fvconfig = toml_to_native(tomlkit.load(f))
            mesh["fvconfig"] = fvconfig
            fvconfig["theta_pde_raw"] = copy.deepcopy(fvconfig["theta_pde"])

            fvsol = fvconfig.setdefault("fvSolution", {})
            raw_pref = fvsol.get("pRefCell", -99)
            if raw_pref == -1:
                fvsol["pRefCell_valid_mask"] = False
            elif raw_pref >= 0:
                fvsol["pRefCell_valid_mask"] = True
            else:
                fvsol["pRefCell_valid_mask"] = False
                fvsol["pRefCell"] = -1
                fvsol["pRefValue"] = 0.0

            pending_global = self._pending_global_slot_index
            if pending_global is not None:
                case_seed = case_seed_for_global_slot(pending_global)
                self._pending_global_slot_index = None
            else:
                rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
                case_seed = 777 + rank * 1000 + CFDmesh._case_counter
            CFDmesh._case_counter += 1
            parametric_manager = ParametricManager(seed=case_seed)
            if isinstance(fvconfig.get("parameter"), dict):
                parametric_manager.register_from_toml(fvconfig["parameter"])

            bc_dir = os.path.join(sidecar_dir, "bc")
            boundary_info, boundary_info_raw = {}, {}
            solver_type = str(fvconfig["solver"]["type"]).lower()
            required_vars = SOLVER_VARS.get(solver_type)
            if required_vars is None:
                raise ValueError(f"Unknown solver type {solver_type!r}; supported: {list(SOLVER_VARS)}")

            if os.path.exists(bc_dir):
                available = {
                    fn[:-5]: os.path.join(bc_dir, fn)
                    for fn in os.listdir(bc_dir)
                    if fn.endswith(".toml")
                }
                for var_name in required_vars:
                    if var_name not in available:
                        raise FileNotFoundError(
                            f"Missing BC file '{var_name}.toml' in {bc_dir} for solver {solver_type!r}"
                        )
                    with open(available[var_name], "r") as f:
                        raw_bc = toml_to_native(tomlkit.load(f))
                    if "parameter" in raw_bc:
                        parametric_manager.register_from_toml(raw_bc["parameter"])
                    boundary_info[var_name] = raw_bc
                    boundary_info_raw[var_name] = raw_bc

            mesh["parametric_manager"] = parametric_manager
            mesh["boundary_info"] = boundary_info
            mesh["boundary_info_raw"] = boundary_info_raw

            if self._pending_param_overrides:
                for pname, pval in self._pending_param_overrides.items():
                    parametric_manager.override_value(pname, pval)
                self._pending_param_overrides = None

            for key in h5_file.keys():
                val = h5_file[key][()]
                mesh[key] = torch.from_numpy(val) if isinstance(val, np.ndarray) else val

        return transform(mesh, self.params)

    def __len__(self):
        return len(self.case_list)
