"""Reading a case (mesh ``*.h5`` with ``patchinfo.toml``, ``fvconfig.toml`` and ``bc/*.toml``)
and building its initial state, boundary conditions and PDE coefficients."""

import copy
import importlib
import importlib.util
import logging
import os
from typing import Optional

import h5py
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

BC_MODULES = {
    "U": "FVdomain.SetBC.onthefly.setU",
    "T": "FVdomain.SetBC.onthefly.setT",
    "p": "FVdomain.SetBC.onthefly.setP",
    "p_rgh": "FVdomain.SetBC.onthefly.setP_rgh",
}


class CFDmesh:
    """Builds a case's state and coefficients; called on loading and on every reset, when
    the case's parameters are drawn again."""

    def __init__(self, params):
        self.params = params

    @staticmethod
    def _get_bc_class(bc_type: str, var_name: str, sidecar_dir: str):
        """Class of a ``type = "..."`` entry: from ``<case>/bc/set<field>.py`` when it defines
        one (case-specific conditions), else from the built-in module of the field."""
        class_name = bc_type[0].upper() + bc_type[1:].lower()
        custom_bc_path = os.path.join(sidecar_dir, "bc", f"set{var_name}.py")
        if os.path.exists(custom_bc_path):
            spec = importlib.util.spec_from_file_location(
                f"custom_bc_{var_name}_{os.path.basename(sidecar_dir)}", custom_bc_path)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
            bc_class = getattr(custom_module, class_name, None)
            if bc_class is not None:
                logger.info(f"Loaded custom BC class '{class_name}' from {custom_bc_path}")
                return bc_class
        bc_class = getattr(importlib.import_module(BC_MODULES[var_name]), class_name, None)
        if bc_class is None:
            raise AttributeError(f"no boundary condition '{bc_type}' for field {var_name}")
        return bc_class

    def init_env(self, mesh):
        """Draw the case parameters, build the initial state ``[N_cpd, C]`` and instantiate
        the boundary conditions."""
        patch_dict = mesh["patch_dict"]
        fvpatch = mesh["fvpatch"]
        fvconfig = mesh["fvconfig"]
        num_cpd_cell = mesh["cpd|cell_pos"].shape[0]
        parametric_manager = mesh["parametric_manager"]
        parametric_manager.reset_all()
        boundary_info = copy.deepcopy(mesh["boundary_info_raw"])

        phi_cpd_cell = []
        bcpatch = {BCType.FIRST: [], BCType.MIXED: [], BCType.CYCLIC: []}
        num_phi_channels = 0
        for var, bc_info in boundary_info.items():
            bcpatch[var] = []
            num_channels = VAR_CONFIG[var]
            internal = parametric_manager.resolve_value(bc_info["internalField"])
            var_cpd_cell = Functionbase(
                value_type=internal["value_type"], value=internal["val"],
                num_element=num_cpd_cell, channel_dim=num_channels,
            ).eval(device="cpu")

            for patch_name, patch_bc in bc_info["boundaryfield"].items():
                bc_class = self._get_bc_class(patch_bc["type"], var, mesh["sidecar_dir"])
                patch_boundary_info = {k: v for k, v in patch_bc.items() if k != "type"}
                patch_boundary_info["fvconfig"] = fvconfig
                patch_boundary_info["patch_dict"] = patch_dict
                bc_instance = bc_class(
                    patch_name=patch_name,
                    patch_idx=patch_dict[patch_name]["patch_idx"],
                    boundary_info=patch_boundary_info,
                    parametric_manager=parametric_manager,
                )
                bcpatch[bc_instance.bcType].append(bc_instance)
                var_cpd_cell = bc_instance.init_field(
                    var_cpd_cell, fvpatch,
                    neighbor_cpd_cell=mesh["cpd|neighbor_cell"],
                    fvconfig=fvconfig,
                    cpd_cell_pos=mesh["cpd|cell_pos"],
                )
                bcpatch[var].append(bc_instance)
            phi_cpd_cell.append(var_cpd_cell)
            num_phi_channels += num_channels

        phi_cpd_cell = torch.cat(phi_cpd_cell, dim=-1)
        mesh["bcpatch"] = bcpatch
        mesh["phi_num_channels"] = num_phi_channels
        # node features: state, initial state, cell-type one-hot (PDE coefficients and the
        # Fourier encoding are added below)
        mesh["num_channels"] = num_phi_channels * 2 + len(FaceType)
        mesh["init_phi_cpd_cell"] = phi_cpd_cell.clone()

        # the mean-pressure constraint applies when no condition fixes the pressure level
        fvsolution = mesh["fvconfig"]["fvSolution"]
        has_dirichlet_pressure = any(
            bc.varType in (VARType.PRESSURE, VARType.PRESSURE_RGH) for bc in bcpatch[BCType.FIRST])
        fvsolution["pRefCell_valid_mask"] = (not has_dirichlet_pressure) and fvsolution.get("pRefCell", -1) >= 0
        return mesh, phi_cpd_cell

    @staticmethod
    def _resolve_param_value(mesh, value, default=None):
        """A ``theta_pde`` entry: ``"$name"``, ``{value_type = "constant", val = ...}`` or a
        plain value."""
        if value is None:
            return default
        if isinstance(value, str):
            return mesh["parametric_manager"].resolve_value(value)
        if isinstance(value, dict):
            if value.get("value_type", "constant") not in ("constant", "uniform"):
                raise ValueError(f"theta_pde entries are constants or $parameters, got {value}")
            return value.get("val", default)
        return value

    @staticmethod
    def _inlet_velocity(mesh):
        """The ``inlet_velocity`` parameter of the case, or 1 when it has none."""
        try:
            v = mesh["parametric_manager"].get_value("inlet_velocity")
        except KeyError:
            return 1.0
        return abs(float(v))

    def set_theta_PDE(self, mesh):
        """The per-case coefficient vector ``theta_PDE`` (columns listed below)."""
        fvconfig = mesh["fvconfig"]
        solver_type = fvconfig["solver"]["type"]
        theta_pde = fvconfig["theta_pde"]
        theta_pde.clear()
        theta_pde.update(copy.deepcopy(fvconfig["theta_pde_raw"]))

        def get(key, default):
            return self._resolve_param_value(mesh, theta_pde.get(key), default=default)

        rho, mu, dt, L, source = get("rho", 1.0), get("mu", 1e-3), get("dt", 0.01), get("L", 1.0), get("source", 0.0)
        re_config = theta_pde.get("Re", None)
        if isinstance(re_config, dict) and "vel" in re_config:
            theta_pde["Re_config"] = re_config
        theta_pde["Re"] = rho * self._inlet_velocity(mesh) * L / mu if mu != 0 else float("inf")
        unsteady_coeff = 1.0 if theta_pde["unsteady"] else 0.0

        if solver_type == "picoext":
            coefficients = [
                unsteady_coeff,                    # 0 time derivative
                1.0,                               # 1 continuity
                1.0,                               # 2 convection
                1.0 / rho if rho != 0 else 1.0,    # 3 pressure gradient
                mu / rho if rho != 0 else mu,      # 4 kinematic viscosity
                source / rho if rho != 0 else source,  # 5 momentum source
                dt,                                # 6 time step
            ]
        else:
            beta, TRef, Pr = get("beta", 0.0), get("TRef", 300.0), get("Pr", 0.71)
            Cp, g, heat_source = get("Cp", 1005.0), get("g", [0.0, 0.0, -9.81]), get("heat_source", 0.0)
            alpha_cfg, hRef = get("alpha", None), get("hRef", 0.0)
            nu = mu / rho
            alpha = (nu / Pr if Pr != 0 else 0.0) if alpha_cfg is None else alpha_cfg
            coefficients = [
                unsteady_coeff,                    # 0 time derivative
                1.0,                               # 1 continuity
                1.0,                               # 2 convection
                1.0,                               # 3 pressure gradient (p_rgh is kinematic)
                mu / rho if rho != 0 else mu,      # 4 kinematic viscosity
                source / rho if rho != 0 else source,  # 5 momentum source
                dt,                                # 6 time step
                alpha,                             # 7 thermal diffusivity
                heat_source / (rho * Cp) if rho != 0 else heat_source,  # 8 heat source
                TRef,                              # 9 reference temperature
                beta,                              # 10 thermal expansion coefficient
                g[0], g[1], g[2],                  # 11-13 gravity
                hRef,                              # 14 reference height of gh
            ]
        theta_PDE = torch.tensor(coefficients, dtype=torch.float32).view(1, -1)
        mesh["num_channels"] = mesh.get("num_channels", 0) + theta_PDE.shape[1]
        mesh["theta_PDE"] = theta_PDE
        return mesh

    @staticmethod
    def _update_gnn_csr_format(mesh, neighbor_cpd_cell, num_cpd_cells):
        """Receiver-sorted two-way edge lists for segment reductions in message passing: per
        node the number of incoming edges, the sorted senders, and each sorted edge's index
        in the interleaved (forward, reverse) edge order."""
        num_edges = neighbor_cpd_cell.shape[1]
        twoway_edges = torch.cat([neighbor_cpd_cell, neighbor_cpd_cell.flip(0)], dim=1)
        sort_idx_raw = twoway_edges[1].argsort(stable=True)
        mesh["gnn|edge_counts"] = torch.bincount(twoway_edges[1][sort_idx_raw], minlength=num_cpd_cells)
        mesh["gnn|edge_indices"] = twoway_edges[0][sort_idx_raw]
        mesh["gnn|edge_sort_idx"] = torch.where(
            sort_idx_raw < num_edges, 2 * sort_idx_raw, 2 * (sort_idx_raw - num_edges) + 1)
        return mesh

    @staticmethod
    def calc_WLSQ_A_B_normal_matrix(mesh):
        """WLSQ normal matrices on the stencil stored in the mesh file."""
        if "A_cell_to_cell" not in mesh:
            neighbor_cpd_cell_x = mesh["cpd|neighbor_cell_x"].long()
            mesh["cpd|neighbor_cell_x"] = neighbor_cpd_cell_x
            A_cell_to_cell, two_way_B_cell_to_cell = compute_normal_matrix(
                node_pos=mesh["cpd|cell_pos"], edge_index=neighbor_cpd_cell_x)
            mesh["A_cell_to_cell"] = A_cell_to_cell.to(torch.float32)
            mesh["single_B_cell_to_cell"] = torch.chunk(two_way_B_cell_to_cell, 2, dim=0)[0].to(torch.float32)
        return mesh

    def __call__(self, mesh, params=None):
        mesh = prepare_cyclic_boundary_geometry(mesh)
        mesh, phi_cpd_cell = self.init_env(mesh)
        fourier_num_freqs = getattr(params, "fourier_num_freqs", 0) if params else 0
        mesh["num_channels"] += 3 * 2 * fourier_num_freqs if fourier_num_freqs > 0 else 0
        mesh["fourier_num_freqs"] = fourier_num_freqs
        mesh = self.set_theta_PDE(mesh)
        mesh = self.calc_WLSQ_A_B_normal_matrix(mesh)
        mesh = self._update_gnn_csr_format(mesh, mesh["cpd|neighbor_cell"], mesh["cpd|cell_pos"].shape[0])
        return mesh, phi_cpd_cell


class H5CFDdataset(Dataset):
    """The cases of ``case_list`` (paths of mesh files), built one at a time."""

    def __init__(self, params, case_list):
        super().__init__()
        self.case_list = case_list
        self.params = params
        self._pending_param_overrides: Optional[dict] = None
        self._pending_global_slot_index: Optional[int] = None

    def set_pending_overrides(self, overrides: Optional[dict]) -> None:
        """Per-case parameter values (``meta.json`` case_parameters) for the next case built."""
        self._pending_param_overrides = overrides

    def set_pending_global_slot_index(self, global_idx: Optional[int]) -> None:
        """The pool slot the next case fills; it seeds the case's parameter draws."""
        self._pending_global_slot_index = global_idx

    def __getitem__(self, index):
        h5_path = self.case_list[index]
        sidecar_dir = os.path.dirname(h5_path)
        with h5py.File(h5_path, "r") as h5_file:
            case_name = h5_file.attrs["case_name"]
            mesh = {key: torch.from_numpy(h5_file[key][()]) for key in h5_file.keys()}
        mesh["case_name"] = case_name.decode("utf-8") if isinstance(case_name, bytes) else case_name
        mesh["sidecar_dir"] = sidecar_dir
        mesh["transform"] = transform = CFDmesh(self.params)

        with open(os.path.join(sidecar_dir, "patchinfo.toml")) as f:
            patch_dict = toml_to_native(tomlkit.load(f))
        mesh["fvpatch"] = fvPatch().register(patch_dict["patch"])
        mesh["patch_dict"] = mesh["fvpatch"].patch_dict

        with open(os.path.join(sidecar_dir, "fvconfig.toml")) as f:
            fvconfig = toml_to_native(tomlkit.load(f))
        fvconfig["theta_pde_raw"] = copy.deepcopy(fvconfig["theta_pde"])
        fvsolution = fvconfig["fvSolution"]
        fvsolution.setdefault("pRefCell", -1)
        fvsolution.setdefault("pRefValue", 0.0)
        mesh["fvconfig"] = fvconfig

        parametric_manager = ParametricManager(seed=case_seed_for_global_slot(self._pending_global_slot_index))
        self._pending_global_slot_index = None
        if isinstance(fvconfig.get("parameter"), dict):
            parametric_manager.register_from_toml(fvconfig["parameter"])

        solver_type = fvconfig["solver"]["type"]
        bc_dir = os.path.join(sidecar_dir, "bc")
        boundary_info = {}
        for var_name in SOLVER_VARS[solver_type]:
            with open(os.path.join(bc_dir, f"{var_name}.toml")) as f:
                raw_bc = toml_to_native(tomlkit.load(f))
            if "parameter" in raw_bc:
                parametric_manager.register_from_toml(raw_bc["parameter"])
            boundary_info[var_name] = raw_bc
        logger.info(f"Loaded boundary files {sorted(boundary_info)} for case {mesh['case_name']} ({solver_type})")
        mesh["parametric_manager"] = parametric_manager
        mesh["boundary_info"] = boundary_info
        mesh["boundary_info_raw"] = boundary_info

        if self._pending_param_overrides:
            for name, value in self._pending_param_overrides.items():
                parametric_manager.override_value(name, value)
            self._pending_param_overrides = None

        return transform(mesh, self.params)
