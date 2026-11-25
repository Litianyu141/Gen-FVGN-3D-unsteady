"""Base types and batching containers for boundary condition descriptors."""

import enum
from abc import ABC, abstractmethod
from typing import List

VAR_CONFIG = {
    "U": 3,
    "p": 1,
    "p_rgh": 1,
    "T": 1,
}

SOLVER_VARS = {
    "picoext": ["U", "p"],
    "picocmpt": ["U", "p"],
    "buoyantboussinesqpicoext": ["U", "p_rgh", "T"],
    "buoyantboussinesqpicocmpt": ["U", "p_rgh", "T"],
}


class BCType(enum.IntEnum):
    FIRST = 0
    MIXED = 1
    CYCLIC = 2
    COUPLED = 3
    EMPTY = 99


class VARType(enum.IntEnum):
    VELOCITY = 1
    PRESSURE = 2
    PRESSURE_RGH = 2
    TEMPERATURE = 3


VAR_TYPE_BY_FIELD = {
    "U": VARType.VELOCITY,
    "p": VARType.PRESSURE,
    "p_rgh": VARType.PRESSURE_RGH,
    "T": VARType.TEMPERATURE,
}


class BoundaryCondition(ABC):
    """Boundary condition descriptor for mesh initialization and Warp compilation."""

    depends_on_old_state = False

    def __init__(self, patch_name: str, patch_idx: int, varType: VARType, bcType: BCType):
        self.patch_name = str(patch_name)
        self.patch_idx = int(patch_idx)
        self.varType = varType
        self.bcType = bcType
        self.is_coupled = bcType == BCType.COUPLED

    @abstractmethod
    def init_field(self, phi_cpd_cell, fvpatch, **kwargs):
        """Populate ghost cells of this patch when constructing the initial field."""

    def eval(self, fvfield, integrator, patch_idx_offset: int = 0, **kwargs):
        raise NotImplementedError(
            "Runtime boundary condition enforcement runs in NVIDIA Warp via WarpIncompressibleBCs."
        )


class BoundaryConditionBatch:
    """Boundary conditions of a batch of cases, grouped by application phase."""

    def __init__(self):
        self.first: List[list] = []
        self.mixed: List[list] = []
        self.cyclic: List[list] = []
        self.empty: List[list] = []
        self.coupled: List[list] = []

        self.first_patch_idx_offset: List[int] = [0]
        self.mixed_patch_idx_offset: List[int] = [0]
        self.cyclic_patch_idx_offset: List[int] = [0]
        self.empty_patch_idx_offset: List[int] = [0]
        self.coupled_patch_idx_offset: List[int] = [0]

    def iter_conditions(self):
        """Yield every concrete BC in deterministic application-group order."""
        for groups in (self.first, self.cyclic, self.mixed, self.coupled, self.empty):
            for case_conditions in groups:
                yield from case_conditions

    @property
    def depends_on_old_state(self) -> bool:
        return any(bc.depends_on_old_state for bc in self.iter_conditions())

    def old_state_dependent_conditions(self):
        return [bc for bc in self.iter_conditions() if bc.depends_on_old_state]

    @classmethod
    def from_bcpatch_list(cls, bcpatch_dict_list: List[dict], fvpatch_list: List):
        batch = cls()
        offset = 0
        for case_i, bcpatch_dict in enumerate(bcpatch_dict_list):
            if BCType.FIRST in bcpatch_dict:
                batch.first.append(bcpatch_dict[BCType.FIRST])
                if case_i > 0:
                    batch.first_patch_idx_offset.append(offset)
            if BCType.MIXED in bcpatch_dict:
                batch.mixed.append(
                    sorted(
                        bcpatch_dict[BCType.MIXED],
                        key=lambda bc: bool(getattr(bc, "APPLY_LAST", False)),
                    )
                )
                if case_i > 0:
                    batch.mixed_patch_idx_offset.append(offset)
            if BCType.CYCLIC in bcpatch_dict:
                batch.cyclic.append(bcpatch_dict[BCType.CYCLIC])
                if case_i > 0:
                    batch.cyclic_patch_idx_offset.append(offset)
            if BCType.EMPTY in bcpatch_dict:
                batch.empty.append(bcpatch_dict[BCType.EMPTY])
                if case_i > 0:
                    batch.empty_patch_idx_offset.append(offset)
            if BCType.COUPLED in bcpatch_dict:
                batch.coupled.append(bcpatch_dict[BCType.COUPLED])
                if case_i > 0:
                    batch.coupled_patch_idx_offset.append(offset)
            offset += fvpatch_list[case_i].num_patches
        return batch
