"""Boundary-condition base class and the per-batch container."""

import enum
from abc import ABC, abstractmethod
from typing import List

# channels of each field, and the fields (one bc/<field>.toml each) of every solver
VAR_CONFIG = {"U": 3, "p": 1, "p_rgh": 1, "T": 1}
SOLVER_VARS = {
    "picoext": ["U", "p"],
    "buoyantboussinesqpicoext": ["U", "p_rgh", "T"],
    "buoyantboussinesqpicocmpt": ["U", "p_rgh", "T"],
}


class BCType(enum.IntEnum):
    FIRST = 0    # Dirichlet: the ghost cell holds the face value
    MIXED = 1    # Neumann-type: ghost value from the owner, zero or prescribed normal flux
    CYCLIC = 2   # periodic


class VARType(enum.IntEnum):
    VELOCITY = 1
    PRESSURE = 2
    PRESSURE_RGH = 2
    TEMPERATURE = 3


class BoundaryCondition(ABC):
    """A condition on one patch for one field.

    ``init_field`` writes the initial ghost values of a single case; ``eval`` enforces the
    condition on the batched fields in ``fvfield`` and returns it.
    """

    def __init__(self, patch_name: str, patch_idx: int, varType: VARType, bcType: BCType):
        self.patch_name = str(patch_name)
        self.patch_idx = int(patch_idx)
        self.varType = varType
        self.bcType = bcType

    @abstractmethod
    def init_field(self, phi_cpd_cell, fvpatch, **kwargs):
        """Initial ghost values."""

    @abstractmethod
    def eval(self, fvfield, integrator, patch_idx_offset: int, **kwargs):
        """Enforce the condition on the batched fields."""


class BoundaryConditionBatch:
    """The conditions of a batch of cases grouped by type, with each case's offset into the
    batched patch list."""

    def __init__(self):
        self.first, self.mixed, self.cyclic = [], [], []
        self.first_patch_idx_offset = [0]
        self.mixed_patch_idx_offset = [0]
        self.cyclic_patch_idx_offset = [0]

    @classmethod
    def from_bcpatch_list(cls, bcpatch_dict_list: List[dict], fvpatch_list: List):
        batch = cls()
        cumulative_num_patches = 0
        for case_i, bcpatch_dict in enumerate(bcpatch_dict_list):
            groups = (
                (BCType.FIRST, batch.first, batch.first_patch_idx_offset),
                (BCType.MIXED, batch.mixed, batch.mixed_patch_idx_offset),
                (BCType.CYCLIC, batch.cyclic, batch.cyclic_patch_idx_offset),
            )
            for bc_type, conditions, offsets in groups:
                if bc_type not in bcpatch_dict:
                    continue
                case_conditions = bcpatch_dict[bc_type]
                if bc_type == BCType.MIXED:
                    # conditions that read other fields' ghost values go last
                    case_conditions = sorted(case_conditions, key=lambda bc: bool(getattr(bc, "APPLY_LAST", False)))
                conditions.append(case_conditions)
                if case_i > 0:
                    offsets.append(cumulative_num_patches)
            cumulative_num_patches += fvpatch_list[case_i].num_patches
        return batch
