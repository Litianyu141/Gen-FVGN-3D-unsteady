"""Mesh entity types."""

import enum


class FaceType(enum.IntEnum):
    """Type of a face, and of the compound cell it bounds (interior cells are NORMAL).  The
    network's cell-type one-hot has one channel per member."""
    NORMAL = 0
    PATCH = 1
    EMPTY = 2
    WALL = 3
    SYMMETRY = 4
    CYCLIC = 5
