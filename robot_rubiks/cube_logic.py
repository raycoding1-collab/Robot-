"""Exact 3x3x3 puzzle logic: cubie state, face turns, facelet export, scramble + solve.

Cube-local frame (used everywhere in this project):
    +X = R face, -X = L face
    +Z = U face, -Z = D face
    -Y = F face (faces the solver), +Y = B face

A face turn "clockwise" means clockwise when looking at that face from outside,
i.e. a rotation of -90 deg about the face's outward normal.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

# face -> (axis index, layer coordinate, outward normal sign)
FACES = {
    "U": (2, +1),
    "D": (2, -1),
    "R": (0, +1),
    "L": (0, -1),
    "F": (1, -1),
    "B": (1, +1),
}

FACE_NORMAL = {
    "U": np.array([0, 0, 1]),
    "D": np.array([0, 0, -1]),
    "R": np.array([1, 0, 0]),
    "L": np.array([-1, 0, 0]),
    "F": np.array([0, -1, 0]),
    "B": np.array([0, 1, 0]),
}

# Western colour scheme, white top / green front
FACE_COLOR = {"U": "white", "D": "yellow", "F": "green", "B": "blue", "R": "red", "L": "orange"}


def rot_matrix(axis: int, quarter_turns: int) -> np.ndarray:
    """Integer rotation matrix for quarter_turns * 90deg about a principal axis (right-hand rule)."""
    q = quarter_turns % 4
    c = [1, 0, -1, 0][q]
    s = [0, 1, 0, -1][q]
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass(frozen=True)
class Move:
    face: str  # U R F D L B
    turns: int  # 1 = clockwise, -1 = counter-clockwise, 2 = half turn

    @property
    def axis(self) -> int:
        return FACES[self.face][0]

    @property
    def layer(self) -> int:
        return FACES[self.face][1]

    @property
    def signed_axis_angle_quarters(self) -> int:
        """Quarter turns about the +axis (right-hand rule) that realise this move."""
        n = FACES[self.face][1]  # outward normal sign along axis
        # clockwise = -90deg about outward normal = -n*90deg about +axis
        t = self.turns if self.turns != 2 else 2
        return -n * t

    def __str__(self) -> str:
        return self.face + {1: "", -1: "'", 2: "2"}[self.turns]

    @staticmethod
    def parse(tok: str) -> "Move":
        face = tok[0]
        suf = tok[1:]
        turns = {"": 1, "'": -1, "2": 2}[suf]
        return Move(face, turns)

    def inverse(self) -> "Move":
        return Move(self.face, -self.turns if self.turns != 2 else 2)


def parse_alg(s: str) -> list[Move]:
    return [Move.parse(t) for t in s.split()]


class CubeState:
    """26 cubies, each with a home position and an integer orientation matrix.

    current_position = R @ home_position   (all turns are about the cube centre)
    """

    def __init__(self):
        self.homes: list[tuple[int, int, int]] = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    if (x, y, z) != (0, 0, 0):
                        self.homes.append((x, y, z))
        self.R = [np.eye(3, dtype=int) for _ in self.homes]

    def copy(self) -> "CubeState":
        c = CubeState()
        c.R = [r.copy() for r in self.R]
        return c

    def position(self, i: int) -> np.ndarray:
        return self.R[i] @ np.array(self.homes[i])

    def layer_members(self, move: Move) -> list[int]:
        return [i for i in range(len(self.homes)) if self.position(i)[move.axis] == move.layer]

    def apply(self, move: Move) -> list[int]:
        q = rot_matrix(move.axis, move.signed_axis_angle_quarters)
        members = self.layer_members(move)
        for i in members:
            self.R[i] = q @ self.R[i]
        return members

    def apply_alg(self, moves):
        for m in moves:
            self.apply(m)

    def is_solved(self) -> bool:
        """Solved up to (invisible) spin of the 6 centre pieces."""
        return all(
            (r == np.eye(3, dtype=int)).all()
            for r, h in zip(self.R, self.homes)
            if sum(abs(c) for c in h) > 1
        )

    # ---------------------------------------------------------------- facelets
    def _cubie_at(self, pos) -> int:
        pos = np.array(pos)
        for i in range(len(self.homes)):
            if (self.position(i) == pos).all():
                return i
        raise KeyError(pos)

    def facelet_color_face(self, pos, normal) -> str:
        """Which home face's colour shows at grid position `pos` on the side with `normal`."""
        i = self._cubie_at(pos)
        home_dir = self.R[i].T @ np.array(normal)  # direction in the cubie's home frame
        for f, n in FACE_NORMAL.items():
            if (n == home_dir).all():
                return f
        raise RuntimeError

    def facelet_string(self) -> str:
        """Kociemba facelet string URFDLB (each 9 facelets, reading order of the standard net)."""
        out = []
        # For each face: list of 9 positions in reading order.
        def grid(face):
            pts = []
            if face == "U":  # view from above, B at top
                for y in (1, 0, -1):
                    for x in (-1, 0, 1):
                        pts.append((x, y, 1))
            elif face == "R":  # view from right, U top, F left
                for z in (1, 0, -1):
                    for y in (-1, 0, 1):
                        pts.append((1, y, z))
            elif face == "F":  # view from front, U top, L left
                for z in (1, 0, -1):
                    for x in (-1, 0, 1):
                        pts.append((x, -1, z))
            elif face == "D":  # view from below, F top
                for y in (-1, 0, 1):
                    for x in (-1, 0, 1):
                        pts.append((x, y, -1))
            elif face == "L":  # view from left, U top, B left
                for z in (1, 0, -1):
                    for y in (1, 0, -1):
                        pts.append((-1, y, z))
            elif face == "B":  # view from behind, U top, R left
                for z in (1, 0, -1):
                    for x in (1, 0, -1):
                        pts.append((x, 1, z))
            return pts

        for face in "URFDLB":
            for p in grid(face):
                out.append(self.facelet_color_face(p, FACE_NORMAL[face]))
        return "".join(out)


def random_scramble(n: int = 22, seed: int = 7) -> list[Move]:
    rng = random.Random(seed)
    moves: list[Move] = []
    axis_of = {"U": 2, "D": 2, "R": 0, "L": 0, "F": 1, "B": 1}
    while len(moves) < n:
        f = rng.choice("URFDLB")
        if moves and moves[-1].face == f:
            continue
        if len(moves) >= 2 and axis_of[moves[-1].face] == axis_of[f] and axis_of[moves[-2].face] == axis_of[f]:
            continue
        moves.append(Move(f, rng.choice([1, -1, 2])))
    return moves


def solve(state: CubeState) -> list[Move]:
    import kociemba

    sol = kociemba.solve(state.facelet_string())
    return parse_alg(sol)


def find_scramble_and_solution(seed: int = 7, target=(18, 21)):
    """Deterministically pick a scramble whose two-phase solution length is in target range."""
    s = seed
    while True:
        scr = random_scramble(22, s)
        st = CubeState()
        st.apply_alg(scr)
        sol = solve(st)
        chk = st.copy()
        chk.apply_alg(sol)
        assert chk.is_solved(), "solver produced invalid solution"
        if target[0] <= len(sol) <= target[1]:
            return scr, sol, s
        s += 1


if __name__ == "__main__":
    st = CubeState()
    assert st.facelet_string() == "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9
    # sanity: known identity (R U R' U') * 6 = I
    st.apply_alg(parse_alg("R U R' U'") * 6)
    assert st.is_solved()
    # superflip check with kociemba string from a known alg
    scr, sol, seed = find_scramble_and_solution()
    print("seed", seed)
    print("scramble:", " ".join(map(str, scr)))
    print("solution (%d):" % len(sol), " ".join(map(str, sol)))
