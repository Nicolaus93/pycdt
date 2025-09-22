# type: ignore

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional

from src.delaunay import Triangulation
from src.geometry import orient2d, ensure_ccw_triangle


EPS = 1e-6


@dataclass
class IsConvexQuadResult:
    ok: bool
    t2: Optional[int]  # neighbor triangle across tested edge
    v0: Optional[int]  # opposite vertex in t1 (edge is (v1,v2))
    v1: Optional[int]  # shared endpoint
    v2: Optional[int]  # shared endpoint
    v3: Optional[int]  # opposite vertex in t2


def _is_convex_quad(
    triangulation: Triangulation, t1: int, s1: int
) -> IsConvexQuadResult:
    V1 = triangulation.triangle_vertices[t1]
    v0 = int(V1[s1])
    v1 = int(V1[(s1 + 1) % 3])
    v2 = int(V1[(s1 + 2) % 3])

    t2 = int(triangulation.triangle_neighbors[t1, s1])
    if t2 < 0:
        return IsConvexQuadResult(False, None, None, None, None, None)

    V2 = triangulation.triangle_vertices[t2]
    shared = {v1, v2}
    try:
        v3 = int(next(x for x in V2 if x not in shared))
    except StopIteration:
        return IsConvexQuadResult(False, t2, v0, v1, v2, None)

    A = triangulation.all_points[v1]
    B = triangulation.all_points[v2]
    P0 = triangulation.all_points[v0]
    P3 = triangulation.all_points[v3]

    o0 = orient2d(A, B, P0)
    o3 = orient2d(A, B, P3)
    if abs(o0) < EPS or abs(o3) < EPS or (o0 * o3) >= 0:
        return IsConvexQuadResult(False, t2, v0, v1, v2, v3)

    return IsConvexQuadResult(True, t2, v0, v1, v2, v3)


def _find_vertex_pos(tri_vertices: NDArray[np.integer], v: int) -> int:
    return int(next(i for i in range(3) if tri_vertices[i] == v))


def _replace_neighbor_ref(
    nei_idx: int, old_tri: int, new_tri: int, neighbors: NDArray[np.integer]
) -> None:
    if nei_idx < 0:
        return
    mask = neighbors[nei_idx] == old_tri
    if np.any(mask):
        neighbors[nei_idx][mask] = new_tri


def flip_adjacent_pair(
    triangulation: Triangulation, t1: int, t2: int, v1: int, v2: int, v0: int, v3: int
) -> tuple[int, int, int, int]:
    """
    Flip the shared diagonal (v1,v2) of adjacent triangles t1 (opposite v0) and t2 (opposite v3).
    Returns (t1, t2, v0, v3) where the **new** diagonal is (v0, v3).
    Reuses the neighbor mapping style from your Lawson implementation.
    """
    vertices = triangulation.triangle_vertices
    neighbors = triangulation.triangle_neighbors
    points = triangulation.all_points

    V1 = vertices[t1].copy()
    V2 = vertices[t2].copy()
    N1 = neighbors[t1].copy()
    N2 = neighbors[t2].copy()

    # positions in original triangles
    pos_v1_in_t1 = _find_vertex_pos(V1, v1)
    pos_v2_in_t1 = _find_vertex_pos(V1, v2)
    pos_v1_in_t2 = _find_vertex_pos(V2, v1)
    pos_v2_in_t2 = _find_vertex_pos(V2, v2)

    # external neighbors around the quad (same pattern as Lawson):
    # t1: opp v2 -> (v0,v1), opp v1 -> (v2,v0)
    n_b = int(N1[pos_v2_in_t1])  # neighbor across (v0, v1)
    n_a = int(N1[pos_v1_in_t1])  # neighbor across (v2, v0)
    # t2: opp v1 -> (v2,v3),  opp v2 -> (v3,v1)
    n_c = int(N2[pos_v1_in_t2])  # neighbor across (v2, v3)
    n_d = int(N2[pos_v2_in_t2])  # neighbor across (v3, v1)

    # build new triangles after flip: t1'=(v0,v1,v3), t2'=(v0,v3,v2) ensuring CCW
    new_t1 = ensure_ccw_triangle(np.array([v0, v1, v3]), points)
    new_t2 = ensure_ccw_triangle(np.array([v0, v3, v2]), points)
    vertices[t1] = new_t1
    vertices[t2] = new_t2

    # neighbors for t1': opp v0→t2, opp v1→n_d, opp v3→n_b
    t1_nei = np.array([-1, -1, -1], dtype=int)
    for i, vx in enumerate(new_t1):
        if vx == v0:
            t1_nei[i] = t2
        elif vx == v1:
            t1_nei[i] = n_d
        elif vx == v3:
            t1_nei[i] = n_b
    neighbors[t1] = t1_nei

    # neighbors for t2': opp v0→n_c, opp v3→n_a, opp v2→t1
    t2_nei = np.array([-1, -1, -1], dtype=int)
    for i, vx in enumerate(new_t2):
        if vx == v0:
            t2_nei[i] = n_c
        elif vx == v3:
            t2_nei[i] = n_a
        elif vx == v2:
            t2_nei[i] = t1
    neighbors[t2] = t2_nei

    # fix back‑references in the four external neighbors
    _replace_neighbor_ref(n_b, t2, t1, neighbors)  # (v0,v1) now sees t1
    _replace_neighbor_ref(n_d, t1, t1, neighbors)  # (v1,v3) stays on t1 after update
    _replace_neighbor_ref(n_c, t1, t2, neighbors)  # (v2,v3) now sees t2
    _replace_neighbor_ref(n_a, t2, t2, neighbors)  # (v2,v0) stays on t2 after update

    return t1, t2, v0, v3  # new diagonal is (v0,v3)


def flip_edge_by_handle(
    triangulation: Triangulation, t1: int, s1: int
) -> tuple[int, int, int, int]:
    """
    Flip the diagonal across the edge referenced by (t1, s1).
    Returns (t1, t2, v0, v3) where the **new** diagonal is (v0, v3).
    """
    res = _is_convex_quad(triangulation, t1, s1)
    if not res.ok or res.t2 is None or res.v0 is None or res.v3 is None:
        raise ValueError("flip called on non-convex/boundary configuration")

    # unpack as in your paper notation
    t2, v0, v1, v2, v3 = res.t2, res.v0, res.v1, res.v2, res.v3
    return flip_adjacent_pair(triangulation, t1, t2, v1, v2, v0, v3)
