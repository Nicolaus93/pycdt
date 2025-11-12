from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from shewchuk import incircle_test

from pycdt.delaunay import Triangulation, incircle_test_debug
from pycdt.geometry import ensure_ccw_triangle


def find_neighbor_edge_index(
    triangle_neighbors: NDArray, triangle_idx: int, neighbor_idx: int
) -> int:
    """
    Find which edge index (0, 1, or 2) connects to the given neighbor.
    Returns the index i such that triangle_neighbors[triangle_idx, i] == neighbor_idx
    """
    neighbors = triangle_neighbors[triangle_idx]
    for i, neighbor in enumerate(neighbors):
        if neighbor == neighbor_idx:
            return i
    raise RuntimeError(
        f"Triangle {triangle_idx} is not a neighbor of triangle {neighbor_idx}"
    )


class SharedEdgeError(Exception): ...


@dataclass(frozen=True)
class SwapDiagonalResult:
    """Result of a diagonal swap operation.

    Attributes
    ----------
    t7 : int
        Triangle opposite to point_idx across one edge of the new diagonal
    t8 : int
        Triangle opposite to point_idx across the other edge of the new diagonal
    diagonal_vk : int
        First vertex index of the new diagonal edge
    diagonal_vl : int
        Second vertex index of the new diagonal edge
    """

    t7: int
    t8: int
    diagonal_vk: int
    diagonal_vl: int


def find_shared_edge(
    tri1_vertices: NDArray, tri2_vertices: NDArray
) -> tuple[int, int, int, int]:
    """
    Find the shared edge between two triangles.
    Returns (v1, v2, opposite1, opposite2) where:
    - v1, v2 are the shared vertices
    - opposite1 is the vertex in tri1 not on the shared edge
    - opposite2 is the vertex in tri2 not on the shared edge
    """
    shared = list(set(tri1_vertices) & set(tri2_vertices))
    if len(shared) != 2:
        raise SharedEdgeError(
            f"Triangles must share exactly one edge. Shared vertices: {shared}"
        )

    v1, v2 = shared
    opposite1 = next(v for v in tri1_vertices if v not in shared)
    opposite2 = next(v for v in tri2_vertices if v not in shared)

    return v1, v2, opposite1, opposite2


def find_vertex_position(triangle_vertices: NDArray, vertex: int) -> int:
    """Find the position (0, 1, or 2) of a vertex in a triangle"""
    return next(i for i in range(3) if triangle_vertices[i] == vertex)


def swap_diagonal(
    triangulation: Triangulation,
    t3_idx: int,
    t4_idx: int,
    point_idx: int | None = None,
) -> SwapDiagonalResult:
    """
    Swap the diagonal between two adjacent triangles.

    This function performs an edge flip operation, replacing the shared edge between
    two triangles with a new edge connecting the opposite vertices.

    The operation replaces the shared edge between two triangles with a new edge
    connecting the two opposite vertices, creating two new triangles.

    Before swap:
        Triangle t4_idx: contains point_idx and shares edge (a, b) with t3_idx
        Triangle t3_idx: shares edge (a, b) with t4_idx, has opposite vertex c
        Shared edge: (a, b)

    After swap:
        New triangle at t3_idx: (point_idx, c, a)
        New triangle at t4_idx: (point_idx, b, c)
        New shared edge: (point_idx, c)

    The function updates:
    - Triangle vertices to form new triangles (ensuring CCW orientation)
    - Triangle neighbor relationships for both modified triangles
    - Neighbor references in adjacent triangles

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation to modify (modified in-place)
    t3_idx : int
        Index of the neighboring triangle
    t4_idx : int
        Index of the candidate triangle (contains point_idx)
    point_idx : int | None, optional
        Index of a vertex in t4_idx that is opposite to the shared edge.
        If None, will be automatically determined from the shared edge.

    Returns
    -------
    SwapDiagonalResult
        A dataclass containing:
        - t7: triangle opposite to point_idx across edge (point_idx, b) in new t4_idx
        - t8: triangle opposite to point_idx across edge (c, point_idx) in new t3_idx
        - diagonal_vk: first vertex index of the new diagonal (point_idx)
        - diagonal_vl: second vertex index of the new diagonal (c)
        The t7 and t8 triangles need to be checked next for Delaunay violations.

    Raises
    ------
    ValueError
        If point_idx is provided but not in triangle t4_idx, or if the triangles don't share an edge

    Notes
    -----
    This function is used both for Lawson's incremental flip algorithm during Delaunay
    construction and for constraint edge insertion in constrained Delaunay triangulation.
    """
    vertices = triangulation.triangle_vertices
    neighbors = triangulation.triangle_neighbors
    points = triangulation.all_points

    t3 = vertices[t3_idx]
    t4 = vertices[t4_idx]

    # Find shared edge and opposite vertices
    a, b, temp, c = find_shared_edge(t4, t3)

    # If point_idx is not provided, use temp (the opposite vertex in t4)
    if point_idx is None:
        point_idx = temp
    else:
        # The candidate triangle should contain the newly inserted point
        if temp not in t4:
            raise ValueError("temp not in t4")
        if not temp == point_idx:
            raise ValueError(f"Expected point_idx {point_idx}, but got {temp}")

    # Create new triangles after edge flip
    # Ensure counterclockwise orientation
    new_t3_vertices = ensure_ccw_triangle(np.array([point_idx, c, a]), points)
    new_t4_vertices = ensure_ccw_triangle(np.array([point_idx, b, c]), points)

    # Find neighbor triangles before updating
    # For triangle t4_idx, find neighbors opposite to each vertex
    old_t4_neighbors = neighbors[t4_idx].copy()
    old_t3_neighbors = neighbors[t3_idx].copy()

    # Find which edges correspond to which neighbors in the old triangles
    # For t4_idx (contains point_idx)
    # Neighbor opposite to A in t4
    pos_a = find_vertex_position(t4, a)
    t6 = int(old_t4_neighbors[pos_a])
    # Neighbor opposite to B in candidate triangle
    pos_b = find_vertex_position(t4, b)
    t5 = int(old_t4_neighbors[pos_b])

    # For t3_idx
    # Neighbor opposite to A in neigh triangle
    pos_a_neigh = find_vertex_position(t3, a)
    t7 = int(old_t3_neighbors[pos_a_neigh])
    # Neighbor opposite to B in neigh triangle
    pos_b_neigh = find_vertex_position(t3, b)
    t8 = int(old_t3_neighbors[pos_b_neigh])

    # Update triangle vertices
    vertices[t3_idx] = new_t3_vertices
    vertices[t4_idx] = new_t4_vertices

    # Update neighbors for new triangles
    # We need to determine neighbors based on the actual vertex ordering after CCW correction

    # For new_t3: determine neighbors for each edge
    t3_neighbors = np.array([-1, -1, -1])
    for i in range(3):
        curr_v = new_t3_vertices[i]
        if curr_v == point_idx:
            t3_neighbors[i] = t8
        elif curr_v == c:
            t3_neighbors[i] = t5
        elif curr_v == a:
            t3_neighbors[i] = t4_idx
        else:
            raise RuntimeError

    # For new_t4: determine neighbors for each edge
    t4_neighbors = np.array([-1, -1, -1])
    for i in range(3):
        curr_v = new_t4_vertices[i]
        if curr_v == point_idx:
            t4_neighbors[i] = t7
        elif curr_v == c:
            t4_neighbors[i] = t6
        elif curr_v == b:
            t4_neighbors[i] = t3_idx
        else:
            raise RuntimeError

    neighbors[t3_idx] = t3_neighbors
    neighbors[t4_idx] = t4_neighbors

    # Update references in neighboring triangles
    def update_neighbor_reference(
        neighbor_idx: int, old_triangle: int, new_triangle: int
    ):
        if neighbor_idx != -1:
            mask = neighbors[neighbor_idx] == old_triangle
            if np.any(mask):
                neighbors[neighbor_idx][mask] = new_triangle

    # Update all affected neighbor references
    update_neighbor_reference(t5, t4_idx, t3_idx)
    update_neighbor_reference(t7, t3_idx, t4_idx)

    # Return triangles opposite to point_idx and the new diagonal vertices
    return SwapDiagonalResult(t7=t7, t8=t8, diagonal_vk=point_idx, diagonal_vl=c)


def lawson_swapping(
    point_idx: int,
    stack: list[tuple[int, int]],
    triangulation: Triangulation,
    debug: bool = False,
) -> None:
    """
    Restore the Delaunay condition by flipping edges as necessary.

    This implementation uses robust geometric predicates to ensure proper
    orientation handling and maintains counterclockwise vertex ordering.

    :param point_idx: Index of the newly inserted point
    :param stack: List of (triangle_idx, candidate_triangle_idx) pairs to check
    :param triangulation: Triangulation structure containing geometry and topology
    """
    p = triangulation.all_points[point_idx]

    logger.debug("Lawson swapping phase")
    while stack:
        logger.trace(f"Stack -> {stack}")
        t3_idx, t4_idx = stack.pop()

        # Skip if either triangle is invalid
        if t3_idx == -1 or t4_idx == -1:
            continue

        t3 = triangulation.triangle_vertices[t3_idx]
        t3_points = triangulation.all_points[t3]

        # Check if point lies in circumcircle of the neighboring triangle
        if incircle_test(*p, *t3_points[0], *t3_points[1], *t3_points[2]) < 0:
            if debug:
                incircle_test_debug(t3_points, p)
            continue

        logger.trace(
            f"Point {point_idx} from triangle {t4_idx} lies in circumcircle of triangle {t3_idx}; flipping shared edge"
        )
        result = swap_diagonal(
            triangulation,
            t3_idx,
            t4_idx,
            point_idx,
        )

        # Add new potentially illegal edges to stack
        #  (check edge opposite to point_idx in both new triangles)
        if result.t8 != -1:
            stack.append((int(result.t8), int(t3_idx)))
        if result.t7 != -1:
            stack.append((int(result.t7), int(t4_idx)))


def reorder_neighbors_for_triangle(
    original_vertices: np.ndarray, final_vertices: np.ndarray, original_neighbors: list
) -> list:
    """Reorder neighbor indices to match reordered vertices"""
    if np.array_equal(original_vertices, final_vertices):
        return original_neighbors

    # Create new neighbor array based on final vertex ordering
    final_neighbors = [None] * 3
    for idx in range(3):
        final_vertex = final_vertices[idx]
        # Find where this vertex was in the original ordering
        original_pos = np.where(original_vertices == final_vertex)[0][0]
        # The neighbor opposite to this vertex in the final triangle
        # is the same as the neighbor opposite to it in the original triangle
        final_neighbors[idx] = original_neighbors[original_pos]

    return final_neighbors
