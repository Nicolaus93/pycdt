"""Constrained Delaunay Triangulation implementation."""

import numpy as np
from numpy.typing import NDArray
from loguru import logger

from pycdt.delaunay import Triangulation
from pycdt.build import find_containing_triangle
from pycdt.geometry import point_inside_triangle, PointInTriangle, is_point_in_box
from pycdt.utils import EPS, Vec2d
from shewchuk import orientation


def find_triangle_for_point(
    triangulation: Triangulation, point: NDArray[np.floating]
) -> int:
    """Find triangle containing a point, handling the vertex case explicitly."""
    # Check if point is on an existing vertex
    vertex_matches = np.all(
        np.isclose(triangulation.all_points, point, atol=EPS), axis=1
    )
    if np.any(vertex_matches):
        vertex_idx = np.argmax(vertex_matches)
        # Find any triangle containing this vertex
        for tri_idx, tri_verts in enumerate(triangulation.triangle_vertices):
            if vertex_idx in tri_verts:
                return tri_idx
        raise RuntimeError(f"No triangle containing point {point}")

    # Point is not on a vertex, use normal search
    # TODO: use AABB search?
    start_idx = 0
    result = find_containing_triangle(triangulation, point, start_idx)
    return result.idx


def find_intesercting_edges(
    triangulation: Triangulation,
    p_idx: int,
    q_idx: int,
) -> list[tuple[int, int]]:
    """
    Find all triangles that are overlapped by segment pq.

    This implements the walking algorithm:
    1. Find triangle containing p (tp) and triangle containing q (tq)
    2. Start from tp, walk towards tq
    3. At each triangle, find which edge the segment exits through
    4. Move to the adjacent triangle across that edge
    5. Continue until reaching tq

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation to search
    p_idx : NDArray[np.floating]
        Start point of constraint segment
    q_idx : NDArray[np.floating]
        End point of constraint segment

    Returns
    -------
    list[tuple[int, int]]
        List of (v1, v2) vertex index pairs of intersected edges, in order from p to q
    """
    p = triangulation.all_points[p_idx]
    q = triangulation.all_points[q_idx]

    # Find triangles containing p and q
    tp = find_triangle_for_point(triangulation=triangulation, point=p)
    tq = find_triangle_for_point(triangulation=triangulation, point=q)

    # If both points in same triangle, it means pq is an edge. Return that triangle
    if tp == tq:
        return [(p_idx, q_idx)]

    # Walk from tp to tq
    intersecting = []
    current = tp
    visited = {tp}
    max_iterations = len(triangulation.triangle_vertices)
    for iteration in range(max_iterations):
        if current == tq:
            break

        # Get current triangle
        tri_verts = triangulation.triangle_vertices[current]
        tri_points = triangulation.all_points[tri_verts]

        # Check if q is in current triangle
        q_status, _ = point_inside_triangle(tri_points, q)
        if q_status != PointInTriangle.outside:
            # Reached destination
            break

        # Find which edge the segment pq exits through
        # Triangle edges are: (v0,v1), (v1,v2), (v2,v0)
        # The neighbor opposite to vertex i is at triangle_neighbors[current][i]
        a, b, c = tri_points
        edges = [
            (a, b, 2),  # edge v0-v1, opposite vertex is v2
            (b, c, 0),  # edge v1-v2, opposite vertex is v0
            (c, a, 1),  # edge v2-v0, opposite vertex is v1
        ]

        next_triangle = -1
        for i, (edge_start, edge_end, opposite_vertex_idx) in enumerate(edges):
            # Skip intersection at starting/ending point
            if (
                np.allclose(p, edge_start, atol=EPS)
                or np.allclose(p, edge_end, atol=EPS)
                or np.allclose(q, edge_start, atol=EPS)
                or np.allclose(q, edge_end, atol=EPS)
            ):
                continue

            # Check if segment pq intersects this edge
            if not segments_intersect(p, q, edge_start, edge_end):
                continue

            # Check if this is an exit edge (q is on the other side)
            # Use orientation test: if q and the opposite vertex are on different sides,
            # this is an exit edge
            o_q = orientation(*edge_start, *edge_end, *q)
            o_opposite = orientation(
                *edge_start, *edge_end, *tri_points[opposite_vertex_idx]
            )

            # If signs differ, q is on the opposite side from the opposite vertex
            # This means this edge leads towards q
            if o_q * o_opposite < 0 or o_q == 0:
                neighbor_idx = triangulation.triangle_neighbors[current][
                    opposite_vertex_idx
                ]
                if neighbor_idx == -1:
                    raise RuntimeError("Ended up outside triangulation")
                if neighbor_idx not in visited:
                    # if it's in visited, it could the one where we're "coming" from
                    next_triangle = neighbor_idx
                    # record intersected edge
                    if i == 0:
                        intersecting.append(tuple(sorted((tri_verts[0], tri_verts[1]))))
                    elif i == 1:
                        intersecting.append(tuple(sorted((tri_verts[1], tri_verts[2]))))
                    else:
                        intersecting.append(tuple(sorted((tri_verts[2], tri_verts[0]))))
                    break

        if next_triangle == -1:
            raise RuntimeError(
                f"Failed to find next triangle at iteration {iteration}, current={current}"
            )

        current = next_triangle
        visited.add(current)

    logger.debug(f"Found {len(intersecting)} intersecting edges from p={p} to q={q}")
    return intersecting


def remove_intersecting_edges(
    triangulation: Triangulation, edges: list[tuple[int, int]]
) -> None:
    logger.warning("TODO")
    return


def insert_constraint_edge(
    triangulation: Triangulation,
    p_idx: int,
    q_idx: int,
) -> bool:
    """
    Insert a constraint edge into the triangulation.

    This function:
    1. Finds all triangles overlapped by the constraint
    2. Finds all edges intersected by the constraint
    3. Removes those edges to create a polygonal cavity
    4. Extracts the boundary polygons on each side of the constraint

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation to modify (modified in-place)
    p_idx : int
        Vertex index of p in triangulation.all_points
    q_idx : int
        Vertex index of q in triangulation.all_points

    Returns
    -------
    bool
        True if constraint was successfully inserted, False otherwise
    """
    # Find all overlapped triangles
    intersected_edges = find_intesercting_edges(triangulation, p_idx, q_idx)

    if not intersected_edges:
        logger.info(f"Constraint edge {p_idx}-{q_idx} does not intersect any edges")
        return True

    logger.info(
        f"Inserting constraint {p_idx}-{q_idx}, removing {len(intersected_edges)} edges"
    )

    # Remove intersected edges
    remove_intersecting_edges(triangulation, intersected_edges)

    return False


def add_constraints(
    triangulation: Triangulation,
    constraints: list[tuple[int, int]],
) -> None:
    """
    Add constraint edges to a triangulation.

    This modifies the triangulation to include the constraint edges,
    creating a Constrained Delaunay Triangulation (CDT).

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation to constrain (modified in-place)
    constraints : list[tuple[int, int]]
        List of constraint edges as (vertex_idx_1, vertex_idx_2) pairs.
        Vertex indices refer to indices in triangulation.all_points.

    Notes
    -----
    The resulting triangulation will:
    - Contain all the constraint edges
    - Be as close to Delaunay as possible while respecting constraints
    - May have some triangles that violate the Delaunay property if necessary
      to accommodate the constraints
    """
    logger.info(f"Adding {len(constraints)} constraints to triangulation")

    for i, (v1_idx, v2_idx) in enumerate(constraints):
        logger.debug(
            f"Processing constraint {i + 1}/{len(constraints)}: {v1_idx}-{v2_idx}"
        )
        success = insert_constraint_edge(triangulation, v1_idx, v2_idx)
        if not success:
            logger.warning(f"Failed to insert constraint {v1_idx}-{v2_idx}")


def segments_intersect(
    p1: Vec2d | NDArray[np.floating],
    p2: Vec2d | NDArray[np.floating],
    q1: Vec2d | NDArray[np.floating],
    q2: Vec2d | NDArray[np.floating],
) -> bool:
    """
    Check if two line segments [p1, p2] and [q1, q2] intersect (including endpoints).

    Uses the orientation-based method: two segments intersect if and only if
    one of the following conditions holds:
    1. General case: (q1, q2, p1) and (q1, q2, p2) have different orientations AND
                     (p1, p2, q1) and (p1, p2, q2) have different orientations
    2. Special case: Points are collinear and segments overlap

    Parameters
    ----------
    p1, p2 : NDArray[np.floating]
        Endpoints of first segment
    q1, q2 : NDArray[np.floating]
        Endpoints of second segment
    eps : float
        Tolerance for coordinate equality

    Returns
    -------
    bool
        True if segments intersect, False otherwise
    """
    o1 = orientation(*q1, *q2, *p1)
    o2 = orientation(*q1, *q2, *p2)
    o3 = orientation(*p1, *p2, *q1)
    o4 = orientation(*p1, *p2, *q2)

    # General case: segments intersect if orientations differ
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    # Special cases: check if points are collinear and segments overlap
    if o1 == 0 and is_point_in_box(q1, q2, p1):
        return True
    if o2 == 0 and is_point_in_box(q1, q2, p2):
        return True
    if o3 == 0 and is_point_in_box(p1, p2, q1):
        return True
    if o4 == 0 and is_point_in_box(p1, p2, q2):
        return True

    return False
