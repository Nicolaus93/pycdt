from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from loguru import logger

from pycdt.debug_utils import _plot_intersecting_edges
from pycdt.delaunay import Triangulation
from pycdt.geometry import (
    point_inside_triangle,
    PointInTriangle,
    is_point_in_box,
)
from pycdt.utils import EPS, Vec2d
from pycdt.topology import find_shared_edge, swap_diagonal
from shewchuk import orientation, incircle_test


def collinear_overlap(p: Vec2d, q: Vec2d, a: Vec2d, b: Vec2d) -> bool:
    """
    Given that a and b are collinear with pq, check if segment ab overlaps pq.
    """
    return (
        is_point_in_box(p, q, a)
        or is_point_in_box(p, q, b)
        or is_point_in_box(a, b, p)
        or is_point_in_box(a, b, q)
    )


def segments_intersect(
    p1: Vec2d | NDArray[np.floating],
    p2: Vec2d | NDArray[np.floating],
    q1: Vec2d | NDArray[np.floating],
    q2: Vec2d | NDArray[np.floating],
) -> bool:
    """
    Check if two line segments [p1, p2] and [q1, q2] intersect (including endpoints).

    Uses the orientation-based method: two segments intersect if and only if:
    (q1, q2, p1) and (q1, q2, p2) have different orientations AND (p1, p2, q1) and (p1, p2, q2) have different
    orientations

    Parameters
    ----------
    p1, p2 : NDArray[np.floating]
        Endpoints of first segment
    q1, q2 : NDArray[np.floating]
        Endpoints of second segment

    Returns
    -------
    bool
        True if segments intersect, False otherwise
    """
    o1 = orientation(*q1, *q2, *p1)
    o2 = orientation(*q1, *q2, *p2)
    o3 = orientation(*p1, *p2, *q1)
    o4 = orientation(*p1, *p2, *q2)

    # If any are exactly collinear, this is not a proper intersection
    # (we handle collinearity separately).
    if o1 == 0 or o2 == 0 or o3 == 0 or o4 == 0:
        return False

    # General case: segments intersect if orientations differ
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    return False


@dataclass(frozen=True)
class IntersectedEdge:
    p1: int
    p2: int
    triangle_1: int
    triangle_2: int


def _check_proper_crossing(
    triangulation: Triangulation,
    current_tri_idx: int,
    p_idx: int,
    q_idx: int,
    visited: set[int],
) -> tuple[int, IntersectedEdge | None]:
    """
    Case C: Check for proper crossing (segments intersect but endpoints not collinear).

    This checks if the constraint segment pq properly crosses any edge of the current triangle.
    A proper crossing means the segments intersect at an interior point (not at endpoints).

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation
    current_tri_idx : int
        Current triangle index
    p_idx : int
        Start vertex index of constraint
    q_idx : int
        End vertex index of constraint
    visited : set[int]
        Set of already visited triangle indices

    Returns
    -------
    tuple[int, IntersectedEdge | None]
        (next_triangle_idx, intersected_edge) or (-1, None) if no crossing found
    """
    p = triangulation.all_points[p_idx]
    q = triangulation.all_points[q_idx]

    tri_verts = triangulation.triangle_vertices[current_tri_idx]
    tri_points = triangulation.all_points[tri_verts]

    a, b, c = tri_points
    edges = [
        (a, b, 2, 0, 1),  # edge v0-v1, opposite vertex is v2
        (b, c, 0, 1, 2),  # edge v1-v2, opposite vertex is v0
        (c, a, 1, 2, 0),  # edge v2-v0, opposite vertex is v1
    ]

    for edge_start, edge_end, opposite_vertex_idx, v_start_local, v_end_local in edges:
        # Orientation tests wrt pq
        pqs_orient = orientation(*p, *q, *edge_start)
        pqe_orient = orientation(*p, *q, *edge_end)

        # Skip if any endpoint is collinear (handled by other cases)
        if pqs_orient == 0 or pqe_orient == 0:
            continue

        # Check if segments intersect
        if not segments_intersect(p, q, edge_start, edge_end):
            continue

        # Check if this is an exit edge (q is on the other side)
        o_q = orientation(*edge_start, *edge_end, *q)
        o_opposite = orientation(
            *edge_start, *edge_end, *tri_points[opposite_vertex_idx]
        )

        # If signs differ, q is on the opposite side from the opposite vertex
        if o_q * o_opposite < 0:
            neighbor_idx = triangulation.triangle_neighbors[current_tri_idx][
                opposite_vertex_idx
            ]

            if neighbor_idx == -1:
                raise RuntimeError("Ended up outside triangulation")

            if neighbor_idx in visited:
                continue

            # Record intersected edge
            v1, v2 = sorted((tri_verts[v_start_local], tri_verts[v_end_local]))
            edge = IntersectedEdge(v1, v2, current_tri_idx, neighbor_idx)
            return neighbor_idx, edge

    return -1, None


def _check_collinear_overlap(
    triangulation: Triangulation,
    current_tri_idx: int,
    p_idx: int,
    q_idx: int,
    visited: set[int],
) -> tuple[int, IntersectedEdge | None]:
    """
    Case A: Both edge endpoints are collinear with pq.

    When both endpoints of a triangle edge lie on the line containing pq,
    we need to check if the edge overlaps with the segment pq. If so,
    we walk across it but DON'T record it as intersecting.

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation
    current_tri_idx : int
        Current triangle index
    p_idx : int
        Start vertex index of constraint
    q_idx : int
        End vertex index of constraint
    visited : set[int]
        Set of already visited triangle indices

    Returns
    -------
    tuple[int, IntersectedEdge | None]
        (next_triangle_idx, None) or (-1, None) if no overlap found.
        Note: Returns None for edge because collinear edges are not counted as intersecting.
    """
    p = triangulation.all_points[p_idx]
    q = triangulation.all_points[q_idx]

    tri_verts = triangulation.triangle_vertices[current_tri_idx]
    tri_points = triangulation.all_points[tri_verts]

    a, b, c = tri_points
    edges = [
        (a, b, 0, 1),  # edge v0-v1
        (b, c, 1, 2),  # edge v1-v2
        (c, a, 2, 0),  # edge v2-v0
    ]

    for edge_start, edge_end, v_start_local, v_end_local in edges:
        # Orientation tests wrt pq
        pqs_orient = orientation(*p, *q, *edge_start)
        pqe_orient = orientation(*p, *q, *edge_end)

        # Only handle case where BOTH endpoints are collinear
        if not (pqs_orient == 0 and pqe_orient == 0):
            continue

        # Check if edge overlaps with pq
        if not collinear_overlap(p, q, edge_start, edge_end):
            continue

        # Find neighbor triangle that shares this edge and is unvisited
        try:
            neighbor_idx = next(
                t
                for t in triangulation.triangle_neighbors[current_tri_idx]
                if t not in visited
                and t != -1  # don't go outside
                and (
                    tri_verts[v_start_local] in triangulation.triangle_vertices[t]
                    or tri_verts[v_end_local] in triangulation.triangle_vertices[t]
                )
            )
            return neighbor_idx, None
        except StopIteration:
            continue

    return -1, None


def _check_one_endpoint_collinear(
    triangulation: Triangulation,
    current_tri_idx: int,
    p_idx: int,
    q_idx: int,
    visited: set[int],
) -> tuple[int, IntersectedEdge | None]:
    """
    Case B: Exactly one endpoint is collinear with pq.

    When exactly one endpoint of a triangle edge is collinear with the line pq,
    and that point lies within the segment pq, we walk to the next triangle
    containing that vertex.

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation
    current_tri_idx : int
        Current triangle index
    p_idx : int
        Start vertex index of constraint
    q_idx : int
        End vertex index of constraint
    visited : set[int]
        Set of already visited triangle indices

    Returns
    -------
    tuple[int, IntersectedEdge | None]
        (next_triangle_idx, None) or (-1, None) if no match found.
    """
    p = triangulation.all_points[p_idx]
    q = triangulation.all_points[q_idx]

    tri_verts = triangulation.triangle_vertices[current_tri_idx]
    tri_points = triangulation.all_points[tri_verts]

    a, b, c = tri_points
    edges = [
        (a, b),  # edge v0-v1
        (b, c),  # edge v1-v2
        (c, a),  # edge v2-v0
    ]

    for pos, val in enumerate(edges):
        edge_start, edge_end = val

        # Orientation tests wrt pq
        pqs_orient = orientation(*p, *q, *edge_start)
        pqe_orient = orientation(*p, *q, *edge_end)

        # Only handle case where EXACTLY one endpoint is collinear
        if (pqs_orient == 0 and pqe_orient == 0) or (
            pqs_orient != 0 and pqe_orient != 0
        ):  # XOR
            # keeps going only when:
            # - both are zero, or
            # - both are non-zero.
            continue

        # Determine which endpoint is collinear
        point = edge_start if pqs_orient == 0 else edge_end
        point_idx = tri_verts[pos] if pqs_orient == 0 else tri_verts[(pos + 1) % 3]
        if not is_point_in_box(p, q, point):
            continue

        # Walk into the triangle that contains this vertex and is unvisited
        try:
            neighbor_idx = next(
                t
                for t in triangulation.triangle_neighbors[current_tri_idx]
                if t not in visited
                and t != -1  # don't go outside
                and point_idx in triangulation.triangle_vertices[t]
            )
            return neighbor_idx, None
        except StopIteration:
            # We're at a vertex we already visited
            continue

    return -1, None


def find_intersecting_edges(
    triangulation: Triangulation,
    p_idx: int,
    q_idx: int,
    debug: bool = False,
) -> list[IntersectedEdge]:
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
    p_idx : int
        Index of the start point of the constraint segment (vertex index)
    q_idx : int
        Index of the end point of the constraint segment (vertex index)
    debug : bool, optional
        If True, display a visualization of the intersected triangles and edges

    Returns
    -------
    list[tuple[int, int]]
        List of (v1, v2) vertex index pairs of intersected edges, in order from p to q
    """
    p = triangulation.all_points[p_idx]
    q = triangulation.all_points[q_idx]

    # Find triangles containing p and q
    tris_containing_p = np.where(triangulation.triangle_vertices == p_idx)[0]
    tris_containing_q = np.where(triangulation.triangle_vertices == q_idx)[0]
    if set(tris_containing_q) & set(tris_containing_p):
        # edge is already part of the triangulation
        return []

    # pick the first one (it doesn't matter)
    tp = tris_containing_p[0]
    tq = tris_containing_q[0]

    # Walk from tp to tq
    intersecting: list[IntersectedEdge] = []
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

        # Try to find the next triangle by checking each case in order:
        # 1. Case C: Proper crossing (non-collinear intersection)
        # 2. Case A: Collinear overlap (both endpoints on line pq)
        # 3. Case B: One endpoint collinear

        # Try Case C first (proper crossing)
        next_triangle, intersected_edge = _check_proper_crossing(
            triangulation, current, p_idx, q_idx, visited
        )

        # If Case C didn't find anything, try Case A (collinear overlap)
        if next_triangle == -1:
            next_triangle, intersected_edge = _check_collinear_overlap(
                triangulation, current, p_idx, q_idx, visited
            )

        # If Case A didn't find anything, try Case B (one endpoint collinear)
        if next_triangle == -1:
            next_triangle, intersected_edge = _check_one_endpoint_collinear(
                triangulation, current, p_idx, q_idx, visited
            )

        # Record the intersected edge if one was found
        if intersected_edge is not None:
            intersecting.append(intersected_edge)

        if next_triangle == -1:
            raise RuntimeError(
                f"Failed to find next triangle at iteration {iteration}, current={current}"
            )

        current = next_triangle
        visited.add(current)

    logger.debug(f"Found {len(intersecting)} intersecting edges from p={p} to q={q}")

    if debug:
        _plot_intersecting_edges(triangulation, p_idx, q_idx, intersecting)

    return intersecting


def remove_intersecting_edges(
    triangulation: Triangulation,
    p_idx: int,
    q_idx: int,
    edges: list[IntersectedEdge],
) -> list[tuple[int, int]]:
    """
    Remove edges that intersect the constraint edge p_idx-q_idx by edge swapping.

    This implements the edge-flipping algorithm:
    While edges still cross the constraint:
    1. Remove an edge Vk-Vl from the intersecting list
    2. If the two triangles sharing Vk-Vl don't form a strictly convex quadrilateral,
       put the edge back and try another
    3. Otherwise, swap the diagonal:
       - Replace two triangles with two new triangles
       - If the new diagonal Vm-Vn intersects the constraint, add it to intersecting list
       - Otherwise, add it to newly created edges list

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation to modify (modified in-place)
    p_idx : int
        Start vertex index of constraint edge
    q_idx : int
        End vertex index of constraint edge
    edges : list[tuple[int, int]]
        Initial list of edges that intersect the constraint

    Returns
    -------
    list[tuple[int, int]]
        List of newly created edges that don't intersect the constraint
    """
    if not edges:
        return []

    p = triangulation.all_points[p_idx]
    q = triangulation.all_points[q_idx]

    intersecting = edges.copy()
    newly_created = []
    max_iterations = len(edges) * 10  # Safety limit

    for iteration in range(max_iterations):
        if not intersecting:
            break

        # Remove an edge from the intersecting list
        edge = intersecting.pop(0)
        vk_idx = edge.p1
        vl_idx = edge.p2

        # Find the two triangles that share edge vk-vl
        tri1_idx = edge.triangle_1
        tri2_idx = edge.triangle_2

        if tri1_idx == -1 or tri2_idx == -1:
            # TODO: raise Error? What if it's a triangle on the border? Then we shouldn't have an intersecting edge
            logger.warning(f"Could not find triangles sharing edge {vk_idx}-{vl_idx}")
            continue

        # Get the four vertices of the quadrilateral
        tri1_verts = triangulation.triangle_vertices[tri1_idx]
        tri2_verts = triangulation.triangle_vertices[tri2_idx]

        # Use find_shared_edge to get the opposite vertices
        try:
            vk_idx, vl_idx, vm_idx, vn_idx = find_shared_edge(tri1_verts, tri2_verts)
        except Exception as e:
            logger.warning(f"Could not find shared edge: {e}")
            continue

        # Check if the quadrilateral is strictly convex
        if not is_quadrilateral_convex(triangulation, vk_idx, vl_idx, vm_idx, vn_idx):
            # Put the edge back on the list and try another
            intersecting.append(edge)
            continue

        # Swap the diagonal: replace vk-vl with vm-vn
        # Call swap_diagonal from topology.py (point_idx will be auto-determined)
        result = swap_diagonal(triangulation, tri1_idx, tri2_idx)

        # Check if the new diagonal intersects the constraint
        # Use the diagonal vertices returned from swap_diagonal
        new_diag_v1 = triangulation.all_points[result.diagonal_vk]
        new_diag_v2 = triangulation.all_points[result.diagonal_vl]

        # special case: check if the new diagonal coincides with the constraint
        # stop flipping! The constraint is now part of the triangulation
        diag_pts = sorted((new_diag_v1, new_diag_v2), key=lambda x: (x[0], x[1]))
        constraint_pts = sorted((p, q), key=lambda x: (x[0], x[1]))
        if np.allclose(diag_pts, constraint_pts, atol=EPS):
            return []

        if segments_intersect(p, q, new_diag_v1, new_diag_v2):
            # New diagonal still intersects, add to intersecting list
            edge = IntersectedEdge(
                result.diagonal_vk, result.diagonal_vl, tri1_idx, tri2_idx
            )
            intersecting.append(edge)
        else:
            # New diagonal doesn't intersect, add to newly created list
            newly_created.append(
                tuple(sorted((result.diagonal_vk, result.diagonal_vl)))
            )

    if intersecting:
        raise RuntimeError(
            f"Failed to remove all intersecting edges after {max_iterations} iterations. "
            f"{len(intersecting)} edges remain."
        )

    return newly_created


def find_triangles_sharing_edge(
    triangulation: Triangulation, v1_idx: int, v2_idx: int
) -> tuple[int, int]:
    """
    Find the two triangles that share an edge.

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation
    v1_idx : int
        First vertex index of the edge
    v2_idx : int
        Second vertex index of the edge

    Returns
    -------
    tuple[int, int]
        Indices of the two triangles sharing the edge, or (-1, -1) if not found
    """
    edge_vertices = {v1_idx, v2_idx}
    triangles_found = []

    for tri_idx, tri_verts in enumerate(triangulation.triangle_vertices):
        tri_verts_set = set(tri_verts)
        if edge_vertices.issubset(tri_verts_set):
            triangles_found.append(tri_idx)
            if len(triangles_found) == 2:
                return triangles_found[0], triangles_found[1]

    if len(triangles_found) == 1:
        return triangles_found[0], -1
    return -1, -1


def is_quadrilateral_convex(
    triangulation: Triangulation,
    vk_idx: int,
    vl_idx: int,
    vm_idx: int,
    vn_idx: int,
) -> bool:
    """
    Check if four vertices form a strictly convex quadrilateral.

    The quadrilateral is formed by two triangles sharing edge vk-vl,
    with vm on one side and vn on the other.

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation
    vk_idx, vl_idx : int
        Indices of the shared edge vertices
    vm_idx, vn_idx : int
        Indices of the opposite vertices in each triangle

    Returns
    -------
    bool
        True if the quadrilateral is strictly convex
    """
    vk = triangulation.all_points[vk_idx]
    vl = triangulation.all_points[vl_idx]
    vm = triangulation.all_points[vm_idx]
    vn = triangulation.all_points[vn_idx]

    # Check that both vm and vn are on opposite sides of edge vk-vl
    o_vm = orientation(*vk, *vl, *vm)
    o_vn = orientation(*vk, *vl, *vn)

    if o_vm * o_vn >= 0:
        # Same side or collinear - not convex
        return False

    # Check that vk and vl are on opposite sides of edge vm-vn
    o_vk = orientation(*vm, *vn, *vk)
    o_vl = orientation(*vm, *vn, *vl)

    if o_vk * o_vl >= 0:
        # Same side or collinear - not convex
        return False

    return True


def _insert_single_constraint(
    triangulation: Triangulation,
    p_idx: int,
    q_idx: int,
) -> bool:
    """
    Insert a single constraint edge into the triangulation.

    This is an internal function. Use add_constraints() instead.

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
    # Find all intersecting edges
    intersected_edges = find_intersecting_edges(triangulation, p_idx, q_idx)

    if not intersected_edges:
        logger.info(f"Constraint edge {p_idx}-{q_idx} does not intersect any edges")
        return True

    logger.info(
        f"Inserting constraint {p_idx}-{q_idx}, removing {len(intersected_edges)} edges"
    )

    # Remove intersected edges by swapping
    newly_created = remove_intersecting_edges(
        triangulation, p_idx, q_idx, intersected_edges
    )

    logger.info(
        f"Constraint {p_idx}-{q_idx} inserted. Created {len(newly_created)} new edges."
    )

    # Restore Delaunay triangulation
    # Repeat until no more swaps occur
    constraint_edge = {p_idx, q_idx}
    max_iterations = len(newly_created) * 10
    swapped = False
    for iteration in range(max_iterations):
        swapped = False
        new_edges = []

        for vk_idx, vl_idx in newly_created:
            # Skip if this is the constraint edge
            if {vk_idx, vl_idx} == constraint_edge:
                new_edges.append((vk_idx, vl_idx))
                continue

            # Find the two triangles sharing this edge
            tri1_idx, tri2_idx = find_triangles_sharing_edge(
                triangulation, vk_idx, vl_idx
            )

            if tri1_idx == -1 or tri2_idx == -1:
                # Edge is on the boundary, keep it
                new_edges.append((vk_idx, vl_idx))
                continue

            # Get the opposite vertices
            tri1_verts = triangulation.triangle_vertices[tri1_idx]
            tri2_verts = triangulation.triangle_vertices[tri2_idx]
            _, _, vm_idx, vn_idx = find_shared_edge(tri1_verts, tri2_verts)

            # Check Delaunay criterion
            # Get points
            vk = triangulation.all_points[vk_idx]
            vl = triangulation.all_points[vl_idx]
            vm = triangulation.all_points[vm_idx]
            vn = triangulation.all_points[vn_idx]

            # Check if vm is inside circumcircle of triangle (vk, vl, vn)
            # incircle_test returns positive if point is inside
            if incircle_test(*vm, *vk, *vl, *vn) > 0:
                # Delaunay criterion violated, swap the diagonal
                swap_diagonal(triangulation, tri1_idx, tri2_idx)
                # Replace edge in list with the new diagonal
                new_edges.append(tuple(sorted((vm_idx, vn_idx))))
                swapped = True
            else:
                # Delaunay criterion satisfied, keep the edge
                new_edges.append((vk_idx, vl_idx))

        newly_created = new_edges

        if not swapped:
            logger.debug(
                f"Delaunay restoration complete after {iteration + 1} iteration(s)"
            )
            break

    if swapped:
        logger.warning(
            f"Delaunay restoration did not converge after {max_iterations} iterations"
        )

    return True


def add_constraints(
    triangulation: Triangulation,
    constraints: list[tuple[int, int]],
) -> bool:
    """
    Add constraint edge(s) to a triangulation.

    This modifies the triangulation to include the constraint edges,
    creating a Constrained Delaunay Triangulation (CDT).

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation to constrain (modified in-place)
    constraints : list[tuple[int, int]] or tuple[int, int]
        Either a single constraint edge as (p_idx, q_idx) or a list of
        constraint edges as [(v1_idx, v2_idx), ...].
        Vertex indices refer to indices in triangulation.all_points.

    Returns
    -------
    bool
        True if all constraints were successfully inserted, False otherwise

    Notes
    -----
    The resulting triangulation will:
    - Contain all the constraint edges
    - Be as close to Delaunay as possible while respecting constraints
    - May have some triangles that violate the Delaunay property if necessary
      to accommodate the constraints

    Example
    --------
    >>> add_constraints(tri, [(0, 5), (1, 6), (2, 7)])
    """
    logger.info(f"Adding {len(constraints)} constraint(s) to triangulation")
    all_success = True
    for i, (v1_idx, v2_idx) in enumerate(constraints):
        logger.debug(
            f"Processing constraint {i + 1}/{len(constraints)}: {v1_idx}-{v2_idx}"
        )
        success = _insert_single_constraint(triangulation, v1_idx, v2_idx)
        if not success:
            logger.warning(f"Failed to insert constraint {v1_idx}-{v2_idx}")
            all_success = False

    return all_success
