from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from loguru import logger

from pycdt.debug_utils import _plot_intersecting_edges
from pycdt.delaunay import Triangulation
from pycdt.build import find_containing_triangle
from pycdt.geometry import (
    point_inside_triangle,
    PointInTriangle,
    is_point_in_box,
    ensure_ccw_triangle,
)
from pycdt.utils import EPS, Vec2d
from pycdt.topology import find_shared_edge, find_vertex_position
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


@dataclass(frozen=True)
class IntersectedEdge:
    p1: int
    p2: int
    triangle_1: int
    triangle_2: int


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
    tp = find_triangle_for_point(triangulation=triangulation, point=p)
    tq = find_triangle_for_point(triangulation=triangulation, point=q)

    # If both points in same triangle, it means pq is an edge. Return that triangle
    if tp == tq:
        return [IntersectedEdge(p_idx, q_idx, tp, tp)]

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
            # case when both p and q are in the same triangle is already by if tp == tq: previously
            # case when only q is equal to edge_start or edge_end is covered by point_inside_triangle(tri_points, q)
            if np.allclose(p, edge_start, atol=EPS) or np.allclose(
                p, edge_end, atol=EPS
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
                    # TODO: what if we walk along the border?
                    #     _   _
                    #   /\  /\  /\
                    # P ---.---.--- Q
                    raise RuntimeError("Ended up outside triangulation")
                if neighbor_idx in visited:
                    # if it's in visited, it could be the one where we're "coming" from
                    continue

                # record intersected edge
                v1, v2 = sorted((tri_verts[i], tri_verts[(i + 1) % 3]))
                intersecting.append(IntersectedEdge(v1, v2, current, neighbor_idx))
                next_triangle = neighbor_idx
                break

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
        new_tri1_idx, new_tri2_idx = swap_diagonal(
            triangulation, tri1_idx, tri2_idx, vk_idx, vl_idx, vm_idx, vn_idx
        )

        # Check if the new diagonal vm-vn intersects the constraint
        vm = triangulation.all_points[vm_idx]
        vn = triangulation.all_points[vn_idx]

        if segments_intersect(p, q, vm, vn):
            # New diagonal still intersects, add to intersecting list
            edge = IntersectedEdge(vm_idx, vn_idx, new_tri1_idx, new_tri2_idx)
            intersecting.append(edge)
        else:
            # New diagonal doesn't intersect, add to newly created list
            newly_created.append(tuple(sorted((vm_idx, vn_idx))))

    if intersecting:
        logger.warning(
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


def swap_diagonal(
    triangulation: Triangulation,
    tri1_idx: int,
    tri2_idx: int,
    vk_idx: int,
    vl_idx: int,
    vm_idx: int,
    vn_idx: int,
) -> tuple[int, int]:
    """
    Swap the diagonal of a quadrilateral formed by two triangles.

    This operation (also called edge flip) replaces the shared edge vk-vl with a new edge vm-vn,
    where vm and vn are the vertices opposite to the shared edge in each triangle.

    Before swap:
        Triangle 1: (vk, vl, vm)
        Triangle 2: (vk, vl, vn)
        Shared edge: vk-vl

    After swap:
        Triangle 1: (vm, vn, vk)
        Triangle 2: (vm, vl, vn)
        Shared edge: vm-vn

    The function updates:
    - Triangle vertices to form the new triangles (ensuring CCW orientation)
    - Triangle neighbor relationships for both modified triangles
    - Neighbor references in adjacent triangles pointing to the modified triangles

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation to modify (modified in-place)
    tri1_idx, tri2_idx : int
        Indices of the two triangles sharing edge vk-vl
    vk_idx, vl_idx : int
        Indices of the old diagonal vertices (shared edge to be removed)
    vm_idx, vn_idx : int
        Indices of the opposite vertices in each triangle (new diagonal endpoints)

    Returns
    -------
    tuple[int, int]
        The indices (tri1_idx, tri2_idx) of the two triangles that now share the new edge vm-vn.
        These are the same triangle indices that were passed in, but with updated geometry.
    """
    points = triangulation.all_points
    vertices = triangulation.triangle_vertices
    neighbors = triangulation.triangle_neighbors

    # Save old neighbor information
    old_tri1_neighbors = neighbors[tri1_idx].copy()
    old_tri2_neighbors = neighbors[tri2_idx].copy()
    old_tri1_verts = vertices[tri1_idx].copy()
    old_tri2_verts = vertices[tri2_idx].copy()

    # Create new triangles after edge flip
    # New tri1: vm, vn, vk (ensure CCW)
    # New tri2: vm, vl, vn (ensure CCW)
    new_tri1_vertices = ensure_ccw_triangle(np.array([vm_idx, vn_idx, vk_idx]), points)
    new_tri2_vertices = ensure_ccw_triangle(np.array([vm_idx, vl_idx, vn_idx]), points)

    # Update triangle vertices
    vertices[tri1_idx] = new_tri1_vertices
    vertices[tri2_idx] = new_tri2_vertices

    # Update neighbors for the new triangles
    # For tri1 (vm, vn, vk):
    # - Edge opposite to vm (edge vn-vk): neighbor is the old neighbor of tri2 opposite to vl
    # - Edge opposite to vn (edge vk-vm): neighbor is the old neighbor of tri2 opposite to vk
    # - Edge opposite to vk (edge vm-vn): neighbor is tri2

    # For tri2 (vm, vl, vn):
    # - Edge opposite to vm (edge vl-vn): neighbor is the old neighbor of tri1 opposite to vk
    # - Edge opposite to vl (edge vn-vm): neighbor is the old neighbor of tri1 opposite to vl
    # - Edge opposite to vn (edge vm-vl): neighbor is tri1

    # Find neighbor opposite to vl in old tri2
    pos_vl_in_tri2 = find_vertex_position(old_tri2_verts, vl_idx)
    neighbor_opp_vl = old_tri2_neighbors[pos_vl_in_tri2]

    # Find neighbor opposite to vk in old tri2
    pos_vk_in_tri2 = find_vertex_position(old_tri2_verts, vk_idx)
    neighbor_opp_vk_tri2 = old_tri2_neighbors[pos_vk_in_tri2]

    # Find neighbor opposite to vk in old tri1
    pos_vk_in_tri1 = find_vertex_position(old_tri1_verts, vk_idx)
    neighbor_opp_vk_tri1 = old_tri1_neighbors[pos_vk_in_tri1]

    # Find neighbor opposite to vl in old tri1
    pos_vl_in_tri1 = find_vertex_position(old_tri1_verts, vl_idx)
    neighbor_opp_vl_tri1 = old_tri1_neighbors[pos_vl_in_tri1]

    # Set neighbors for new tri1 (vm, vn, vk)
    tri1_neighbors = np.array([-1, -1, -1], dtype=np.int32)
    for i in range(3):
        curr_v = new_tri1_vertices[i]
        if curr_v == vm_idx:
            tri1_neighbors[i] = neighbor_opp_vl
        elif curr_v == vn_idx:
            tri1_neighbors[i] = neighbor_opp_vk_tri2
        elif curr_v == vk_idx:
            tri1_neighbors[i] = tri2_idx

    # Set neighbors for new tri2 (vm, vl, vn)
    tri2_neighbors = np.array([-1, -1, -1], dtype=np.int32)
    for i in range(3):
        curr_v = new_tri2_vertices[i]
        if curr_v == vm_idx:
            tri2_neighbors[i] = neighbor_opp_vk_tri1
        elif curr_v == vl_idx:
            tri2_neighbors[i] = neighbor_opp_vl_tri1
        elif curr_v == vn_idx:
            tri2_neighbors[i] = tri1_idx

    neighbors[tri1_idx] = tri1_neighbors
    neighbors[tri2_idx] = tri2_neighbors

    # Update references in neighboring triangles
    def update_neighbor_reference(
        neighbor_idx: int, old_triangle: int, new_triangle: int
    ):
        if neighbor_idx != -1:
            mask = neighbors[neighbor_idx] == old_triangle
            if np.any(mask):
                neighbors[neighbor_idx][mask] = new_triangle

    # Update neighbor references
    update_neighbor_reference(neighbor_opp_vk_tri2, tri2_idx, tri1_idx)
    update_neighbor_reference(neighbor_opp_vl_tri1, tri1_idx, tri2_idx)

    return tri1_idx, tri2_idx


def insert_constraint_edge(
    triangulation: Triangulation,
    p_idx: int,
    q_idx: int,
) -> bool:
    """
    Insert a constraint edge into the triangulation.

    This function:
    1. Finds all edges intersected by the constraint
    2. Removes those edges by edge swapping (edge flipping)
    3. Returns True if the constraint was successfully inserted

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

    return True


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
