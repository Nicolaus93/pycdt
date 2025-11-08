"""Query functions for triangulation structures."""

import numpy as np
from numpy.typing import NDArray
from shewchuk import orientation

from pycdt.geometry import PointInTriangle, is_point_in_box, point_inside_triangle
from pycdt.utils import Vec2d, Triangle
from pycdt.build import find_containing_triangle


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
    o1 = orientation(q1[0], q1[1], q2[0], q2[1], p1[0], p1[1])
    o2 = orientation(q1[0], q1[1], q2[0], q2[1], p2[0], p2[1])
    o3 = orientation(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1])
    o4 = orientation(p1[0], p1[1], p2[0], p2[1], q2[0], q2[1])

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


def segment_intersects_triangle_interior(
    p: Vec2d | NDArray[np.floating],
    q: Vec2d | NDArray[np.floating],
    triangle: Triangle | NDArray[np.floating],
    eps: float = 1e-9,
) -> bool:
    """
    Check if segment pq intersects the interior of a triangle.

    A segment intersects the triangle interior if:
    - It intersects any edge of the triangle, OR
    - One or both endpoints are inside the triangle

    Parameters
    ----------
    p, q : NDArray[np.floating]
        Segment endpoints (shape (2,))
    triangle : NDArray[np.floating]
        Triangle vertices (shape (3, 2))
    eps : float
        Tolerance for geometric tests

    Returns
    -------
    bool
        True if segment intersects triangle interior
    """
    a, b, c = triangle

    # Check if either endpoint is inside or on the triangle
    p_status, _ = point_inside_triangle(triangle, p, eps)
    q_status, _ = point_inside_triangle(triangle, q, eps)

    if p_status in (PointInTriangle.inside, PointInTriangle.edge):
        return True
    if q_status in (PointInTriangle.inside, PointInTriangle.edge):
        return True

    # Check if segment intersects any edge of the triangle
    if segments_intersect(p, q, a, b):
        return True
    if segments_intersect(p, q, b, c):
        return True
    if segments_intersect(p, q, c, a):
        return True

    return False


def segment_triangle_walk(
    triangulation,
    p: Vec2d | NDArray[np.floating],
    q: Vec2d | NDArray[np.floating],
    start_triangle_idx: int | None = None,
) -> list[int]:
    """
    Find all triangles whose interiors are intersected by segment pq.

    Uses a triangle walking algorithm: starts at a triangle near p,
    then walks through the triangulation following the segment until
    reaching q.

    Parameters
    ----------
    triangulation : Triangulation
        The triangulation to search
    p : Vec2d | NDArray[np.floating]
        Start point of segment (tuple or array of shape (2,))
    q : Vec2d | NDArray[np.floating]
        End point of segment (tuple or array of shape (2,))
    start_triangle_idx : int | None
        Optional starting triangle index. If None, will search for a triangle containing p.

    Returns
    -------
    list[int]
        List of triangle indices whose interiors are intersected by segment pq.
        Returns empty list if no triangles are intersected.
    """
    # Convert to arrays if tuples
    if isinstance(p, tuple):
        p = np.array(p)
    if isinstance(q, tuple):
        q = np.array(q)

    # Find starting triangle
    if start_triangle_idx is None:
        # Validate last_triangle_idx before using it
        last_idx = triangulation.last_triangle_idx
        if last_idx >= len(triangulation.triangle_vertices) or last_idx < 0:
            last_idx = 0

        try:
            result = find_containing_triangle(triangulation, p, last_idx)
            start_triangle_idx = result.idx
        except (ValueError, IndexError):
            # Finalized triangulations may have broken neighbors, try from index 0
            if last_idx != 0:
                try:
                    result = find_containing_triangle(triangulation, p, 0)
                    start_triangle_idx = result.idx
                except (ValueError, IndexError):
                    return []
            else:
                return []

    if start_triangle_idx == -1:
        return []

    result = []
    visited = set()
    current = start_triangle_idx

    # Walk through triangulation following the segment
    max_iterations = len(triangulation.triangle_vertices)
    for _ in range(max_iterations):
        if current in visited or current == -1:
            break

        visited.add(current)

        tri_vertices = triangulation.triangle_vertices[current]
        tri_points = triangulation.all_points[tri_vertices]

        # Check if segment intersects this triangle
        if segment_intersects_triangle_interior(p, q, tri_points):
            result.append(current)

        # Check if we've reached the end point
        q_status, _ = point_inside_triangle(tri_points, q)
        if q_status in (
            PointInTriangle.inside,
            PointInTriangle.vertex,
            PointInTriangle.edge,
        ):
            break

        # Find which edge the segment exits through
        a, b, c = tri_points
        edges = [(a, b, 2), (b, c, 0), (c, a, 1)]  # (v1, v2, opposite_vertex_idx)

        next_triangle = -1
        for edge_start, edge_end, opposite_idx in edges:
            if segments_intersect(p, q, edge_start, edge_end):
                neighbor_idx = triangulation.triangle_neighbors[current][opposite_idx]
                # Validate neighbor_idx is within bounds (important after finalization)
                if (
                    neighbor_idx != -1
                    and neighbor_idx < max_iterations
                    and neighbor_idx not in visited
                ):
                    # Check if the segment actually intersects the neighboring triangle
                    neighbor_tri_verts = triangulation.triangle_vertices[neighbor_idx]
                    neighbor_tri_points = triangulation.all_points[neighbor_tri_verts]
                    if segment_intersects_triangle_interior(p, q, neighbor_tri_points):
                        next_triangle = neighbor_idx
                        break

        current = next_triangle

    return result
