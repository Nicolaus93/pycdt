from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray
from shewchuk import orientation

from pycdt.utils import EPS, Vec2d, Triangle


class PointInTriangle(Enum):
    vertex = auto()
    edge = auto()
    inside = auto()
    outside = auto()


def is_point_in_box(
    a: Vec2d,
    b: Vec2d,
    p: Vec2d,
    eps: float = EPS,
) -> bool:
    # check if p is within the bounding box of [a, b]
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def point_inside_triangle(
    triangle: Triangle,
    point: Vec2d,
    eps: float = 1e-9,
    debug: bool = False,
) -> tuple[PointInTriangle, int | None]:
    """
    Classify a point relative to a triangle using Shewchuk's exact orientation predicate.

    Parameters
    ----------
    triangle : NDArray[np.floating]
        Array of shape (3, 2) with triangle vertices [A, B, C].
    point : NDArray[np.floating]
        Array of shape (2,) representing the query point.
    eps : float, optional
        Tolerance for vertex coordinate equality (not used in orientation).
    debug : bool, optional
        If True, visualize the triangle and point.

    Returns
    -------
    (PointInTriangle, Optional[int])
        - Classification (inside, edge, vertex, outside)
        - Index of the vertex opposite to the intersected edge, if applicable.
    """
    a, b, c = triangle

    # Check vertex match (geometric equality)
    if np.allclose(point, a, atol=eps):
        return PointInTriangle.vertex, 0
    if np.allclose(point, b, atol=eps):
        return PointInTriangle.vertex, 1
    if np.allclose(point, c, atol=eps):
        return PointInTriangle.vertex, 2

    # Exact orientation results: -1 (CW), 0 (collinear), +1 (CCW)
    o1 = orientation(a[0], a[1], b[0], b[1], point[0], point[1])
    o2 = orientation(b[0], b[1], c[0], c[1], point[0], point[1])
    o3 = orientation(c[0], c[1], a[0], a[1], point[0], point[1])

    # Check if it's on edge
    # To test whether the point lies on the segment [A,B], you need two checks:
    #     1. orientation(a, b, p) == 0 → the point is collinear.
    #     2. The point’s coordinates lie between A and B (i.e., it’s within the segment’s bounding box).
    if o1 == 0 and is_point_in_box(a, b, point):
        return PointInTriangle.edge, 2  # opposite vertex 2
    if o2 == 0 and is_point_in_box(b, c, point):
        return PointInTriangle.edge, 0  # opposite vertex 0
    if o3 == 0 and is_point_in_box(c, a, point):
        return PointInTriangle.edge, 1  # opposite vertex 1

    # Inside / outside test
    # If all orientations are the same sign, the point is inside
    if (
        (o1 == o2 == o3)
        or (o1 >= 0 and o2 >= 0 and o3 >= 0)
        or (o1 <= 0 and o2 <= 0 and o3 <= 0)
    ):
        return PointInTriangle.inside, None

    # Otherwise, outside
    if debug:
        import matplotlib.pyplot as plt

        tri_closed = np.vstack([triangle, triangle[0]])  # type: ignore[reportArgumentType]
        plt.figure()
        plt.plot(tri_closed[:, 0], tri_closed[:, 1], "b-")
        plt.scatter(*point, color="red")  # type: ignore[reportArgumentType]
        plt.axis("equal")
        plt.title("Outside Triangle")
        plt.show()

    return PointInTriangle.outside, None


def orient2d(pa: Vec2d, pb: Vec2d, pc: Vec2d) -> float:
    """
    Shewchuk's robust 2D orientation predicate.
    Returns > 0 if points are in counterclockwise order
    Returns < 0 if points are in clockwise order
    Returns = 0 if points are collinear

    This is a simplified version - for full robustness, use Shewchuk's exact arithmetic
    """
    detleft = (pa[0] - pc[0]) * (pb[1] - pc[1])
    detright = (pa[1] - pc[1]) * (pb[0] - pc[0])
    det = detleft - detright
    return det


def ensure_ccw_triangle(vertices: NDArray, points: NDArray) -> NDArray:
    """Ensure triangle vertices are in counterclockwise order"""
    p0, p1, p2 = points[vertices]
    if orient2d(p0, p1, p2) < 0:
        # Swap vertices to make counterclockwise
        return np.array([vertices[0], vertices[2], vertices[1]])
    return vertices


def is_point_inside_polygon(
    x: float, y: float, poly: list[tuple[float, float]]
) -> bool:
    """
    From https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule
    Determine if the point is on the path, corner, or boundary of the polygon

    Args:
      x -- The x coordinates of point.
      y -- The y coordinates of point.
      poly -- a list of tuples [(x, y), (x, y), ...]

    Returns:
      True if the point is in the path or is a corner or on the boundary
    """
    inside = False
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[i - 1]
        if (x == x0) and (y == y0):
            # point is a corner
            return True
        # Check where the ray intersects the edge horizontally
        if (y0 > y) != (y1 > y):
            # determines the relative position of the point (x, y) to the edge (x0,y0)→(x1,y1) using cross product
            # between:
            # - Vector A: from vertex (x0,y0) to point (x,y)
            # - Vector B: from vertex (x0,y0) to vertex (x1,y1)
            # slope > 0 -> Point is to the left of the edge
            # slope < 0 -> Point is to the right of the edge
            # slope == 0 -> Point lies exactly on the edge (colinear)
            cross = (x - x0) * (y1 - y0) - (x1 - x0) * (y - y0)
            if cross == 0:
                # TODO: point is on boundary, what to return?
                return True
            if (cross < 0) != (y1 < y0):
                inside = not inside
    return inside
