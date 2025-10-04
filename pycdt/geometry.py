from enum import Enum, auto

import numpy as np
from numpy._typing import NDArray

EPS = 1e-12


class PointInTriangle(Enum):
    vertex = auto()
    edge = auto()
    inside = auto()
    outside = auto()


def point_inside_triangle(
    triangle: NDArray[np.floating],
    point: NDArray[np.floating],
    eps: float = 1e-3,
    debug: bool = False,
) -> tuple[PointInTriangle, int | None]:
    """
    Check if a point lies inside a triangle using the determinant method with numpy.
    - triangle: List of three vertices [(x1, y1), (x2, y2), (x3, y3)].
    - point: The point to check (x, y).
    - debug: If True, plot the triangle and the point.
    Returns True if the point is inside the triangle, False otherwise.
    """
    a, b, c = np.array(triangle)

    det1 = orient2d(a, b, point)
    det2 = orient2d(b, c, point)
    det3 = orient2d(c, a, point)

    # On vertex?
    if np.allclose(point, a, atol=eps):
        return PointInTriangle.vertex, None
    if np.allclose(point, b, atol=eps):
        return PointInTriangle.vertex, None
    if np.allclose(point, c, atol=eps):
        return PointInTriangle.vertex, None

    # On which edge?
    if is_point_on_segment(a, b, point, eps):
        return PointInTriangle.edge, 2  # opposite vertex 2
    if is_point_on_segment(b, c, point, eps):
        return PointInTriangle.edge, 0  # opposite vertex 0
    if is_point_on_segment(c, a, point, eps):
        return PointInTriangle.edge, 1  # opposite vertex 1

    # Inside / outside via signs (boundary counted inside)
    has_neg = (det1 < -eps) or (det2 < -eps) or (det3 < -eps)
    has_pos = (det1 > eps) or (det2 > eps) or (det3 > eps)

    if debug:
        import matplotlib.pyplot as plt

        tri_closed = np.vstack([triangle, triangle[0]])
        plt.figure()
        plt.plot(tri_closed[:, 0], tri_closed[:, 1], "b-")
        plt.scatter(*point, color="red")
        plt.axis("equal")
        plt.show()

    if not (has_neg and has_pos):
        # it's zero
        return PointInTriangle.inside, None
    return PointInTriangle.outside, None


def is_point_on_segment(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    p: NDArray[np.floating],
    eps: float = 1e-12,
) -> bool:
    # colinear?
    if abs(orient2d(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def classify_point_in_triangle(
    tri: NDArray[np.floating], p: NDArray[np.floating], eps: float = 1e-6
) -> tuple[PointInTriangle, int | None]:
    # tri is (3,2) in the same order as triangulation.triangle_vertices for that triangle
    a, b, c = tri
    d1 = orient2d(a, b, p)
    d2 = orient2d(b, c, p)
    d3 = orient2d(c, a, p)

    # On vertex?
    if np.allclose(p, a, atol=eps):
        return PointInTriangle.vertex, None
    if np.allclose(p, b, atol=eps):
        return PointInTriangle.vertex, None
    if np.allclose(p, c, atol=eps):
        return PointInTriangle.vertex, None

    # On which edge?
    if is_point_on_segment(a, b, p, eps):
        return PointInTriangle.edge, 2  # opposite vertex 2
    if is_point_on_segment(b, c, p, eps):
        return PointInTriangle.edge, 0  # opposite vertex 0
    if is_point_on_segment(c, a, p, eps):
        return PointInTriangle.edge, 1  # opposite vertex 1

    # Inside / outside via signs (boundary counted inside)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    if not (has_neg and has_pos):
        return PointInTriangle.inside, None
    return PointInTriangle.outside, None


def orient2d(pa: NDArray, pb: NDArray, pc: NDArray) -> float:
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


def is_point_inside(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
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


def is_inside_domain(
    point: tuple[float, float],
    poly_outer: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]],
) -> bool:
    x, y = point
    if not is_point_inside(x, y, poly_outer):
        return False
    for hole in holes:
        if is_point_inside(x, y, hole):
            return False
    return True
