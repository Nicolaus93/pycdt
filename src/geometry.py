import numpy as np
from numpy._typing import NDArray

EPS = 1e-12


def point_inside_triangle(
    triangle: NDArray[np.floating],
    point: NDArray[np.floating],
    debug: bool = False,
) -> bool:
    """
    Check if a point lies inside a triangle using the determinant method with numpy.
    - triangle: List of three vertices [(x1, y1), (x2, y2), (x3, y3)].
    - point: The point to check (x, y).
    - debug: If True, plot the triangle and the point.
    Returns True if the point is inside the triangle, False otherwise.
    """
    a, b, c = np.array(triangle)
    p = np.array(point)

    # Compute vectors
    ab = b - a
    ap = p - a

    bc = c - b
    bp = p - b

    ca = a - c
    cp = p - c

    det1 = ab[0] * ap[1] - ab[1] * ap[0]
    det2 = bc[0] * bp[1] - bc[1] * bp[0]
    det3 = ca[0] * cp[1] - ca[1] * cp[0]
    determinants = np.array([det1, det2, det3])

    # Check if all determinants have the same sign (inside the triangle)
    inside = np.all(determinants > 0) or np.all(determinants < 0)

    if debug:
        import matplotlib.pyplot as plt

        triangle_with_closure = np.vstack([triangle, triangle[0]])  # Close the triangle
        plt.figure()
        plt.plot(
            triangle_with_closure[:, 0],
            triangle_with_closure[:, 1],
            "b-",
            label="Triangle",
        )
        plt.scatter(*point, color="red", label="Point")
        plt.axis("equal")
        plt.legend()
        plt.title("Point Inside Triangle" if inside else "Point Outside Triangle")
        plt.show()

    return bool(inside)


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
