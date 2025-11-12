"""Example script demonstrating the debug visualization for find_intersecting_edges.

The visualization shows:
- Full triangulation with all triangles numbered at centroids (dark blue numbers)
- All vertices numbered next to each vertex (dark green numbers)
- Intersected triangles highlighted in orange with dark red numbers
- Intersected edges highlighted in red
- Constraint segment shown as a thick blue line
- Start point (p) in green, end point (q) in purple
"""

import numpy as np

from pycdt.build import triangulate
from pycdt.constrained import find_intersecting_edges


def main():
    # Create a simple grid of points
    n = 5
    x = np.linspace(0, 4, n)
    y = np.linspace(0, 4, n)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # Triangulate
    tri = triangulate(points)

    # Add a diagonal constraint from bottom-left to top-right
    p_idx = 0  # (0, 0)
    q_idx = len(points) - 1  # (4, 4)

    print(f"Finding intersecting edges for constraint {p_idx} -> {q_idx}")
    print(f"Point p: {points[p_idx]}")
    print(f"Point q: {points[q_idx]}")
    print(f"\nTotal triangles: {len(tri.triangle_vertices)}")
    print(f"Total vertices: {len(tri.all_points)}")

    # Call with debug=True to show visualization with numbered triangles and vertices
    edges = find_intersecting_edges(tri, p_idx, q_idx, debug=True)

    print(f"\nFound {len(edges)} intersecting edges:")
    for i, edge in enumerate(edges):
        print(
            f"  {i + 1}. Edge {edge.p1}-{edge.p2}: "
            f"triangles ({edge.triangle_1}, {edge.triangle_2})"
        )


if __name__ == "__main__":
    main()
