"""Example: Add multiple constraints forming a polygon boundary.

This example demonstrates inserting multiple constraint edges that form
the boundary of an L-shaped polygon with interior points.
"""

import numpy as np

from pycdt.build import triangulate
from pycdt.constrained import add_constraints


def main():
    """Example: Polygon boundary constraints."""
    print("\n" + "=" * 70)
    print("POLYGON BOUNDARY CONSTRAINTS EXAMPLE")
    print("=" * 70 + "\n")

    # Create an L-shaped polygon
    # Outer boundary vertices (going counter-clockwise)
    boundary = np.array(
        [
            [0.0, 0.0],  # 0
            [3.0, 0.0],  # 1
            [3.0, 2.0],  # 2
            [1.0, 2.0],  # 3
            [1.0, 3.0],  # 4
            [0.0, 3.0],  # 5
            [2.0, 0.0],  # 6
            [1.0, 0.0],  # 7
        ]
    )

    # Add some interior points
    interior = np.array(
        [
            [0.5, 0.5],  # 6
            [2.5, 0.5],  # 7
            [2.0, 1.5],  # 8
            [0.5, 2.5],  # 9
        ]
    )

    points = np.vstack([boundary, interior])
    print(f"Number of points: {len(points)}")
    print(f"  Boundary vertices: {len(boundary)}")
    print(f"  Interior points: {len(interior)}")

    # Create initial triangulation
    tri = triangulate(points)
    print(f"\nNumber of triangles: {len(tri.triangle_vertices)}")

    # Plot before constraints
    print("\nPlotting triangulation before constraints...")
    tri.plot(
        show=False,
        title="L-Shape Before Constraints",
        point_labels=True,
        fontsize=8,
    )

    # Define boundary constraints
    constraints = [
        # (5, 0),
        # (1, 2),
        # (5, 3),
        # (0, 1),
        (2, 3),  # Horizontal middle edge
    ]

    print(
        f"\nAdding {len(constraints)} constraint edge(s) forming part of the boundary:"
    )
    for i, (v1, v2) in enumerate(constraints):
        print(f"  Constraint {i + 1}: {v1} -> {v2}")

    # Add all constraints
    print("\nAdding constraints...")
    success = add_constraints(tri, constraints)
    if success:
        print(f"  ✓ Successfully added all {len(constraints)} constraint(s)")
        successful_constraints = constraints
    else:
        print("  ⚠ Some constraints may have failed")
        successful_constraints = constraints

    print(f"\nAdded {len(successful_constraints)}/{len(constraints)} constraints")

    # Plot after constraints
    print("\nPlotting triangulation after constraints...")
    tri.plot(
        show=True,
        title=f"L-Shape After {len(successful_constraints)} Constraints",
        point_labels=True,
        fontsize=8,
        constraints=successful_constraints,
    )

    print(f"\nAfter constraint insertion: {len(tri.triangle_vertices)} triangles")
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
