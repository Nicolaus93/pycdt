"""
Test script for incremental point insertion.
This demonstrates the new functionality of adding points in multiple stages.
"""

import numpy as np
from pycdt.build import create_triangulation, update_triangulation

# Stage 1: Create initial triangulation (without finalizing)
print("Stage 1: Creating initial triangulation with 5 points")
initial_points = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ]
)

tri = create_triangulation(initial_points, finalize=False)
print(f"  Triangulation has {len(tri.all_points)} points (including supertriangle)")
print(f"  Number of triangles: {len(tri.triangle_vertices)}")

# Optional: Plot the triangulation
print("\nPlotting initial triangulation...")
tri.plot(show=True, exclude_super_t=True, title="Stage 1: Initial Triangulation")

# Stage 2: Add more points
print("\nStage 2: Adding 4 more points")
new_points_1 = np.array(
    [
        [0.25, 0.25],
        [0.75, 0.25],
        [0.75, 0.75],
        [0.25, 0.75],
    ]
)

tri = update_triangulation(tri, new_points_1, finalize=False)
print(f"  Triangulation has {len(tri.all_points)} points (including supertriangle)")
print(f"  Number of triangles: {len(tri.triangle_vertices)}")

# Plot after second stage
print("\nPlotting after adding second batch...")
tri.plot(show=True, exclude_super_t=True, title="Stage 2: After Adding 4 Points")

# Stage 3: Add even more points (outside original bounds) and finalize
print("\nStage 3: Adding 4 points outside original bounds and finalizing")
new_points_2 = np.array(
    [
        [-0.5, -0.5],
        [1.5, -0.5],
        [1.5, 1.5],
        [-0.5, 1.5],
    ]
)

tri = update_triangulation(tri, new_points_2, finalize=True)
print(f"  Final triangulation has {len(tri.all_points)} points")
print(f"  Number of triangles: {len(tri.triangle_vertices)}")

# Plot final result
print("\nPlotting final triangulation...")
tri.plot(show=True, title="Final Triangulation", point_labels=True)

print("\n✓ Test completed successfully!")
print("\nSummary:")
print("  - Started with 5 points")
print("  - Added 4 points in stage 2")
print("  - Added 4 points in stage 3 and finalized")
print(
    f"  - Final: {len(tri.all_points)} points, {len(tri.triangle_vertices)} triangles"
)
