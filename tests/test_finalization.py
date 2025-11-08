"""Tests for triangulation finalization (super-triangle removal)."""

import numpy as np

from pycdt.build import triangulate, find_containing_triangle


class TestFinalization:
    """Tests for finalization process."""

    def test_neighbor_indices_valid_after_finalization(self):
        """Test that all neighbor indices are valid after removing super-triangle."""
        points = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ])

        tri = triangulate(points, finalize=True)

        n_triangles = len(tri.triangle_vertices)

        # Verify all neighbor indices are either -1 or within valid range
        for tri_idx, neighbors in enumerate(tri.triangle_neighbors):
            for neighbor_idx in neighbors:
                # Neighbor must be -1 (border) or a valid triangle index
                assert neighbor_idx == -1 or (0 <= neighbor_idx < n_triangles), \
                    f"Triangle {tri_idx} has invalid neighbor {neighbor_idx} (valid range: 0-{n_triangles-1})"

    def test_neighbor_symmetry_after_finalization(self):
        """Test that neighbor relationships are symmetric after finalization."""
        points = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ])

        tri = triangulate(points, finalize=True)

        # For each triangle and its neighbors, verify the relationship is symmetric
        for tri_idx, neighbors in enumerate(tri.triangle_neighbors):
            for neighbor_idx in neighbors:
                if neighbor_idx != -1:
                    # The neighbor should have this triangle as one of its neighbors
                    neighbor_neighbors = tri.triangle_neighbors[neighbor_idx]
                    assert tri_idx in neighbor_neighbors, \
                        f"Triangle {tri_idx} lists {neighbor_idx} as neighbor, " \
                        f"but {neighbor_idx} doesn't list {tri_idx} as neighbor"

    def test_find_containing_triangle_works_after_finalization(self):
        """Test that find_containing_triangle works correctly on finalized triangulation."""
        points = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ])

        tri = triangulate(points, finalize=True)

        # Test points inside the triangulation
        test_points = [
            np.array([0.5, 0.5]),
            np.array([1.0, 1.0]),
            np.array([1.5, 0.5]),
            np.array([0.5, 1.5]),
        ]

        for p in test_points:
            # Should not raise an exception
            result = find_containing_triangle(tri, p, 0)
            assert result.idx >= 0
            assert result.idx < len(tri.triangle_vertices)

    def test_finalization_with_grid(self):
        """Test finalization with a larger grid of points."""
        x = np.linspace(0, 3, 5)
        y = np.linspace(0, 3, 5)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        tri = triangulate(points, finalize=True)

        n_triangles = len(tri.triangle_vertices)

        # Verify all neighbor indices are valid
        for tri_idx, neighbors in enumerate(tri.triangle_neighbors):
            for neighbor_idx in neighbors:
                assert neighbor_idx == -1 or (0 <= neighbor_idx < n_triangles), \
                    f"Invalid neighbor index {neighbor_idx} for triangle {tri_idx}"

    def test_no_super_triangle_vertices_in_finalized(self):
        """Test that super-triangle vertices are completely removed after finalization."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])

        tri = triangulate(points, finalize=True)

        # All points should be from the original set
        assert len(tri.all_points) == len(points)

        # All triangle vertices should reference points in valid range
        for tri_verts in tri.triangle_vertices:
            for v_idx in tri_verts:
                assert 0 <= v_idx < len(points), \
                    f"Triangle vertex {v_idx} out of range (0-{len(points)-1})"

    def test_finalization_preserves_triangulation_structure(self):
        """Test that finalization preserves the core triangulation structure."""
        points = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ])

        tri = triangulate(points, finalize=True)

        # Should have 2 triangles for a square
        assert len(tri.triangle_vertices) == 2

        # Each triangle should have 3 vertices
        for tri_verts in tri.triangle_vertices:
            assert len(tri_verts) == 3

        # Each triangle should have 3 neighbors (some may be -1 for borders)
        for neighbors in tri.triangle_neighbors:
            assert len(neighbors) == 3

    def test_last_triangle_idx_valid_after_finalization(self):
        """Test that last_triangle_idx is valid after finalization."""
        points = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ])

        tri = triangulate(points, finalize=True)

        n_triangles = len(tri.triangle_vertices)

        # last_triangle_idx should be within valid range
        assert 0 <= tri.last_triangle_idx < n_triangles
