"""Tests for swap_diagonal function in topology.py."""

import numpy as np
import pytest

from pycdt.build import triangulate
from pycdt.topology import swap_diagonal, SwapDiagonalResult


def test_swap_diagonal_basic():
    """Test basic diagonal swap operation."""
    # Create a simple square with 4 triangles
    points = np.array(
        [
            [0.0, 0.0],  # 0
            [2.0, 0.0],  # 1
            [2.0, 2.0],  # 2
            [0.0, 2.0],  # 3
            [1.0, 1.0],  # 4 - center
        ]
    )

    tri = triangulate(points)

    # Find two triangles that share an edge
    # Look for triangles sharing an edge
    tri1_idx = None
    tri2_idx = None
    for i in range(len(tri.triangle_vertices)):
        for j in range(i + 1, len(tri.triangle_vertices)):
            verts_i = set(tri.triangle_vertices[i])
            verts_j = set(tri.triangle_vertices[j])
            shared = verts_i & verts_j
            if len(shared) == 2:
                tri1_idx = i
                tri2_idx = j
                break
        if tri1_idx is not None:
            break

    assert tri1_idx is not None
    assert tri2_idx is not None

    # Store old vertices
    tri.triangle_vertices[tri1_idx].copy()
    tri.triangle_vertices[tri2_idx].copy()

    # Perform swap
    result = swap_diagonal(tri, tri1_idx, tri2_idx)

    # Check return type
    assert isinstance(result, SwapDiagonalResult)
    assert isinstance(result.t7, (int, np.integer))
    assert isinstance(result.t8, (int, np.integer))
    assert isinstance(result.diagonal_vk, (int, np.integer))
    assert isinstance(result.diagonal_vl, (int, np.integer))

    # Check that triangles still exist at same indices
    assert tri.triangle_vertices[tri1_idx] is not None
    assert tri.triangle_vertices[tri2_idx] is not None

    # Check that vertices changed
    new_tri1_verts = tri.triangle_vertices[tri1_idx]
    new_tri2_verts = tri.triangle_vertices[tri2_idx]

    # Verify the diagonal vertices are valid
    assert 0 <= result.diagonal_vk < len(points)
    assert 0 <= result.diagonal_vl < len(points)
    assert result.diagonal_vk != result.diagonal_vl

    # Verify new diagonal is in both triangles
    assert result.diagonal_vk in new_tri1_verts
    assert result.diagonal_vk in new_tri2_verts
    assert result.diagonal_vl in new_tri1_verts
    assert result.diagonal_vl in new_tri2_verts


def test_swap_diagonal_with_explicit_point_idx():
    """Test swap_diagonal with explicitly provided point_idx."""
    points = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )

    tri = triangulate(points)

    # Find two adjacent triangles
    tri1_idx = None
    tri2_idx = None
    for i in range(len(tri.triangle_vertices)):
        for j in range(i + 1, len(tri.triangle_vertices)):
            verts_i = set(tri.triangle_vertices[i])
            verts_j = set(tri.triangle_vertices[j])
            shared = verts_i & verts_j
            if len(shared) == 2:
                # Find the opposite vertex in tri2
                opposite_in_j = list(verts_j - shared)[0]
                tri1_idx = i
                tri2_idx = j
                point_idx = opposite_in_j
                break
        if tri1_idx is not None:
            break

    # Perform swap with explicit point_idx
    result = swap_diagonal(tri, tri1_idx, tri2_idx, point_idx)

    # Verify diagonal contains point_idx
    assert result.diagonal_vk == point_idx or result.diagonal_vl == point_idx


def test_swap_diagonal_neighbors_updated():
    """Test that neighbor relationships are correctly updated after swap."""
    points = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )

    tri = triangulate(points)

    # Find two triangles sharing an edge
    tri1_idx = None
    tri2_idx = None
    for i in range(len(tri.triangle_vertices)):
        for j in range(i + 1, len(tri.triangle_vertices)):
            verts_i = set(tri.triangle_vertices[i])
            verts_j = set(tri.triangle_vertices[j])
            if len(verts_i & verts_j) == 2:
                tri1_idx = i
                tri2_idx = j
                break
        if tri1_idx is not None:
            break

    # Store old neighbors
    tri.triangle_neighbors[tri1_idx].copy()
    tri.triangle_neighbors[tri2_idx].copy()

    # Perform swap
    swap_diagonal(tri, tri1_idx, tri2_idx)

    # Check that triangles are now neighbors of each other
    new_neighbors_1 = tri.triangle_neighbors[tri1_idx]
    new_neighbors_2 = tri.triangle_neighbors[tri2_idx]

    # The two triangles should be neighbors after the swap
    assert tri2_idx in new_neighbors_1
    assert tri1_idx in new_neighbors_2


def test_swap_diagonal_preserves_ccw():
    """Test that triangles remain in CCW order after swap."""
    from shewchuk import orientation

    points = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )

    tri = triangulate(points)

    # Find two triangles sharing an edge
    tri1_idx = None
    tri2_idx = None
    for i in range(len(tri.triangle_vertices)):
        for j in range(i + 1, len(tri.triangle_vertices)):
            verts_i = set(tri.triangle_vertices[i])
            verts_j = set(tri.triangle_vertices[j])
            if len(verts_i & verts_j) == 2:
                tri1_idx = i
                tri2_idx = j
                break
        if tri1_idx is not None:
            break

    # Perform swap
    swap_diagonal(tri, tri1_idx, tri2_idx)

    # Check CCW orientation of both triangles
    verts1 = tri.triangle_vertices[tri1_idx]
    verts2 = tri.triangle_vertices[tri2_idx]

    pts1 = tri.all_points[verts1]
    pts2 = tri.all_points[verts2]

    # Triangles should be in CCW order (orientation >= 0)
    # Note: orientation can be 0 if points include super-triangle vertices
    orient1 = orientation(*pts1[0], *pts1[1], *pts1[2])
    orient2 = orientation(*pts2[0], *pts2[1], *pts2[2])

    # Check that orientation is non-negative (CCW or degenerate)
    assert orient1 >= 0, f"Triangle 1 has negative orientation: {orient1}"
    assert orient2 >= 0, f"Triangle 2 has negative orientation: {orient2}"


def test_swap_diagonal_return_values():
    """Test that return values t7 and t8 are correct."""
    points = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )

    tri = triangulate(points)

    # Find two triangles sharing an edge
    tri1_idx = None
    tri2_idx = None
    for i in range(len(tri.triangle_vertices)):
        for j in range(i + 1, len(tri.triangle_vertices)):
            verts_i = set(tri.triangle_vertices[i])
            verts_j = set(tri.triangle_vertices[j])
            if len(verts_i & verts_j) == 2:
                tri1_idx = i
                tri2_idx = j
                break
        if tri1_idx is not None:
            break

    # Perform swap
    result = swap_diagonal(tri, tri1_idx, tri2_idx)

    # t7 and t8 should be valid triangle indices or -1
    assert result.t7 >= -1
    assert result.t8 >= -1

    if result.t7 >= 0:
        assert result.t7 < len(tri.triangle_vertices)
    if result.t8 >= 0:
        assert result.t8 < len(tri.triangle_vertices)


def test_swap_diagonal_with_none_point_idx():
    """Test that point_idx=None automatically determines the point."""
    points = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )

    tri = triangulate(points)

    # Find two triangles sharing an edge
    tri1_idx = None
    tri2_idx = None
    for i in range(len(tri.triangle_vertices)):
        for j in range(i + 1, len(tri.triangle_vertices)):
            verts_i = set(tri.triangle_vertices[i])
            verts_j = set(tri.triangle_vertices[j])
            if len(verts_i & verts_j) == 2:
                tri1_idx = i
                tri2_idx = j
                break
        if tri1_idx is not None:
            break

    # Perform swap with point_idx=None
    result = swap_diagonal(tri, tri1_idx, tri2_idx, point_idx=None)

    # Should work and return valid result
    assert isinstance(result, SwapDiagonalResult)
    assert result.diagonal_vk >= 0
    assert result.diagonal_vl >= 0


def test_swap_diagonal_invalid_point_idx():
    """Test that invalid point_idx raises ValueError."""
    points = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )

    tri = triangulate(points)

    # Find two triangles sharing an edge
    tri1_idx = None
    tri2_idx = None
    for i in range(len(tri.triangle_vertices)):
        for j in range(i + 1, len(tri.triangle_vertices)):
            verts_i = set(tri.triangle_vertices[i])
            verts_j = set(tri.triangle_vertices[j])
            if len(verts_i & verts_j) == 2:
                tri1_idx = i
                tri2_idx = j
                break
        if tri1_idx is not None:
            break

    # Find a vertex that's NOT in tri2_idx
    verts_in_tri2 = set(tri.triangle_vertices[tri2_idx])
    invalid_point = None
    for v_idx in range(len(points)):
        if v_idx not in verts_in_tri2:
            invalid_point = v_idx
            break

    # Try to swap with invalid point_idx (not in the triangle)
    if invalid_point is not None:
        with pytest.raises(ValueError):
            swap_diagonal(tri, tri1_idx, tri2_idx, point_idx=invalid_point)


def test_swap_diagonal_diagonal_vertices():
    """Test that diagonal vertices form an edge in the new triangulation."""
    points = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )

    tri = triangulate(points)

    # Find two triangles sharing an edge
    tri1_idx = None
    tri2_idx = None
    for i in range(len(tri.triangle_vertices)):
        for j in range(i + 1, len(tri.triangle_vertices)):
            verts_i = set(tri.triangle_vertices[i])
            verts_j = set(tri.triangle_vertices[j])
            if len(verts_i & verts_j) == 2:
                tri1_idx = i
                tri2_idx = j
                break
        if tri1_idx is not None:
            break

    # Perform swap
    result = swap_diagonal(tri, tri1_idx, tri2_idx)

    # The diagonal vertices should appear together in both modified triangles
    verts1 = set(tri.triangle_vertices[tri1_idx])
    verts2 = set(tri.triangle_vertices[tri2_idx])

    diagonal_verts = {result.diagonal_vk, result.diagonal_vl}

    # Both diagonal vertices should be in both triangles
    assert diagonal_verts.issubset(verts1)
    assert diagonal_verts.issubset(verts2)


def test_swap_diagonal_multiple_swaps():
    """Test that multiple consecutive swaps work correctly."""
    points = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 3.0],
            [0.0, 3.0],
            [1.5, 1.5],
        ]
    )

    tri = triangulate(points)

    # Perform multiple swaps
    swaps_performed = 0
    max_swaps = 3

    for _ in range(max_swaps):
        # Find two triangles sharing an edge
        tri1_idx = None
        tri2_idx = None
        for i in range(len(tri.triangle_vertices)):
            for j in range(i + 1, len(tri.triangle_vertices)):
                verts_i = set(tri.triangle_vertices[i])
                verts_j = set(tri.triangle_vertices[j])
                if len(verts_i & verts_j) == 2:
                    tri1_idx = i
                    tri2_idx = j
                    break
            if tri1_idx is not None:
                break

        if tri1_idx is None:
            break

        # Perform swap
        result = swap_diagonal(tri, tri1_idx, tri2_idx)
        assert isinstance(result, SwapDiagonalResult)
        swaps_performed += 1

    # Should have performed at least one swap
    assert swaps_performed >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
