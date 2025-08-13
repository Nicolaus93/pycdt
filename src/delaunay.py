from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from shewchuk import incircle_test
from pathlib import Path


@dataclass
class Triangulation:
    all_points: NDArray[np.floating]
    triangle_vertices: NDArray[np.integer]
    triangle_neighbors: NDArray[np.integer]
    last_triangle_idx: int = 0
    debug_plots: list[NDArray[np.floating]] = field(default_factory=list)

    def plot(
        self,
        show: bool = False,
        title: str = "Triangulation",
        point_labels: bool = False,
        exclude_super_t: bool = False,
    ) -> None:
        """
        Plot the triangulation using matplotlib.

        :param show: Whether to call plt.show() after plotting
        :param title: Title of the plot
        :param point_labels: Whether to label points with their indices
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        offset = 0.01  # Adjust as needed depending on your scale

        # Draw triangles and label triangle indices and vertices
        if exclude_super_t:
            triangle_vertices = self.triangle_vertices[
                3:
            ]  # exclude super-triangle vertices  TODO: hard-coded 3 is not safe
            all_points = self.all_points[:-3]
        else:
            triangle_vertices = self.triangle_vertices
            all_points = self.all_points

        for tri_idx, tri in enumerate(triangle_vertices):
            pts = all_points[tri]
            tri_closed = np.vstack([pts, pts[0]])  # Close the triangle
            ax.plot(tri_closed[:, 0], tri_closed[:, 1], "k-", linewidth=1)

            # Triangle index at centroid
            centroid = np.mean(pts, axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                str(tri_idx),
                fontsize=8,
                ha="center",
                va="center",
                color="green",
            )

            # Vertex indices at triangle corners with slight offset
            for vert_idx, (x, y) in zip(tri, pts):
                ax.text(
                    x + offset,
                    y + offset,
                    str(vert_idx),
                    fontsize=7,
                    ha="left",
                    va="bottom",
                    color="purple",
                )

        # Draw points
        ax.plot(all_points[:, 0], all_points[:, 1], "ro", markersize=3)

        # Optional: Label all points with their indices in blue
        if point_labels:
            for idx, (x, y) in enumerate(all_points):
                ax.text(
                    x, y, str(idx), fontsize=8, ha="right", va="bottom", color="blue"
                )

        ax.set_aspect("equal")
        ax.set_title(title)

        if show:
            plt.show()

        # Convert figure to RGB image in memory
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]  # Convert to RGB by discarding alpha
        plt.close(fig)
        self.debug_plots.append(img)

    def export_animation_matplotlib(self, filepath: str | Path, fps: int = 2) -> None:
        """
        Export animation using matplotlib.

        :param filepath: Output .mp4 or .gif file
        :param fps: Frames per second
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        if not self.debug_plots:
            raise ValueError("No debug plots to export.")

        fig, ax = plt.subplots()
        img_artist = ax.imshow(self.debug_plots[0])
        ax.axis("off")  # Optional: Hide axes

        def update(frame):
            img_artist.set_data(self.debug_plots[frame])
            return [img_artist]

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.debug_plots),
            interval=1000 / fps,
            blit=True,
        )

        # Save based on file extension
        filepath = Path(filepath)
        if filepath.suffix == ".mp4":
            anim.save(filepath, fps=fps, writer="ffmpeg")
        elif filepath.suffix == ".gif":
            anim.save(filepath, fps=fps, writer="pillow")
        else:
            raise ValueError("Unsupported file format. Use .gif or .mp4")

        plt.close(fig)


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

    return inside


def find_containing_triangle(
    triangulation: Triangulation,
    point: NDArray[np.floating],
    last_triangle_idx: int,
) -> int:
    """
    Implementation of Lawson's algorithm to find the triangle containing a point.
    Starts from the most recently added triangle and "walks" towards the point.
    Uses the adjacency information from triangle_neighbors to improve efficiency.

    Parameters:
    - all_points: Array of all point coordinates
    - triangle_vertices: Matrix where each row contains vertex indices of a triangle
    - triangle_neighbors: Matrix where each row contains adjacent triangle indices
    - point: The point to locate
    - last_triangle_idx: Index of the last formed triangle (starting point for search)

    Returns:
    - Index of the triangle containing the point, or None if not found
    """
    if triangulation.triangle_vertices.shape[0] == 0 or last_triangle_idx < 0:
        raise ValueError("No triangles available or invalid starting triangle")

    # for triangle_idx, val in enumerate(triangulation.triangle_vertices):
    #     v_indices = triangulation.triangle_vertices[triangle_idx]
    #     triangle = triangulation.all_points[v_indices]
    #     if point_inside_triangle(triangle, point):
    #         return triangle_idx
    #
    # raise RuntimeError("No triangles found")

    # Start from the last added triangle
    triangle_idx = last_triangle_idx

    # Keep track of visited triangles to avoid cycles
    visited = {triangle_idx}
    while True:
        logger.debug(f"visiting {triangle_idx}")

        # Get the current triangle vertices
        v_indices = triangulation.triangle_vertices[triangle_idx]
        triangle = triangulation.all_points[v_indices]

        # Check if the point is inside this triangle
        if point_inside_triangle(triangle, point):
            return triangle_idx

        # If not inside, find which edge to cross using the adjacent triangles information
        # Get edge indices in the triangle. NOTE!!! => it should be consistent with triangle_neighbors
        edges = [(1, 2), (2, 0), (0, 1)]
        next_idx = None
        for i, (e1, e2) in enumerate(edges):
            # Vector from edge to point
            edge_vector = triangle[e2] - triangle[e1]
            point_vector = point - triangle[e1]

            # If cross product is negative, the point is on the "outside" of this edge
            cross_prod = edge_vector[0] * point_vector[1] - edge_vector[1] * point_vector[0]
            if cross_prod < 0:
                # Get the adjacent triangle for this edge
                adjacent_idx = triangulation.triangle_neighbors[triangle_idx, i]

                # If there's an adjacent triangle (not a boundary) and we haven't visited it
                if adjacent_idx != -1 and adjacent_idx not in visited:
                    next_idx = adjacent_idx
                    visited.add(adjacent_idx)
                    break

        if next_idx is None:
            raise ValueError(f"Couldn't find a triangle containing {point}")
        triangle_idx = next_idx


def get_sorted_points(
    points: NDArray[np.floating], debug: bool = False
) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """
    Sort points into a spatially coherent order to improve incremental insertion efficiency.

    :param points: input points
    :param debug: if True, show a plot of the grid and points
    :return: sorted points and their original indices
    """

    # sort the points into bins
    grid_size = int(np.sqrt(len(points)))
    grid_size = max(grid_size, 4)  # Minimum grid size of 4x4

    y_idxs = (0.99 * grid_size * points[:, 1]).astype(int)
    x_idxs = (0.99 * grid_size * points[:, 0]).astype(int)

    # Create bin numbers in a snake-like pattern
    bin_numbers = np.zeros(len(points), dtype=int)
    for i in range(len(points)):
        y, x = y_idxs[i], x_idxs[i]
        if y % 2 == 0:
            bin_numbers[i] = y * grid_size + x
        else:
            bin_numbers[i] = (y + 1) * grid_size - x - 1

    # Sort the points by their bin numbers
    sorted_indices = np.argsort(bin_numbers)
    sorted_points = points[sorted_indices]

    if debug:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot the grid
        for i in range(grid_size + 1):
            ax.axhline(i / grid_size, color="gray", linestyle="--", linewidth=0.5)
            ax.axvline(i / grid_size, color="gray", linestyle="--", linewidth=0.5)

        # Plot the points
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c="blue",
            label="Original points",
        )
        ax.scatter(
            sorted_points[:, 0],
            sorted_points[:, 1],
            c="red",
            s=10,
            label="Sorted points",
            alpha=0.6,
        )

        ax.set_title("Debug: Normalized Points and Grid")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend()
        plt.grid(False)
        plt.show()

    return sorted_points, sorted_indices


def initialize_triangulation(
    sorted_points: NDArray[np.floating],
    margin: float = 10.0,
) -> Triangulation:
    """
    Initialize the triangulation with a super triangle.

    :param sorted_points: Sorted input points
    :param margin: extra margin to ensure all points are inside the super triangle
    :param debug: if True, show a plot with the super triangle
    :return: All points array, triangle vertices, triangle neighbors, and last triangle index
    """
    super_vertices = np.array(
        [
            [-margin + 0.5, -margin / 2],
            [margin + 0.5, -margin / 2],
            [0.5, margin],
        ]
    )

    # Add super-triangle vertices to the points array
    all_points = np.vstack([sorted_points, super_vertices])
    n_original_points = len(sorted_points)

    # Initial triangle is the super-triangle
    # Vertex indices are n_original_points, n_original_points+1, n_original_points+2
    triangle_vertices = np.array(
        [[n_original_points, n_original_points + 1, n_original_points + 2]]
    )

    # Initial triangle has no neighbors (all boundaries)
    triangle_neighbors = np.array([-1, -1, -1], dtype=int).reshape(1, -1)

    return Triangulation(
        all_points=all_points,
        triangle_vertices=triangle_vertices,
        triangle_neighbors=triangle_neighbors,
    )


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


def find_neighbor_edge_index(
    triangle_neighbors: NDArray, triangle_idx: int, neighbor_idx: int
) -> int:
    """
    Find which edge index (0, 1, or 2) connects to the given neighbor.
    Returns the index i such that triangle_neighbors[triangle_idx, i] == neighbor_idx
    """
    neighbors = triangle_neighbors[triangle_idx]
    for i, neighbor in enumerate(neighbors):
        if neighbor == neighbor_idx:
            return i
    raise RuntimeError(
        f"Triangle {triangle_idx} is not a neighbor of triangle {neighbor_idx}"
    )


def find_shared_edge(
    tri1_vertices: NDArray, tri2_vertices: NDArray
) -> tuple[int, int, int, int]:
    """
    Find the shared edge between two triangles.
    Returns (v1, v2, opposite1, opposite2) where:
    - v1, v2 are the shared vertices
    - opposite1 is the vertex in tri1 not on the shared edge
    - opposite2 is the vertex in tri2 not on the shared edge
    """
    shared = list(set(tri1_vertices) & set(tri2_vertices))
    if len(shared) != 2:
        raise RuntimeError(
            f"Triangles must share exactly one edge. Shared vertices: {shared}"
        )

    v1, v2 = shared
    opposite1 = next(v for v in tri1_vertices if v not in shared)
    opposite2 = next(v for v in tri2_vertices if v not in shared)

    return v1, v2, opposite1, opposite2


def find_vertex_position(triangle_vertices: NDArray, vertex: int) -> int:
    """Find the position (0, 1, or 2) of a vertex in a triangle"""
    return next(i for i in range(3) if triangle_vertices[i] == vertex)


def incircle_test_debug(t3_points, p):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    def circumcenter(A, B, C):
        # Compute circumcenter using perpendicular bisector intersection
        D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
        if np.isclose(D, 0):
            return None, None  # Degenerate triangle

        Ux = (
            np.dot(A, A) * (B[1] - C[1])
            + np.dot(B, B) * (C[1] - A[1])
            + np.dot(C, C) * (A[1] - B[1])
        ) / D
        Uy = (
            np.dot(A, A) * (C[0] - B[0])
            + np.dot(B, B) * (A[0] - C[0])
            + np.dot(C, C) * (B[0] - A[0])
        ) / D
        return np.array([Ux, Uy]), np.linalg.norm(A - np.array([Ux, Uy]))

    p1, p2, p3 = t3_points
    center, radius = circumcenter(p1, p2, p3)

    fig, ax = plt.subplots()
    tri_pts = np.vstack([t3_points, t3_points[0]])
    ax.plot(tri_pts[:, 0], tri_pts[:, 1], "k-", label="Triangle")
    ax.plot(*p, "ro", label="Query point")

    if center is not None:
        circ = Circle(
            center,
            radius,
            fill=False,
            color="blue",
            linestyle="--",
            label="Circumcircle",
        )
        ax.add_patch(circ)
        ax.plot(*center, "bx", label="Circumcenter")

    for i, (x, y) in enumerate(t3_points):
        ax.text(x, y, f"v{i}", fontsize=8, color="purple", ha="center", va="center")

    ax.set_aspect("equal")
    ax.set_title("Incircle Test Debug")
    ax.legend()
    plt.show()


def lawson_swapping(
    point_idx: int,
    stack: list[tuple[int, int]],
    triangulation: Triangulation,
    debug: bool = False,
) -> None:
    """
    Restore the Delaunay condition by flipping edges as necessary.

    This implementation uses robust geometric predicates to ensure proper
    orientation handling and maintains counterclockwise vertex ordering.

    :param point_idx: Index of the newly inserted point
    :param stack: List of (triangle_idx, candidate_triangle_idx) pairs to check
    :param triangulation: Triangulation structure containing geometry and topology
    """
    vertices = triangulation.triangle_vertices
    neighbors = triangulation.triangle_neighbors
    points = triangulation.all_points

    p = points[point_idx]

    logger.info("Lawson swapping phase")
    while stack:
        logger.debug(f"Stack -> {stack}")
        t3_idx, t4_idx = stack.pop()

        # Skip if either triangle is invalid
        if t3_idx == -1 or t4_idx == -1:
            continue

        t3 = vertices[t3_idx]
        t4 = vertices[t4_idx]
        t3_points = points[t3]

        # Check if point lies in circumcircle of the neighboring triangle
        if incircle_test(*p, *t3_points[0], *t3_points[1], *t3_points[2]) < 0:
            if debug:
                incircle_test_debug(t3_points, p)
            continue

        logger.warning(
            f"Point {point_idx} from triangle {t4_idx} lies in circumcircle of triangle {t3_idx}; flipping shared edge"
        )

        # Find shared edge and opposite vertices
        a, b, temp, c = find_shared_edge(t4, t3)

        # The candidate triangle should contain the newly inserted point
        assert temp in t4
        assert temp == point_idx

        # Create new triangles after edge flip
        # Ensure counterclockwise orientation
        new_t3_vertices = ensure_ccw_triangle(np.array([point_idx, c, a]), points)
        new_t4_vertices = ensure_ccw_triangle(np.array([point_idx, b, c]), points)

        # Find neighbor triangles before updating
        # For triangle t4_idx, find neighbors opposite to each vertex
        old_t4_neighbors = neighbors[t4_idx].copy()
        old_t3_neighbors = neighbors[t3_idx].copy()

        # Find which edges correspond to which neighbors in the old triangles
        # For t4_idx (contains point_idx)
        # Neighbor opposite to A in t4
        pos_a = find_vertex_position(t4, a)
        t6 = int(old_t4_neighbors[pos_a])
        # Neighbor opposite to B in candidate triangle
        pos_b = find_vertex_position(t4, b)
        t5 = int(old_t4_neighbors[pos_b])

        # For t3_idx
        # Neighbor opposite to A in neigh triangle
        pos_a_neigh = find_vertex_position(t3, a)
        t7 = int(old_t3_neighbors[pos_a_neigh])
        # Neighbor opposite to B in neigh triangle
        pos_b_neigh = find_vertex_position(t3, b)
        t8 = int(old_t3_neighbors[pos_b_neigh])

        # Update triangle vertices
        vertices[t3_idx] = new_t3_vertices
        vertices[t4_idx] = new_t4_vertices

        # Update neighbors for new triangles
        # We need to determine neighbors based on the actual vertex ordering after CCW correction

        # For new_t3: determine neighbors for each edge
        t3_neighbors = np.array([-1, -1, -1])
        for i in range(3):
            curr_v = new_t3_vertices[i]
            if curr_v == point_idx:
                t3_neighbors[i] = t8
            elif curr_v == c:
                t3_neighbors[i] = t5
            elif curr_v == a:
                t3_neighbors[i] = t4_idx
            else:
                raise RuntimeError

        # For new_t4: determine neighbors for each edge
        t4_neighbors = np.array([-1, -1, -1])
        for i in range(3):
            curr_v = new_t4_vertices[i]
            if curr_v == point_idx:
                t4_neighbors[i] = t7
            elif curr_v == c:
                t4_neighbors[i] = t6
            elif curr_v == b:
                t4_neighbors[i] = t3_idx
            else:
                raise RuntimeError

        neighbors[t3_idx] = t3_neighbors
        neighbors[t4_idx] = t4_neighbors

        # Update references in neighboring triangles
        def update_neighbor_reference(
            neighbor_idx: int, old_triangle: int, new_triangle: int
        ):
            if neighbor_idx != -1:
                mask = neighbors[neighbor_idx] == old_triangle
                if np.any(mask):
                    neighbors[neighbor_idx][mask] = new_triangle

        # Update all affected neighbor references
        update_neighbor_reference(t5, t4_idx, t3_idx)
        update_neighbor_reference(t7, t3_idx, t4_idx)

        # Add new potentially illegal edges to stack
        # Check edge opposite to point_idx in both new triangles
        if t8 != -1:
            stack.append((int(t8), int(t3_idx)))
        if t7 != -1:
            stack.append((int(t7), int(t4_idx)))


def reorder_neighbors_for_triangle(
    original_vertices: np.ndarray, final_vertices: np.ndarray, original_neighbors: list
) -> list:
    """Reorder neighbor indices to match reordered vertices"""
    if np.array_equal(original_vertices, final_vertices):
        return original_neighbors

    # Create new neighbor array based on final vertex ordering
    final_neighbors = [None] * 3
    for idx in range(3):
        final_vertex = final_vertices[idx]
        # Find where this vertex was in the original ordering
        original_pos = np.where(original_vertices == final_vertex)[0][0]
        # The neighbor opposite to this vertex in the final triangle
        # is the same as the neighbor opposite to it in the original triangle
        final_neighbors[idx] = original_neighbors[original_pos]

    return final_neighbors


def insert_point(
    point_idx: int,
    point: NDArray[np.floating],
    triangulation: Triangulation,
    debug: bool = False,
) -> Triangulation:
    """
    Insert a point into the triangulation.

    :param point_idx: Index of the point to insert
    :param point: Coordinates of the point
    :param triangulation:
    :param debug:
    :return: Updated triangle_vertices, triangle_neighbors, last_triangle_idx
    """
    # Find the triangle containing the point
    containing_idx = find_containing_triangle(
        triangulation, point, triangulation.last_triangle_idx
    )

    # Get vertices of the containing triangle
    v1_idx, v2_idx, v3_idx = triangulation.triangle_vertices[containing_idx]
    logger.debug(
        f"Triangle {containing_idx} with vertices {v1_idx}, {v2_idx}, {v3_idx} contains point {point_idx}"
    )

    # Get the original neighbors before we modify anything
    orig_neighbors = triangulation.triangle_neighbors[containing_idx].copy()
    neighbor_opp_v1 = orig_neighbors[0]  # neighbor opposite to v1 (across edge v2-v3)
    neighbor_opp_v2 = orig_neighbors[1]  # neighbor opposite to v2 (across edge v3-v1)
    neighbor_opp_v3 = orig_neighbors[2]  # neighbor opposite to v3 (across edge v1-v2)
    logger.debug(
        f"Neighbours: opposite v3={neighbor_opp_v3}, opposite v1={neighbor_opp_v1}, opposite v2={neighbor_opp_v2}"
    )

    # Create three new triangles by connecting point to each vertex
    # Note that we need to maintain CCW ordering for each triangle

    # Triangle 1: point + edge (v1, v2)
    original_t12_vertices = np.array([point_idx, v1_idx, v2_idx])
    new_triangle_12 = ensure_ccw_triangle(
        original_t12_vertices, triangulation.all_points
    )

    # Triangle 2: point + edge (v2, v3)
    original_t23_vertices = np.array([point_idx, v2_idx, v3_idx])
    new_triangle_23 = ensure_ccw_triangle(
        original_t23_vertices, triangulation.all_points
    )

    # Triangle 3: point + edge (v3, v1)
    original_t31_vertices = np.array([point_idx, v3_idx, v1_idx])
    new_triangle_31 = ensure_ccw_triangle(
        original_t31_vertices, triangulation.all_points
    )

    # Update triangulation data structure
    # first, update containing_idx
    triangulation.triangle_vertices[containing_idx] = new_triangle_12

    # Then add two more triangles
    triangulation.triangle_vertices = np.vstack(
        (
            triangulation.triangle_vertices,
            [new_triangle_23],
            [new_triangle_31],
        )
    )

    # Get indices for the new triangles
    triangle_count = len(triangulation.triangle_vertices)
    new_triangle_12_idx = containing_idx  # reusing the original index
    new_triangle_23_idx = triangle_count - 2
    new_triangle_31_idx = triangle_count - 1

    # Update neighbors array to accommodate new triangles
    # Add rows for the two new triangles
    triangulation.triangle_neighbors = np.vstack(
        (
            triangulation.triangle_neighbors,
            np.full((2, 3), -1, dtype=int),  # Initialize with -1 for no neighbor
        )
    )

    # Set up neighbor relationships for the three new triangles
    # for every triangle [v1, v2, v3] we define the neighbor as:
    # [t_sharing_edge_opposite_of_v1, t_sharing_edge_opposite_of_v2, t_sharing_edge_opposite_of_v3]

    # Triangle 1 (point, v1, v2)
    original_neighs_t12 = [
        neighbor_opp_v3,  # Edge (v1, v2) -> original neighbor opposite to v3
        new_triangle_23_idx,  # Edge (v2, point) -> triangle_23
        new_triangle_31_idx,  # Edge (point, v1) -> triangle_31
    ]
    triangulation.triangle_neighbors[new_triangle_12_idx] = (
        reorder_neighbors_for_triangle(
            original_t12_vertices,
            new_triangle_12,
            original_neighs_t12,
        )
    )

    # Triangle 2 (point, v2, v3)
    original_neighs_t23 = [
        neighbor_opp_v1,  # Edge (v2, v3) -> original neighbor opposite to v1
        new_triangle_31_idx,  # Edge (v3, point) -> triangle_31
        new_triangle_12_idx,  # Edge (point, v2) -> triangle_12
    ]
    triangulation.triangle_neighbors[new_triangle_23_idx] = (
        reorder_neighbors_for_triangle(
            original_t23_vertices,
            new_triangle_23,
            original_neighs_t23,
        )
    )

    # Triangle 3 (point, v3, v1)
    original_neighs_t31 = [
        neighbor_opp_v2,  # Edge (v3, v1) -> original neighbor opposite to v2
        new_triangle_12_idx,  # Edge (v1, point) -> triangle_12
        new_triangle_23_idx,  # Edge (point, v3) -> triangle_23
    ]
    triangulation.triangle_neighbors[new_triangle_31_idx] = (
        reorder_neighbors_for_triangle(
            original_t31_vertices,
            new_triangle_31,
            original_neighs_t31,
        )
    )

    # Update the original neighbors to point to the correct new triangles
    # and place them on the stack used for Lawson swapping
    stack = []

    def update_external_neighbor(neighbor_idx: int, new_triangle_idx: int):
        if neighbor_idx >= 0:
            stack.append((int(neighbor_idx), int(new_triangle_idx)))
            neighbor_refs = triangulation.triangle_neighbors[neighbor_idx]
            for i, ref in enumerate(neighbor_refs):
                if ref == containing_idx:
                    triangulation.triangle_neighbors[neighbor_idx, i] = new_triangle_idx
                    break

    # Update each external neighbor to point to the correct new triangle
    update_external_neighbor(neighbor_opp_v1, new_triangle_23_idx)  # v2-v3 edge
    update_external_neighbor(neighbor_opp_v2, new_triangle_31_idx)  # v3-v1 edge
    update_external_neighbor(neighbor_opp_v3, new_triangle_12_idx)  # v1-v2 edge

    # Restore Delaunay triangulation (edge flipping)
    if stack:
        lawson_swapping(point_idx, stack, triangulation)

    if debug:
        triangulation.plot()

    # Update last_triangle_idx to one of the new triangles
    triangulation.last_triangle_idx = triangle_count - 1
    return triangulation


def remove_super_triangle_triangles(
    triangulation: Triangulation,
    n_original_points: int,
) -> None:
    """
    Modify the Triangulation object in-place by removing triangles
    that contain vertices from the super triangle.

    :param triangulation: The Triangulation object to modify.
    :param n_original_points: Number of original points (before adding super triangle vertices).
    """
    mask = np.all(triangulation.triangle_vertices < n_original_points, axis=1)

    triangulation.triangle_vertices = triangulation.triangle_vertices[mask]
    triangulation.triangle_neighbors = triangulation.triangle_neighbors[mask]
    triangulation.last_triangle_idx = triangulation.triangle_vertices.shape[0]


def triangulate(points: NDArray[np.floating], debug: bool = False):
    """
    Implement Delaunay triangulation using the incremental algorithm with efficient
    adjacency tracking.

    :param points: Input points to triangulate
    :param debug: plot debug images
    :return: List of triangles forming the Delaunay triangulation
    """
    # Sort points for efficient insertion
    # Find the min and max along each axis (x and y)
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # Normalize the points to [0, 1] range
    normalized_points = (points - min_vals) / (max_vals - min_vals)
    # sorted_points, sorted_indices = get_sorted_points(normalized_points)  # TODO: reactivate?
    sorted_points = normalized_points

    # Initialize triangulation with super triangle
    triangulation = initialize_triangulation(sorted_points, margin=2.0)

    n_original_points = len(sorted_points)

    # Loop over each point and insert into triangulation
    for point_idx, point in enumerate(sorted_points):
        # Insert point and update triangulation
        triangulation = insert_point(
            point_idx=point_idx,
            point=point,
            triangulation=triangulation,
            debug=debug,
        )

    # Remove triangles that contain vertices of the super triangle
    remove_super_triangle_triangles(triangulation, n_original_points)

    return triangulation


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


def remove_holes(
    triangulation: Triangulation, outer: list[int], holes: list[list[int]]
) -> None:
    """
    Get the final triangulation, compute the centroids for all the triangles. If a centroid is outside the domain,
    remove the corresponding triangle.
    TODO: caveats we can have 1) Centroid Inside, Triangle Outside, 2) Centroid Outside, Triangle Inside
    for a more robust version, use this strategy for all vertices (if a point is on the boundary, then it's inside)
    TODO: points in outer and holes should be sorted as those in triangulation do!
    """
    tri_vertices = triangulation.triangle_vertices
    to_delete = []
    poly_outer = [triangulation.all_points[p] for p in outer]
    poly_holes = []
    for hole in holes:
        poly_hole = [triangulation.all_points[p] for p in hole]
        poly_holes.append(poly_hole)

    for idx, row in enumerate(tri_vertices):
        tri_points = triangulation.all_points[row]
        centroid = np.mean(tri_points, axis=0)
        if not is_inside_domain(centroid, poly_outer, poly_holes):
            to_delete.append(idx)

    logger.info(f"Removing triangles {to_delete}")
    new_tri_vertices = [
        row for idx, row in enumerate(tri_vertices) if idx not in to_delete
    ]
    # TODO: update triangle neighbors?
    triangulation.triangle_vertices = new_tri_vertices


if __name__ == "__main__":
    arr = np.array(
        [
            [24.311, -7.358],
            [23.574, -9.456],
            [22.657, -11.481],
            [21.566, -13.419],
            [20.31, -15.253],
            [18.899, -16.971],
            [17.342, -18.558],
            [15.653, -20.004],
            [13.844, -21.296],
            [11.929, -22.425],
            [6.53, -24.546],
            [0.791, -25.388],
            [-4.989, -24.905],
            [-10.509, -23.124],
            [-15.48, -20.137],
            [-19.645, -16.1],
            [-22.786, -11.224],
            [-24.738, -5.762],
            [-25.4, 0.0],
            [-23.868, 8.687],
            [-19.458, 16.327],
            [-12.7, 21.997],
            [-4.411, 25.014],
            [4.411, 25.014],
            [12.7, 21.997],
            [19.458, 16.327],
            [23.868, 8.687],
            [25.4, 0.0],
            [25.386, -0.829],
            [25.346, -1.658],
            [25.278, -2.485],
            [25.184, -3.309],
            [25.062, -4.129],
            [24.914, -4.945],
            [24.739, -5.756],
            [24.538, -6.561],
        ]
    )

    yy = triangulate(arr)
    yy.plot(exclude_super_t=True)
