from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
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
        exclude_unused_points: bool = True,
        fontsize: int = 7,
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
            # Indices of super triangle vertices (last 3 points)
            super_vertices = set(range(len(self.all_points) - 3, len(self.all_points)))

            # Keep only triangles that don’t touch super vertices
            mask = ~np.any(
                np.isin(self.triangle_vertices, list(super_vertices)), axis=1
            )
            triangle_vertices = self.triangle_vertices[mask]

            # Drop the last 3 points (super triangle vertices)
            all_points = self.all_points[:-3]
        else:
            triangle_vertices = self.triangle_vertices
            all_points = self.all_points

        # restrict points to those used in triangles
        if exclude_unused_points:
            used_vertices = np.unique(triangle_vertices)
        else:
            used_vertices = np.arange(len(all_points))

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
                fontsize=fontsize,
                ha="center",
                va="center",
                color="green",
            )

            if point_labels:
                # Vertex indices at triangle corners with slight offset
                for vert_idx, (x, y) in zip(tri, pts):
                    if vert_idx in used_vertices:
                        ax.text(
                            x + offset,
                            y + offset,
                            str(vert_idx),
                            fontsize=fontsize,
                            ha="left",
                            va="bottom",
                            color="purple",
                        )
                        # ax.plot(x, y, "ro", markersize=3)

        # Draw points
        ax.plot(all_points[:, 0], all_points[:, 1], "ro", markersize=fontsize)
        # # Optional: Label all points with their indices in blue
        # if point_labels:
        #     for idx, (x, y) in enumerate(all_points):
        #         ax.text(
        #             x, y, str(idx), fontsize=8, ha="right", va="bottom", color="blue"
        #         )

        ax.set_aspect("equal")
        ax.set_title(title)

        if show:
            plt.show()

        # Convert figure to RGB image in memory
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()  # type: ignore[reportAttributeAccessIssue]
        img = np.asarray(buf)[:, :, :3]  # Convert to RGB by discarding alpha
        plt.close(fig)
        self.debug_plots.append(img)

    def debug_plot_triangles(self, tri_indices, title=""):
        """Plot only selected triangles with their vertices."""
        import matplotlib.pyplot as plt

        pts = self.all_points

        fig, ax = plt.subplots()
        for tidx in tri_indices:
            verts = self.triangle_vertices[tidx]
            tri = pts[verts]
            # Close the polygon loop
            poly = np.vstack([tri, tri[0]])
            ax.plot(poly[:, 0], poly[:, 1], "k-")
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.2)
            for v in verts:
                ax.scatter(pts[v][0], pts[v][1], c="r")
                ax.text(pts[v][0], pts[v][1], str(v), fontsize=8)

            # # Label the triangle itself at its centroid
            # centroid = tri.mean(axis=0)
            # ax.text(*centroid, f"T{tidx}", fontsize=10, color="b", weight="bold")

        ax.set_title(title)
        ax.axis("equal")
        plt.show()

    def debug_plot_edge_region(self, edge_point_1, edge_point_2, title=""):
        """Plot all triangles that contain the given edge (edge_point_1, edge_point_2)."""
        import matplotlib.pyplot as plt

        pts = self.all_points

        mask = np.any(
            (self.triangle_vertices == edge_point_1)
            | (self.triangle_vertices == edge_point_2),
            axis=1,
        )
        tri_indices = np.where(mask)[0]

        offset = 0.01
        fig, ax = plt.subplots()
        for tidx in tri_indices:
            verts = self.triangle_vertices[tidx]
            tri = pts[verts]
            # Close the polygon loop
            poly = np.vstack([tri, tri[0]])
            ax.plot(poly[:, 0], poly[:, 1], "k-")
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.2)

            # Plot vertices
            for v in verts:
                if v in (edge_point_1, edge_point_2):
                    continue
                ax.scatter(*pts[v], c="red", s=2, zorder=3)
                ax.text(*(pts[v] + 8 * offset), str(v), fontsize=1, color="red")  # type: ignore[reportCallIssue]

            centroid = tri.mean(axis=0)
            ax.text(*centroid, f"{tidx}", fontsize=1, color="green", weight="bold")  # type: ignore[reportCallIssue]

        ax.scatter(*pts[edge_point_1], c="blue", s=2, zorder=3)
        ax.text(
            *(pts[edge_point_1] + 8 * offset),
            f"{edge_point_1}",  # type: ignore[reportCallIssue]
            fontsize=1,
            color="b",
            weight="bold",
        )
        ax.scatter(*pts[edge_point_2], c="blue", s=2, zorder=3)
        ax.text(
            *(pts[edge_point_2] + 8 * offset),
            f"{edge_point_2}",  # type: ignore[reportCallIssue]
            fontsize=1,
            color="b",
            weight="bold",
        )

        ax.set_title(title)
        ax.axis("equal")
        plt.show()

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
            center,  # type: ignore[reportArgumentType]
            radius,  # type: ignore[reportArgumentType]
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
