import numpy as np
from scipy.ndimage import binary_dilation, label
from dataclasses import dataclass, field
from typing import List, Tuple

# Cell state constants — adjust to match your existing grid values
UNKNOWN  = 0
FREE     = 1
OCCUPIED = 2

@dataclass
class FrontierCluster:
    cells: List[Tuple[int, int]]    # (row, col) grid indices of every frontier cell
    centroid: Tuple[float, float]   # (row, col) centroid of the cluster
    size: int = 0                   # number of cells in cluster

    def __post_init__(self):
        self.size = len(self.cells)


def find_frontier_cells(grid: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask where True = frontier cell.
    A frontier cell is a FREE cell with at least one UNKNOWN 8-neighbour.

    grid : 2D numpy array
    """
    free_mask    = (grid == FREE)
    unknown_mask = (grid == UNKNOWN)

    # Dilate unknown mask by 1 cell in all 8 directions
    # Any free cell touching this dilated mask is a frontier
    unknown_adjacent = binary_dilation(unknown_mask, structure=np.ones((3, 3)))

    return free_mask & unknown_adjacent


def find_frontier_clusters(
    frontier_mask,
    grid: np.ndarray,
    min_cluster_size: int = 5,
) -> List[FrontierCluster]:
    """
    Identify and cluster frontier cells into connected components.

    grid             : 2D numpy array (UNKNOWN=-1, FREE=0, OCCUPIED=1)
    min_cluster_size : discard clusters smaller than this (filters noise)

    Returns a list of FrontierCluster objects, largest cluster first.
    """
    # frontier_mask = find_frontier_cells(grid)

    if not np.any(frontier_mask):
        print("empty frontier mask")
        return []

    # 4-connected structure for clustering
    # (diagonal frontier cells are separate clusters — more conservative)
    # structure_4 = np.array([[0, 1, 0],
    #                          [1, 1, 1],
    #                          [0, 1, 0]])

    structure_8 = np.ones((3,3), dtype=int) #NEW COUNT DIAGS AS ONE CLUSTER

    labelled, n_clusters = label(frontier_mask, structure=structure_8)

    clusters = []
    for cluster_id in range(1, n_clusters + 1):
        rows, cols = np.where(labelled == cluster_id)
        cells = list(zip(rows.tolist(), cols.tolist()))

        if len(cells) < min_cluster_size:
            continue

        centroid_row = float(np.mean(rows))
        centroid_col = float(np.mean(cols))

        # print("clust found") #CAN REMOVE
        clusters.append(FrontierCluster(
            cells=cells,
            centroid=(centroid_row, centroid_col),
        ))

    # Sort largest cluster first
    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters


def rank_frontier_clusters(
    clusters: List[FrontierCluster],
    drone_pos: Tuple[float, float],     # (row, col) which is (y, x)
    strategy: str = "nearest",          
    # strategy: str = "largest",
    # strategy: str = "score",
    # "nearest" | "largest" | "score"
    resolution=0.1,
) -> List[FrontierCluster]:
    """
    Rank clusters by strategy:
      "nearest" — closest centroid to drone
      "largest" — most frontier cells
      "score"   — size / distance (balances both)
    """
    if not clusters:
        return []

    dr, dc = drone_pos

    for cluster in clusters:
        cr, cc = cluster.centroid
        cluster.distance = float(np.hypot(cr - dr, cc - dc))
        cluster.score    = cluster.size / max(cluster.distance, resolution)

    if strategy == "nearest":
        clusters.sort(key=lambda c: c.distance)
    elif strategy == "largest":
        clusters.sort(key=lambda c: c.size, reverse=True)
    else:  # "score"
        clusters.sort(key=lambda c: c.score, reverse=True)

    return clusters

def closest_cell_to_centroid(cluster: FrontierCluster) -> Tuple[int, int]:
    """Return the cell (row, col) in the cluster closest to the centroid."""
    cells_array = np.array(cluster.cells)           # (N, 2)
    centroid    = np.array(cluster.centroid)        # (2,)

    dists  = np.hypot(cells_array[:, 0] - centroid[0],
                      cells_array[:, 1] - centroid[1])  # (N,)
    idx    = np.argmin(dists)
    return cluster.cells[idx]