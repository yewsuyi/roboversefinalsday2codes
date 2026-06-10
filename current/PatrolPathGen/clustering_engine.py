# clustering_engine.py
import numpy as np
from sklearn_extra.cluster import KMedoids
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import lil_matrix

def compute_workload_regions(frequency_matrix):
    """
    Step 3: Segments the safe flight terrain into 3 continuous drone regions.
    Uses Geodesic (Shortest Path) Graph Distances to prevent regions from crossing walls.
    Balances workload via distance matrix upsampling to bypass library limitations.
    """
    y_indices, x_indices = np.where(frequency_matrix > 0)
    if len(y_indices) < 3:
        return np.zeros_like(frequency_matrix)
        
    coordinates = np.column_stack((y_indices, x_indices))
    num_points = len(coordinates)
    
    # 1. Build a Spatial Adjacency Graph tracking valid open steps
    coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(coordinates)}
    adj_matrix = lil_matrix((num_points, num_points), dtype=float)
    
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for idx, (y, x) in enumerate(coordinates):
        for dy, dx in moves:
            neighbor = (y + dy, x + dx)
            if neighbor in coord_to_idx:
                n_idx = coord_to_idx[neighbor]
                adj_matrix[idx, n_idx] = 1.0  
                
    # 2. Compute the true Shortest Path Distance Matrix around obstacles
    print("  Calculating Geodesic Graph Distance Matrix...")
    geodesic_distances = dijkstra(csgraph=adj_matrix.tocsr(), directed=False)
    geodesic_distances[np.isinf(geodesic_distances)] = 9999.0
    
    # 3. Build a map of duplication indices based on painted frequencies
    # High frequency points are cloned to make them 'heavier' to the cluster engine
    duplicate_indices = []
    for idx, coord in enumerate(coordinates):
        weight = int(frequency_matrix[coord[0], coord[1]])
        # Repeat the index based on its frequency weight (1x, 2x, or 3x)
        for _ in range(max(1, weight)):
            duplicate_indices.append(idx)
            
    duplicate_indices = np.array(duplicate_indices)
    
    # Slice the matrix to duplicate rows and columns for high frequency nodes
    weighted_distances = geodesic_distances[duplicate_indices, :][:, duplicate_indices]
    
    # 4. Execute K-Medoids using the Upsampled Geodesic Distance Matrix
    kmedoids = KMedoids(n_clusters=3, metric='precomputed', method='pam', random_state=42)
    kmedoids.fit(weighted_distances)
    
    # Map the trained cluster centers (medoids) back to the original points list
    # kmedoids.medoid_indices_ points to indices within 'weighted_distances'
    actual_medoid_points_indices = duplicate_indices[kmedoids.medoid_indices_]
    
    # Assign every original coordinate point to its closest valid medoid center 
    # using our original obstacle-aware geodesic distance lookup array
    final_labels = []
    for idx in range(num_points):
        distances_to_medoids = geodesic_distances[idx, actual_medoid_points_indices]
        final_labels.append(np.argmin(distances_to_medoids))
        
    # Map back to the native 110x55 grid layout array shape
    region_map = np.zeros_like(frequency_matrix, dtype=np.uint8)
    region_map[y_indices, x_indices] = np.array(final_labels) + 1
    return region_map
