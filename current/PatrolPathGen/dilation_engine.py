# dilation_engine.py
import numpy as np
from scipy.ndimage import binary_dilation

def generate_dilation_map(obstacle_matrix, mode='radial', radius=3):
    """
    Step 2: Expands obstacle boundaries to establish a safe drone configuration flight zone.
    """
    if radius <= 0:
        return obstacle_matrix.copy()
        
    if mode == 'radial':
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        stencil = x**2 + y**2 <= radius**2
    elif mode == 'square':
        stencil = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    else:
        raise ValueError("Dilation mode must be either 'radial' or 'square'")
        
    dilated_map = binary_dilation(obstacle_matrix, structure=stencil).astype(np.uint8)
    return dilated_map
