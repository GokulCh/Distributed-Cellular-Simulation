import numpy as np
import time
from src.config import FUEL, BURNING, BURNT, P_SPREAD, P_IGNITE

def update_grid(grid_obj, heavy_load=False):
    current_state = grid_obj.data_with_ghost

    rows, cols = grid_obj.rows, grid_obj.cols
    next_state = current_state[1:-1, :].copy()
    
    burning_mask = (current_state[1:-1, :] == BURNING)
    next_state[burning_mask] = BURNT
    
    fuel_mask = (current_state[1:-1, :] == FUEL)
    burning_neighbors = np.zeros((rows, cols), dtype=int)
    
    burning_neighbors += (current_state[0:-2, :] == BURNING).astype(int)
    burning_neighbors += (current_state[2:, :] == BURNING).astype(int)
    
    inner = current_state[1:-1, :]
    burning_neighbors[:, 1:] += (inner[:, :-1] == BURNING).astype(int)
    burning_neighbors[:, :-1] += (inner[:, 1:] == BURNING).astype(int)
    
    random_vals = np.random.random((rows, cols))
    ignition_prob = 1 - (1 - P_SPREAD) ** burning_neighbors
    ignite_mask = (random_vals < ignition_prob) & fuel_mask
    
    # Spontaneous ignition
    ignite_mask |= (random_vals < P_IGNITE) & fuel_mask
    
    next_state[ignite_mask] = BURNING
    
    if heavy_load:
        num_burning = np.sum(burning_mask) + np.sum(ignite_mask)
        if num_burning > 0:
            target = time.time() + (num_burning * 0.00001) 
            while time.time() < target:
                pass

    return next_state
