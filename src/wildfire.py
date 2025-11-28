import numpy as np
from src.config import FUEL, BURNING, BURNT, P_SPREAD

def update_grid(grid_obj):
    """ Update the grid based on wildfire rules. """
    
    # Get the data with ghost rows
    current_state = grid_obj.data_with_ghost
    rows, cols = grid_obj.rows, grid_obj.cols

    # Copy the inner part of the grid    
    next_state = current_state[1:-1, :].copy()

    # Burning cells turn into burnt cells
    burning_mask = (current_state[1:-1, :] == BURNING)
    next_state[burning_mask] = BURNT
    
    # Fuel cells might catch fire if neighbors are burning
    fuel_mask = (current_state[1:-1, :] == FUEL)
    
    # Check neighbors (Up, Down, Left, Right)
    
    # Neighbors that are burning
    burning_neighbors = np.zeros((rows, cols), dtype=int)
    
    # Up
    burning_neighbors += (current_state[0:-2, :] == BURNING).astype(int)
    
    # Down
    burning_neighbors += (current_state[2:, :] == BURNING).astype(int)
    
    # Left
    inner = current_state[1:-1, :]
    burning_neighbors[:, 1:] += (inner[:, :-1] == BURNING).astype(int)
    
    # Right
    burning_neighbors[:, :-1] += (inner[:, 1:] == BURNING).astype(int)
    
    # Generate random numbers for fuel cells
    random_vals = np.random.random((rows, cols))
    
    # Ignition probability based on neighbors
    ignition_prob = 1 - (1 - P_SPREAD) ** burning_neighbors
    
    # Fuel cells that catch fire
    ignite_mask = (random_vals < ignition_prob) & fuel_mask
    
    # Update next state
    next_state[ignite_mask] = BURNING
    
    return next_state
