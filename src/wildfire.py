import numpy as np
from src.config import FUEL, BURNING, BURNT, P_SPREAD


def update_grid(grid_obj):

    # Get the data with ghost rows
    current_state = grid_obj.data_with_ghost
    rows, cols = grid_obj.rows, grid_obj.cols

    # Copy inner grid (excluding ghost rows)
    next_state = current_state[1:-1, :].copy()

    # Burning cells turn into burnt cells
    burning_mask = current_state[1:-1, :] == BURNING
    next_state[burning_mask] = BURNT

    # Fuel cells might catch fire if neighbors are burning
    fuel_mask = current_state[1:-1, :] == FUEL

    # Check neighbors (Up, Down, Left, Right)

    # Count burning neighbors (von Neumann neighborhood)
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

    random_vals = np.random.random((rows, cols))

    # Ignition probability based on neighbors
    ignition_prob = 1 - (1 - P_SPREAD) ** burning_neighbors

    # Fuel cells that catch fire
    ignite_mask = (random_vals < ignition_prob) & fuel_mask

    # Update next state
    next_state[ignite_mask] = BURNING

    return next_state
