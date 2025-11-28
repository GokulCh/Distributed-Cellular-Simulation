import numpy as np
from src.config import FUEL, BURNING, BURNT

class Grid:
    """Local grid class."""
    def __init__(self, rows, cols):
        """Initialize the local grid."""
        self.rows = rows
        self.cols = cols
        self.data = np.full((rows, cols), FUEL, dtype=np.int8)
        self.data_with_ghost = np.full((rows + 2, cols), FUEL, dtype=np.int8)

    def update_from_ghost(self):
        """Copy the internal data to the center of data_with_ghost."""
        self.data_with_ghost[1:-1, :] = self.data

    def commit_updates(self, new_data):
        """Update the internal data from the new calculation."""
        self.data = new_data
        self.update_from_ghost()

    def set_fire(self, r, c):
        """Ignite a specific cell (local coordinates)."""
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.data[r, c] = BURNING
            self.data_with_ghost[r+1, c] = BURNING

    def get_state(self):
        """Return the internal data."""
        return self.data
