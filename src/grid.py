import numpy as np
from src.config import FUEL, BURNING, BURNT

class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = np.full((rows, cols), FUEL, dtype=np.int8)
        self.data_with_ghost = np.full((rows + 2, cols), FUEL, dtype=np.int8)

    def update_from_ghost(self):
        self.data_with_ghost[1:-1, :] = self.data

    def commit_updates(self, new_data):
        self.data = new_data
        self.update_from_ghost()

    def set_fire(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.data[r, c] = BURNING
            self.data_with_ghost[r+1, c] = BURNING

    def get_state(self):
        return self.data
