import unittest
import numpy as np
from src.grid import Grid
from src.config import FUEL, BURNING, BURNT

class TestGrid(unittest.TestCase):
    def setUp(self):
        self.rows = 10
        self.cols = 10
        self.grid = Grid(self.rows, self.cols)

    def test_initialization(self):
        self.assertEqual(self.grid.rows, self.rows)
        self.assertEqual(self.grid.cols, self.cols)
        self.assertTrue(np.all(self.grid.data == FUEL))
        
    def test_set_fire(self):
        r, c = 5, 5
        self.grid.set_fire(r, c)
        self.assertEqual(self.grid.data[r, c], BURNING)
        
    def test_ghost_rows_structure(self):
        expected_rows = self.rows + 2
        self.assertEqual(self.grid.data_with_ghost.shape, (expected_rows, self.cols))
        
    def test_commit_updates(self):
        new_data = np.full((self.rows, self.cols), BURNT, dtype=np.int8)
        self.grid.commit_updates(new_data)
        self.assertTrue(np.all(self.grid.data == BURNT))

if __name__ == '__main__':
    unittest.main()
