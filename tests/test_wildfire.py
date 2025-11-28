import unittest
from src.grid import Grid
from src.wildfire import update_grid
from src.config import FUEL, BURNING, BURNT

class TestWildfire(unittest.TestCase):
    """Test the wildfire simulation."""
    
    def setUp(self):
        """Set up the test environment."""
        self.grid = Grid(10, 10)

    def test_burning_to_burnt(self):
        """Test that burning cells turn into burnt cells."""
        self.grid.set_fire(5, 5)
        new_data = update_grid(self.grid)
        self.assertEqual(new_data[5, 5], BURNT)

    def test_propagation(self):
        """Test that fire spreads to neighbors."""
        self.grid.set_fire(5, 5)
        new_data = update_grid(self.grid)
        
        neighbors = [(4,5), (6,5), (5,4), (5,6)]
        
        for r, c in neighbors:
            state = new_data[r, c]
            self.assertIn(state, [FUEL, BURNING])

    def test_burnt_stays_burnt(self):
        """Test that burnt cells do not reignite."""
        self.grid.data[5, 5] = BURNT
        self.grid.update_from_ghost()
        new_data = update_grid(self.grid)
        self.assertEqual(new_data[5, 5], BURNT)

if __name__ == '__main__':
    unittest.main()
