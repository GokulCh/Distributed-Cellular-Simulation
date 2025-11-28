import unittest
from src.grid import Grid
from src.load_balancer import LoadBalancer
from tests.mocks import MockComm, MPI

class MockCommunicatorWrapper:
    def __init__(self):
        self.comm = MockComm()
        self.rank = 0
        self.size = 1
        self.up = MPI.PROC_NULL
        self.down = MPI.PROC_NULL

class TestLoadBalancer(unittest.TestCase):
    def setUp(self):
        self.comm = MockCommunicatorWrapper()
        self.balancer = LoadBalancer(self.comm)
        self.grid = Grid(10, 10)

    def test_check_imbalance(self):
        self.grid.set_fire(5, 5)
        self.grid.set_fire(5, 6)
        load = self.balancer.check_imbalance(self.grid)
        self.assertEqual(load, 2)

    def test_redistribute_no_mpi(self):
        changed = self.balancer.redistribute(self.grid)
        self.assertFalse(changed)

if __name__ == '__main__':
    unittest.main()
