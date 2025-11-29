import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mpi_comm import Communicator
from src.grid import Grid
from src.load_balancer import LoadBalancer
from src.config import BURNING

def test_load_balancing():
    comm_obj = Communicator()
    rank = comm_obj.rank
    size = comm_obj.size
    
    if size < 2:
        if rank == 0:
            print("Skipping load balancer test (needs > 1 process)")
        return

    # Create a small grid
    rows = 10
    cols = 10
    grid = Grid(rows, cols)
    
    # Create imbalance: Rank 0 has fire, others don't
    if rank == 0:
        grid.data[:] = BURNING
    
    balancer = LoadBalancer(comm_obj)
    
    initial_rows = grid.rows
    comm_obj.comm.Barrier()
    if rank == 0:
        print(f"Initial Rows: Rank {rank} = {initial_rows}")
    comm_obj.comm.Barrier()
    
    # Run redistribute multiple times to allow propagation
    for i in range(5):
        balancer.redistribute(grid)
        
    final_rows = grid.rows
    
    # Gather results
    all_initial = comm_obj.comm.gather(initial_rows, root=0)
    all_final = comm_obj.comm.gather(final_rows, root=0)
    
    if rank == 0:
        print(f"Final Rows: {all_final}")
        
        # Verification logic
        # Rank 0 should have fewer rows, Rank 1 should have more
        if all_final[0] < all_initial[0]:
            print("SUCCESS: Rank 0 shed load.")
        else:
            print("FAILURE: Rank 0 did not shed load.")
            
        if all_final[1] > all_initial[1]:
            print("SUCCESS: Rank 1 accepted load.")
        else:
            print("FAILURE: Rank 1 did not accept load.")
            
        if sum(all_final) == sum(all_initial):
             print("SUCCESS: Total rows conserved.")
        else:
             print(f"FAILURE: Total rows changed! {sum(all_initial)} -> {sum(all_final)}")

if __name__ == "__main__":
    test_load_balancing()
