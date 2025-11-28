import argparse
import time
import os
import numpy as np
from src.mpi_comm import Communicator, MPI
from src.grid import Grid
from src.wildfire import update_grid
from src.load_balancer import LoadBalancer
from src.config import BURNING

def main():
"""Main function to run the distributed wildfire simulation."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Distributed Wildfire Simulation')
    parser.add_argument('--rows', type=int, default=100, help='Total rows')
    parser.add_argument('--cols', type=int, default=100, help='Total cols')
    parser.add_argument('--steps', type=int, default=100, help='Simulation steps')
    parser.add_argument('--balance', action='store_true', help='Enable dynamic load balancing')
    parser.add_argument('--procs', type=int, default=1, help='Number of processes (ignored, set by mpiexec)')
    parser.add_argument('--save', action='store_true', help='Save grid snapshots for visualization')
    parser.add_argument('--fire-pos', choices=['center', 'top'], default='center', help='Initial fire position')
    args = parser.parse_args()

    # Create directories for results if they don't exist
    if args.save:
        if not os.path.exists("results/logs"):
            os.makedirs("results/logs")
        if not os.path.exists("results/plots"):
            os.makedirs("results/plots")

    # Initialize MPI communicator
    comm_obj = Communicator()
    rank = comm_obj.rank
    size = comm_obj.size
    
    # Domain decomposition
    total_rows = args.rows
    rows_per_rank = total_rows // size
    remainder = total_rows % size
    
    # Distribute remainder
    local_rows = rows_per_rank + (1 if rank < remainder else 0)
    
    # Initialize Grid
    grid = Grid(local_rows, args.cols)

    # Initialize Fire    
    offset = sum([rows_per_rank + (1 if r < remainder else 0) for r in range(rank)])
    
    # Set initial fire position
    if args.fire_pos == 'center':
        global_center_r = total_rows // 2
        if offset <= global_center_r < offset + local_rows:
            local_r = global_center_r - offset
            grid.set_fire(local_r, args.cols // 2)

    elif args.fire_pos == 'top':
        if offset == 0 and local_rows > 0:
            grid.set_fire(0, args.cols // 2)
    
    # Load Balancer
    balancer = LoadBalancer(comm_obj) if args.balance else None
    
    # Main Loop
    start_time = time.time()
    
    for step in range(args.steps):
        # 1. Start Ghost Exchange
        requests = comm_obj.start_ghost_exchange(grid)
        
        # 2. Wait for Ghost Exchange
        comm_obj.end_ghost_exchange(grid, requests)
        
        # 3. Update Grid
        new_data = update_grid(grid)
        grid.commit_updates(new_data)
        
        # 4. Load Balancing (Every 5 steps)
        if args.balance and step % 5 == 0:
            balancer.redistribute(grid)
            
        # 5. Logging (Every 10 steps)
        if step % 10 == 0:
            total_burning = comm_obj.comm.reduce(np.sum(grid.data == BURNING), op=MPI.SUM, root=0)
            
            if args.save:
                full_grid = comm_obj.gather_grid(grid)
                if rank == 0:
                    np.save(f"results/logs/step_{step:03d}.npy", full_grid)
            
            if rank == 0:
                print(f"Step {step}: Total Burning = {total_burning}")
                
    end_time = time.time()
    
    if rank == 0:
        print(f"Simulation completed in {end_time - start_time:.4f} seconds.")

if __name__ == "__main__":
    main()
