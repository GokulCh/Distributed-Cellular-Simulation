import numpy as np
from src.config import BURNING
from src.mpi_comm import MPI

class LoadBalancer:
    """Load balancer class."""
    def __init__(self, communicator):
        self.comm_obj = communicator
        self.comm = communicator.comm
        self.rank = communicator.rank
        self.size = communicator.size

    def check_imbalance(self, grid):
        """Calculate local load (number of burning cells)."""
        local_load = np.sum(grid.data == BURNING)
        return local_load

    def redistribute(self, grid):
        """Attempt to balance load by moving rows between neighbors."""

        # If there are less than 2 ranks, no redistribution is possible
        if self.size < 2:
            return False

        # Ensure everyone enters together
        self.comm.Barrier() 

        # Calculate local load
        local_load = int(self.check_imbalance(grid))
        
        TAG_LOAD = 10
        
        # Exchange loads with upper neighbor
        load_up = 0
        if self.comm_obj.up != MPI.PROC_NULL:
            load_up = self.comm.sendrecv(local_load, dest=self.comm_obj.up, source=self.comm_obj.up, sendtag=TAG_LOAD, recvtag=TAG_LOAD)
            
        # Exchange loads with lower neighbor
        load_down = 0
        if self.comm_obj.down != MPI.PROC_NULL:
            load_down = self.comm.sendrecv(local_load, dest=self.comm_obj.down, source=self.comm_obj.down, sendtag=TAG_LOAD, recvtag=TAG_LOAD)
        
        changed = False
        
        # Phase 1: Even boundaries
        if self.rank % 2 == 0 and self.comm_obj.down != MPI.PROC_NULL:
            self._balance_pair(grid, self.comm_obj.down)
        elif self.rank % 2 == 1 and self.comm_obj.up != MPI.PROC_NULL:
            self._balance_pair_passive(grid, self.comm_obj.up)
            
        self.comm.Barrier()
        
        # Phase 2: Odd boundaries
        if self.rank % 2 == 1 and self.comm_obj.down != MPI.PROC_NULL:
            self._balance_pair(grid, self.comm_obj.down)
        elif self.rank % 2 == 0 and self.rank != 0 and self.comm_obj.up != MPI.PROC_NULL:
             self._balance_pair_passive(grid, self.comm_obj.up)
             
        return changed

    def _balance_pair(self, grid, other_rank):
        """Balance load between two ranks."""
        
        # Calculate local load
        my_load = int(self.check_imbalance(grid))
        
        TAG_BAL = 11

        # Send load to other rank
        self.comm.send(my_load, dest=other_rank, tag=TAG_BAL)
        
        # Receive other rank's load
        other_load = self.comm.recv(source=other_rank, tag=TAG_BAL)
        
        THRESHOLD = 5
        
        TAG_CMD = 12
        
        # If load is greater than other rank's load + threshold and I have more than 2 rows
        if my_load > other_load + THRESHOLD and grid.rows > 2:
            # Send bottom row to other rank
            self.comm.send(1, dest=other_rank, tag=TAG_CMD)
            row_to_send = grid.data[-1, :].copy()
            self.comm.Send(row_to_send, dest=other_rank)
            # Remove row
            grid.data = grid.data[:-1, :]
            grid.rows -= 1
            grid.data_with_ghost = np.zeros((grid.rows + 2, grid.cols), dtype=np.int8)
            grid.update_from_ghost() 
            
        elif other_load > my_load + THRESHOLD:
            # Take top row from other rank
            self.comm.send(-1, dest=other_rank, tag=TAG_CMD)
            recv_buf = np.empty(grid.cols, dtype=np.int8)
            self.comm.Recv(recv_buf, source=other_rank)
            # Add row
            grid.data = np.vstack((grid.data, recv_buf))
            grid.rows += 1
            # Resize ghost
            grid.data_with_ghost = np.zeros((grid.rows + 2, grid.cols), dtype=np.int8)
            grid.update_from_ghost()
        else:
            self.comm.send(0, dest=other_rank, tag=TAG_CMD)

    def _balance_pair_passive(self, grid, other_rank):
        """Balance load between two ranks."""

        # Receive load from other rank
        my_load = int(self.check_imbalance(grid))   
        
        TAG_BAL = 11

        # Receive other rank's load
        other_load = self.comm.recv(source=other_rank, tag=TAG_BAL)
        
        # Send my load to other rank
        self.comm.send(my_load, dest=other_rank, tag=TAG_BAL)
        
        TAG_CMD = 12

        # Receive command from other rank
        command = self.comm.recv(source=other_rank, tag=TAG_CMD)
        
        # Handle command
        if command == 1:
            recv_buf = np.empty(grid.cols, dtype=np.int8)
            self.comm.Recv(recv_buf, source=other_rank)
            grid.data = np.vstack((recv_buf, grid.data))
            grid.rows += 1
        elif command == -1:
            if grid.rows > 2:
                row_to_send = grid.data[0, :].copy()
                self.comm.Send(row_to_send, dest=other_rank)
                grid.data = grid.data[1:, :]
                grid.rows -= 1
            else:
                pass
        
        # Resize ghost structure
        if command != 0:
            grid.data_with_ghost = np.zeros((grid.rows + 2, grid.cols), dtype=np.int8)
            grid.update_from_ghost()
