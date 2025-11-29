import numpy as np
from src.config import BURNING, TAG_LOAD, TAG_BAL, TAG_CMD, LB_THRESHOLD
from src.mpi_comm import MPI

class LoadBalancer:
    def __init__(self, communicator):
        self.comm_obj = communicator
        self.comm = communicator.comm
        self.rank = communicator.rank
        self.size = communicator.size

    def check_imbalance(self, grid):
        local_load = np.sum(grid.data == BURNING)
        return local_load

    def redistribute(self, grid):

        if self.size < 2:
            return False

        self.comm.Barrier() 
        local_load = int(self.check_imbalance(grid))
        
        changed = False
        
        # Balance pair
        if self.rank % 2 == 0 and self.comm_obj.down != MPI.PROC_NULL:
            self._balance_pair(grid, self.comm_obj.down)
        elif self.rank % 2 == 1 and self.comm_obj.up != MPI.PROC_NULL:
            self._balance_pair_passive(grid, self.comm_obj.up)

        self.comm.Barrier()
        
        # Balance 
        if self.rank % 2 == 1 and self.comm_obj.down != MPI.PROC_NULL:
            self._balance_pair(grid, self.comm_obj.down)
        elif self.rank % 2 == 0 and self.rank != 0 and self.comm_obj.up != MPI.PROC_NULL:
             self._balance_pair_passive(grid, self.comm_obj.up)
             
        return changed

    def _balance_pair(self, grid, other_rank):
        my_load = int(self.check_imbalance(grid))
        
        my_load = int(self.check_imbalance(grid))
        
        self.comm.send(my_load, dest=other_rank, tag=TAG_BAL)
        other_load = self.comm.recv(source=other_rank, tag=TAG_BAL)
        
        if my_load > other_load + LB_THRESHOLD and grid.rows > 2:
            self.comm.send(1, dest=other_rank, tag=TAG_CMD)
            row_to_send = grid.data[-1, :].copy()
            self.comm.Send(row_to_send, dest=other_rank)
            grid.data = grid.data[:-1, :]
            grid.rows -= 1
            grid.data_with_ghost = np.zeros((grid.rows + 2, grid.cols), dtype=np.int8)
            grid.update_from_ghost() 
            
        elif other_load > my_load + LB_THRESHOLD:
            self.comm.send(-1, dest=other_rank, tag=TAG_CMD)
            recv_buf = np.empty(grid.cols, dtype=np.int8)
            self.comm.Recv(recv_buf, source=other_rank)
            grid.data = np.vstack((grid.data, recv_buf))
            grid.rows += 1
            grid.data_with_ghost = np.zeros((grid.rows + 2, grid.cols), dtype=np.int8)
            grid.update_from_ghost()
        else:
            self.comm.send(0, dest=other_rank, tag=TAG_CMD)

    def _balance_pair_passive(self, grid, other_rank):

        my_load = int(self.check_imbalance(grid))       
        other_load = self.comm.recv(source=other_rank, tag=TAG_BAL)

        self.comm.send(my_load, dest=other_rank, tag=TAG_BAL)
        
        command = self.comm.recv(source=other_rank, tag=TAG_CMD)
        
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
        
        if command != 0:
            grid.data_with_ghost = np.zeros((grid.rows + 2, grid.cols), dtype=np.int8)
            grid.update_from_ghost()
