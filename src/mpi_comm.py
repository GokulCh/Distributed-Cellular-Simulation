from mpi4py import MPI
import numpy as np

class Communicator:

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.up = self.rank - 1 if self.rank > 0 else MPI.PROC_NULL
        self.down = self.rank + 1 if self.rank < self.size - 1 else MPI.PROC_NULL

    def start_ghost_exchange(self, grid):

        if self.size == 1:
            return []

        requests = []
        
        send_up = grid.data[0, :].copy()
        send_down = grid.data[-1, :].copy()
        
        self.recv_up_buf = np.empty_like(send_up)
        self.recv_down_buf = np.empty_like(send_down)
        
        TAG_UP = 1
        TAG_DOWN = 2
        
        if self.up != MPI.PROC_NULL:
            req_s_up = self.comm.Isend(send_up, dest=self.up, tag=TAG_DOWN)
            requests.append(req_s_up)
            
            req_r_up = self.comm.Irecv(self.recv_up_buf, source=self.up, tag=TAG_UP)
            requests.append(req_r_up)
            
        if self.down != MPI.PROC_NULL:
            req_s_down = self.comm.Isend(send_down, dest=self.down, tag=TAG_UP)
            requests.append(req_s_down)
            
            req_r_down = self.comm.Irecv(self.recv_down_buf, source=self.down, tag=TAG_DOWN)
            requests.append(req_r_down)
            
        return requests

    def end_ghost_exchange(self, grid, requests):

        if self.size == 1:
            return

        if requests:
            MPI.Request.Waitall(requests)
            
        if self.up != MPI.PROC_NULL:
            grid.data_with_ghost[0, :] = self.recv_up_buf
            
        if self.down != MPI.PROC_NULL:
            grid.data_with_ghost[-1, :] = self.recv_down_buf

    def gather_grid(self, grid):
        local_rows = grid.rows
        all_rows = self.comm.gather(local_rows, root=0)
        
        local_data = grid.data
        
        if self.rank == 0:
            total_rows = sum(all_rows)
            full_grid = np.empty((total_rows, grid.cols), dtype=np.int8)
            
            displacements = [0]
            for r in all_rows[:-1]:
                displacements.append(displacements[-1] + r)
            
            counts = [r * grid.cols for r in all_rows]
            displacements_bytes = [d * grid.cols for d in displacements] 
            
            if self.size > 1:
                self.comm.Gatherv(local_data, [full_grid, counts, displacements_bytes, MPI.SIGNED_CHAR], root=0)
            else:
                full_grid[:] = local_data
            return full_grid
        else:
            self.comm.Gatherv(local_data, None, root=0)
            return None
