import numpy as np
from src.config import BURNING
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
        """
        Check load imbalance and migrate rows if necessary.
        Returns True if any migration occurred.
        """

        # If there are less than 2 ranks, no redistribution is possible
        if self.size < 2:
            return False

        # Sync before measuring load to ensure consistent state
        self.comm.Barrier()

        # Calculate local load
        local_load = int(self.check_imbalance(grid))

        TAG_LOAD = 10

        # Exchange load metrics with neighbors
        load_up = 0
        if self.comm_obj.up != MPI.PROC_NULL:
            load_up = self.comm.sendrecv(
                local_load,
                dest=self.comm_obj.up,
                source=self.comm_obj.up,
                sendtag=TAG_LOAD,
                recvtag=TAG_LOAD,
            )

        load_down = 0
        if self.comm_obj.down != MPI.PROC_NULL:
            load_down = self.comm.sendrecv(
                local_load,
                dest=self.comm_obj.down,
                source=self.comm_obj.down,
                sendtag=TAG_LOAD,
                recvtag=TAG_LOAD,
            )

        changed = False

        # Phase 1: Even ranks initiate with Down, Odd ranks initiate with Up
        # This prevents deadlock by ordering the exchanges
        if self.rank % 2 == 0 and self.comm_obj.down != MPI.PROC_NULL:
            self._balance_pair(grid, self.comm_obj.down)
        elif self.rank % 2 == 1 and self.comm_obj.up != MPI.PROC_NULL:
            self._balance_pair_passive(grid, self.comm_obj.up)

        self.comm.Barrier()

        # Phase 2: Odd ranks initiate with Down, Even ranks initiate with Up
        if self.rank % 2 == 1 and self.comm_obj.down != MPI.PROC_NULL:
            self._balance_pair(grid, self.comm_obj.down)
        elif (
            self.rank % 2 == 0 and self.rank != 0 and self.comm_obj.up != MPI.PROC_NULL
        ):
            self._balance_pair_passive(grid, self.comm_obj.up)

        return changed

    def _balance_pair(self, grid, other_rank):

        # Calculate local load
        my_load = int(self.check_imbalance(grid))

        TAG_BAL = 11

        # Exchange current load values
        self.comm.send(my_load, dest=other_rank, tag=TAG_BAL)

        THRESHOLD = 5

        TAG_CMD = 12

        # If I'm overloaded compared to neighbor, offload a row
        if my_load > other_load + THRESHOLD and grid.rows > 2:
            # Send bottom row to other rank
            self.comm.send(1, dest=other_rank, tag=TAG_CMD)  # Command: Take 1
            row_to_send = grid.data[-1, :].copy()
            self.comm.Send(row_to_send, dest=other_rank)
            # Remove row
            grid.data = grid.data[:-1, :]
            grid.rows -= 1
            grid.data_with_ghost = np.zeros((grid.rows + 2, grid.cols), dtype=np.int8)
            grid.update_from_ghost()

        elif other_load > my_load + THRESHOLD:
            # Take top row from other rank
            self.comm.send(-1, dest=other_rank, tag=TAG_CMD)  # Command: Give 1
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

        my_load = int(self.check_imbalance(grid))

        TAG_BAL = 11

        other_load = self.comm.recv(source=other_rank, tag=TAG_BAL)

        self.comm.send(my_load, dest=other_rank, tag=TAG_BAL)

        TAG_CMD = 12

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
