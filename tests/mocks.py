class MockMPI:
    """Mock MPI class for testing."""
    COMM_WORLD = None 
    PROC_NULL = -2
    SUM = lambda x, y: x + y
    INT8 = None
    class Request:
        @staticmethod
        def Waitall(requests): pass
    
MPI = MockMPI()

class MockComm:
    """Mock MPI communicator for testing."""
    def __init__(self):
        self.rank = 0
        self.size = 1
        self.up = -1
        self.down = -1
        
    def Get_rank(self): return 0
    def Get_size(self): return 1
    def Barrier(self): pass
    def Sendrecv(self, sendbuf, dest, recvbuf, source): pass
    def gather(self, sendobj, root=0): return [sendobj]
    def allgather(self, sendobj): return [sendobj]
    def Gatherv(self, sendbuf, args, root=0):
        recvbuf = args[0]
        recvbuf[:] = sendbuf
    def reduce(self, sendobj, op, root=0): return sendobj
    def send(self, obj, dest): pass
    def recv(self, source): return 0
    def Send(self, buf, dest): pass
    def Recv(self, buf, source): pass
    def Isend(self, buf, dest, tag=0): return None
    def Irecv(self, buf, source, tag=0): return None
