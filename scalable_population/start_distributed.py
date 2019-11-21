from mpi4py import MPI
import os
import sys

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        pass
        #os.system(' '.join(['bash', 'start_learner.bash', sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]]))
    else:
        os.system(' '.join(['bash', 'start_actors.bash', sys.argv[1], str(rank - 1), sys.argv[2], sys.argv[6]]))