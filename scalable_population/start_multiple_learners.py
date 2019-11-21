from mpi4py import MPI
import os
import sys

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank < int(sys.argv[1]):
        os.system(' '.join(['bash', 'start_learner_multiple_learners.bash', sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], str(rank)]))
    else:
        os.system(' '.join(['bash', 'start_actors_multiple_learners.bash', sys.argv[1], str(rank - int(sys.argv[1])), sys.argv[2], sys.argv[6]]))