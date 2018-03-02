
import scipy
import numpy as np

def make_sink(transmat,sink_states):
    """
    Constructs a transition matrix with "sink states", where the columns are
    replaced with identity vectors (diagonal element = 1, off-diagonals = 0).

    Input:

    transmat -- An N x N transition matrix in scipy sparse coo format.  
                Columns should sum to 1. Indices: [to][from].

    sink_states: A list of integers denoting sinks.

    Output:     A transition matrix in scipy sparse coo format.
    """
    sink_mat = transmat.copy()

    set_to_one = np.zeros(len(sink_states),dtype=bool)
    for i in range(len(sink_mat.data)):
        if sink_mat.col[i] in sink_states:
            if sink_mat.col[i] != sink_mat.row[i]:
                sink_mat.data[i] = 0.
            else:
                sink_mat.data[i] = 1.
                set_to_one[sink_states.index(sink_mat.col[i])] = True

    # set diagonal elements to 1
    for i in range(len(sink_states)):
        if not set_to_one[i]:
            # add element sink_mat[sink_states[i]][sink_states[i]] = 1
            np.append(sink_mat.row,sink_states[i])
            np.append(sink_mat.col,sink_states[i])
            np.append(sink_mat.data,1.)

    # remove zeros
    sink_mat.eliminate_zeros()

    return sink_mat
    
def eig_weights(transmat):
    """
    Calculates the weights as the top eigenvector of the transition matrix.

    Input:

    transmat -- An N x N transition matrix as a numpy array or in 
                scipy sparse coo format.  Columns should sum to 1. 
                Indices: [to][from]

    Output:     An array of weights of size N.
    """

    vals, vecs = scipy.sparse.linalg.eigs(transmat,k=1)
    return np.real(vecs[:,0])/np.real(vecs[:,0].sum())
    
def mult_weights(transmat,tol=1e-6):
    """
    Calculates the steady state weights as the columns of transmat^infinity.
    transmat^infinity is approximated by successively squaring transmat until
    the maximum variation in the rows is less than tol.

    Input:

    transmat -- An N x N transition matrix as a numpy array or in 
                scipy sparse coo format.  Columns should sum to 1. 
                Indices: [to][from]

    tol      -- Threshold for stopping the iterative multiplication.

    Output:     An array of weights of size N.
    """

    banded_mat = trans_mult_iter(transmat,tol)
    return banded_mat[:,0]

def trans_mult_iter(transmat,tol,maxstep=20):
    """
    Performs iterative multiplication of transmat until the maximum variation in 
    the rows is less than tol.
    """
    if type(transmat) is np.ndarray:
        t = transmat.copy()
    else:
        t = transmat.toarray()

    var = 1
    step = 0
    while var > tol or step > maxstep:
        newmat = np.matmul(t,t)
        var = np.abs(newmat-t).max()
        t = newmat.copy()
        step += 1

    if step > maxstep:
        print("Warning: iterative multiplication not converged after",maxstep,"steps: (var = ",var)

    return t

def committor(transmat,basins,tol=1e-6,maxstep=20):
    """
    This function computes committor probabilities, given a transition matrix
    and a list of states that comprise the basins.

    Input:
    
    transmat -- An N x N transition matrix in scipy sparse coo format.  
                Columns should sum to 1. Indices: [to][from]

    basins -- A list of lists, describing which states make up the
              basins of attraction.  There can be any number of basins.
              e.g. [[basin1_a,basin1_b,...],[basin2_a,basin2_b,...]]

    Output:   An array of committor probabilities of size N x B, where B
              is the number of basins. Committors will sum to 1 for each state.
    """

    # make sink_matrix
    sink_mat = make_sink(transmat,list(np.array(basins).flatten()))

    sink_results = trans_mult_iter(sink_mat,tol,maxstep)

    committor = np.zeros((transmat.shape[0],len(basins)),dtype=float)
    
    for i in range(transmat.shape[0]):
        comm_done = False
        for j,b in enumerate(basins):
            if i in b:
                committor[i][j] = 1
                comm_done = True
                break
        if not comm_done:
            for j,b in enumerate(basins):
                committor[i][j] = 0.
                for bstate in b:
                    committor[i][j] += sink_results[bstate][i]

    return committor

