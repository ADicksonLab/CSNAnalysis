import scipy
import numpy as np
from itertools import compress

def count_to_trans(countmat):
    """
    Converts a count matrix (in scipy sparse format) to a transition
    matrix.
    """
    tmp = np.array(countmat.toarray(),dtype=float)
    colsums = tmp.sum(axis=0)
    for i,c in enumerate(colsums):
        if c > 0:
            tmp[:,i] /= c

    return(scipy.sparse.coo_matrix(tmp))

def symmetrize_matrix(countmat):
    """
    Symmetrizes a count matrix (in scipy sparse format).
    """
    return scipy.sparse.coo_matrix(0.5*(countmat + countmat.transpose()))

def _make_sink(transmat,sink_states):
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

    # remove redundant elements in sink_states
    sink_states = list(set(sink_states))

    set_to_one = np.zeros(len(sink_states),dtype=bool)
    for i in range(len(sink_mat.data)):
        if sink_mat.col[i] in sink_states:
            if sink_mat.col[i] != sink_mat.row[i]:
                sink_mat.data[i] = 0.
            else:
                sink_mat.data[i] = 1.
                set_to_one[sink_states.index(sink_mat.col[i])] = True

    # set diagonal elements to 1 that haven't been set to one already
    statelist = np.asarray(list(compress(sink_states, np.logical_not(set_to_one))),
                           dtype=int)

    if statelist.shape[0] > 0:
        sink_mat.row = np.append(sink_mat.row, statelist)
        sink_mat.col = np.append(sink_mat.col,statelist)
        sink_mat.data = np.append(sink_mat.data, np.ones_like(statelist, dtype=int))

    # remove zeros
    sink_mat.eliminate_zeros()

    # check if sink_mat is well-conditioned
    if not well_conditioned(sink_mat.toarray()):
        raise ValueError("Error! sink matrix is no longer well-conditioned in make_sink!")

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

    banded_mat = _trans_mult_iter(transmat,tol)
    return banded_mat[:,0]

def _renorm(mat,tol=1e-7):
    """
    Renormalizes a matrix (to,from) so that its columns sum to one.
    This is meant to encourage numerical stability during long
    matrix multiplication chains.
    """
    for i in range(mat.shape[1]):
        col_sum = mat[:,i].sum()
        assert np.abs(1.0-col_sum) < tol, f"Error! 1 - column sum ({1.0-col_sum}) is greater than tolerance ({tol}) in _renorm!"
        mat[:,i] /= col_sum

    return mat

def fptd(transmat,sinks,maxsteps=200,tol=0.0):
    """
    Calculates the first passage time distribution for transmat with a set
    of sink states (sinks). The FPTD is evaluated at the set of points lag*2^i.  
    It will run for a total of maxsteps, or until the maximum "un-sunk" probability 
    of a state falls below tol.
    """

    sm = _renorm(_make_sink(transmat,sinks).toarray())

    step = 0
    non_sinks = [i for i in range(transmat.shape[0]) if i not in sinks]
    max_prob_to = sm.sum(axis=1)[non_sinks].max()

    fptd = []
    last_step_warped = np.zeros((transmat.shape[0]))
    while step < maxsteps and max_prob_to > tol:
        newmat = _renorm(np.matmul(sm,sm))
        
        warped = newmat[sinks,:].sum(axis=0)
        warped[sinks] = 0
        fptd.append(warped-last_step_warped)
        last_step_warped = warped
        
        sm = newmat.copy()
        max_prob_to = sm.sum(axis=1)[non_sinks].max()
        step += 1

    return np.array(fptd)
        

def _trans_mult_iter(transmat,tol,maxstep=200):
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
    while (var > tol) and (step < maxstep):
        newmat = np.matmul(t,t)
        var = np.abs(newmat-t).max()
        t = newmat.copy()
        step += 1

    if step == maxstep and var > tol:
        print("Warning: iterative multiplication not converged after",step,"steps: (var = ",var,"), (tol = ",tol,")")

    return t

def committor(transmat,basins,tol=1e-6,maxstep=20):
    """
    This function computes committor probabilities, given a transition matrix
    and a list of states that comprise the basins. It uses iterative multiplication of
    a modified transition matrix, with identity vectors for each basin state.

    Note that this method works regardless of the number of basins.

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

    flat_sink = [i for b in basins for i in b]
    sink_mat = _make_sink(transmat,flat_sink)
    sink_results = _trans_mult_iter(sink_mat,tol,maxstep)

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

def committor_linalg(transmat,basins):
    """
    This function computes committor probabilities, given a transition matrix
    and a list of states that comprise the basins, by solving the system
    of equations:

    0 = q_i - sum_j T_ij * q_j      for i not in a basin

    by solving the equation AQ = B.

    Note: this requires that the number of basins is 2, and q_i is the 
    probability that a trajectory in state i commits to the SECOND basin.

    Input:
    
    transmat -- An N x N transition matrix in scipy sparse coo format.  
                Columns should sum to 1. Indices: [to][from]

    basins -- A list of lists, describing which states make up the
              basins of attraction.  There can be any number of basins.
              e.g. [[basin1_a,basin1_b,...],[basin2_a,basin2_b,...]]

    Output:   An array of committor probabilities of size N x 2, where 2
              is the number of basins. Committors will sum to 1 for each state.
    """

    assert len(basins) == 2, 'Error! linalg method only works with two basins.'

    trans_arr = transmat.toarray()
    n = trans_arr.shape[0]
    A_mat = np.zeros((n,n))
    B_vec = np.zeros((n))

    for i in range(n):
        A_mat[i,i] = 1
        if i in basins[0]:
            B_vec[i] = 0
        elif i in basins[1]:
            B_vec[i] = 1
        else:
            B_vec[i] = 0
            for j in range(n):
                if i != j:
                    A_mat[i,j] = -trans_arr[j,i]
                else:
                    A_mat[i,i] = 1-trans_arr[j,i]

    Q_vec = np.linalg.solve(A_mat,B_vec)

    return np.array([1-Q_vec,Q_vec]).T

def _extend(transmat,hubstates):
    """
    This function returns an extended transition matrix (2N x 2N)
    where one set of states (0..N-1) have NOT yet visited hubstates,
    and states (N..2N-1) HAVE visited the hubstates.
    """
    n = transmat.shape[0]

    # data, rows and cols of the future extended matrix
    data = []
    rows = []
    cols = []

    for i in range(len(transmat.data)):
        if transmat.row[i] in hubstates:
            # transition TO a hubstate, add to lower left and lower right
            # lower left
            data.append(transmat.data[i])
            rows.append(transmat.row[i] + n)
            cols.append(transmat.col[i])
            # lower right
            data.append(transmat.data[i])
            rows.append(transmat.row[i] + n)
            cols.append(transmat.col[i] + n)
        else:
            # transition not to a hubstate, add to upper left and lower right
            # upper left
            data.append(transmat.data[i])
            rows.append(transmat.row[i])
            cols.append(transmat.col[i])
            # lower right
            data.append(transmat.data[i])
            rows.append(transmat.row[i] + n)
            cols.append(transmat.col[i] + n)

    ext_mat = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(2*n,2*n))
    return ext_mat

def _getring(transmat,basin,wts,tol,maxstep):
    """
    Given a transition matrix, and a set of states that form a basin,
    this returns a vector describing how probability exits that basin.
    """
    # make a matrix with sink states in every non-basin state
    n = transmat.shape[0]
    flat_sink = [i for i in range(n) if i not in basin]
    sink_mat = _make_sink(transmat,flat_sink)

    # see where the probability goes
    sink_results = _trans_mult_iter(sink_mat,tol,maxstep)

    ringprob = np.zeros((n))
    for b in basin:
        for i in range(n):
            if i not in basin:
                ringprob[i] += wts[b]*sink_results[i][b]

    return ringprob/wts[basin].sum()

def hubscores(transmat,hubstates,basins,tol=1e-6,maxstep=30,wts=None):
    """
    This function computes hub scores, which are the probabilities that
    transitions between a set of communities will use a given community as
    an intermediate.  e.g. h_a,b,c is the probability that transitions from
    basin a to basin b will use c as an intermediate.

    For more information see:
    Dickson, A and Brooks III, CL. JCTC, 8, 3044-3052 (2012).

    Input:

    transmat -- An N x N transition matrix in scipy sparse coo format.
                Columns should sum to 1. Indices: [to][from]

    hubstates -- A list describing the states in transmat that make up
              the hub being measured.

    basins -- A list of two lists, describing which two states make up the
              basins of attraction.
              e.g. [[basin_a_1,basin_a_2,...],[basin_b_1,basin_b_2,...]].

    wts    -- The equilibrium weights of all states in transmat.  If this is not
              given then the function will compute them from eig_weights.

    Output:   [h_a,b,c , h_b,a,c]
    """

    # make extended sink_matrix
    n = transmat.shape[0]
    ext_transmat = _extend(transmat,hubstates)

    flat_sink = [i for b in basins for i in b]
    flat_sink_ext = flat_sink + [i + n for i in flat_sink]

    sink_mat = _make_sink(ext_transmat,flat_sink_ext)

    sink_results = _trans_mult_iter(sink_mat,tol,maxstep)

    if wts is None:
        wts = eig_weights(transmat)


    h = np.zeros((2,2),dtype=float)
    ring = [_getring(transmat,b,wts,tol,maxstep) for b in basins]

    for source,sink in [[0,1],[1,0]]:
        for i,p in enumerate(ring[source]):
            if p > 0:
                # i is a ring state of source basin, with probability p
                if i in hubstates:
                    testi = i + n
                else:
                    testi = i
                c_no = 0
                c_yes = 0
                for b in basins[sink]:
                    c_no += sink_results[b][testi]
                    c_yes += sink_results[b+n][testi]
                if (c_no + c_yes) > 0:
                    h[source][sink] += p*c_yes/(c_no+c_yes)

    return [h[0,1],h[1,0]]

def get_eigenvectors(transmat, n_eig=3, return_wt_vec=False):
    """
    This function returns a set of eigenvectors with the highest
    eigenvalues. It wraps the scipy.linalg.eig function.

    Input:
    
    transmat -- An N x N transition matrix in scipy sparse coo format.
                Columns should sum to 1. Indices: [to][from]

    n_eig    -- The number of eigenvectors to return

    return_wt_vec -- Whether or not to include the eigenvector with
                     eigenvalue = 1.  Note that this is equal to the 
                     steady state weights.

    Output:

    eig_vecs -- A numpy array (N, n_eig) of eigenvector elements (real part only)

    eig_vals -- A numpy array of the n_eig eigenvalues (real part only)

    eig_vecs_imag -- A numpy array (N, n_eig) of eigenvector elements (imaginary part only)

    eig_vals_imag - A numpy array of the n_eig eigenvalues (imaginary part only)
    """

    e_vals_complex, e_vecs_complex = scipy.linalg.eig(transmat)

    e_vals_real = np.real(e_vals_complex)
    e_vals_imag = np.imag(e_vals_complex)

    sort_idxs = list(np.argsort(e_vals_real))

    if return_wt_vec:
        idxs_to_return = sort_idxs[-n_eig:]
    else:
        idxs_to_return = sort_idxs[-(n_eig+1):-1]

    # change order to highest to lowest
    idxs_to_return.reverse()

    return np.real(e_vecs_complex)[:,idxs_to_return], e_vals_real[idxs_to_return], \
        np.imag(e_vecs_complex)[:,idxs_to_return], e_vals_imag[idxs_to_return]

def well_conditioned(transmat):
    tol = 1e-5
    minval = transmat.sum(axis=0).min()
    maxval = transmat.sum(axis=0).max()
    if 1 - minval > tol or maxval - 1 > tol:
        return False
    else:
        return True
