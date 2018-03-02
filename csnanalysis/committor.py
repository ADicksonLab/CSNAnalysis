
import scipy
import numpy as np

def committor(matrix,basins,tol=1e-6):
    """
    This function computes committor probabilities, given a transition matrix
    and a list of states that comprise the basins.

    Input:
    
    matrix -- An N x N transition matrix in scipy sparse coo format.  
              Columns should sum to 1. Indices: [from][to]

    basins -- A list of lists, describing which states make up the
              basins of attraction.  There can be any number of basins.
              e.g. [[basin1_a,basin1_b,...],[basin2_a,basin2_b,...]]

    Output:   An array of committor probabilities of size N x B, where B
              is the number of basins. Committors will sum to 1 for each state.
    """

    return
