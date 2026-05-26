import numpy as np
import scipy.sparse
import pytest

from csnanalysis.csn import CSN
from csnanalysis.matrix import count_to_trans


@pytest.fixture
def three_state_counts():
    """3-state linear chain 0 <-> 1 <-> 2, no self-loops.

    T[:,0] = [0, 1, 0]   (from 0 -> always 1)
    T[:,1] = [0.5, 0, 0.5]  (from 1 -> 0 or 2 equally)
    T[:,2] = [0, 1, 0]   (from 2 -> always 1)

    Steady state: [0.25, 0.5, 0.25]
    Committor to basin {2} (with basin {0}): state 1 = 0.5
    """
    return np.array([[0, 2, 0],
                     [2, 0, 2],
                     [0, 2, 0]], dtype=float)


@pytest.fixture
def three_state_transmat(three_state_counts):
    return count_to_trans(scipy.sparse.coo_matrix(three_state_counts))


@pytest.fixture
def disconnected_counts():
    """4-state system: pair {0,1} disconnected from pair {2,3}.
    MSN is node 0 (ties broken by argmax); trim should keep only {0,1}.
    """
    return np.array([[0, 3, 0, 0],
                     [3, 0, 0, 0],
                     [0, 0, 0, 3],
                     [0, 0, 3, 0]], dtype=float)


@pytest.fixture
def aperiodic_three_state_counts():
    """3-state chain with self-loops, breaking the period-2 symmetry.

    T[:,0] = [0.5, 0.5, 0]     T[:,1] = [1/3, 1/3, 1/3]   T[:,2] = [0, 0.5, 0.5]

    Stationary distribution (via detailed balance): [2/7, 3/7, 2/7]
    mult_weights converges correctly on this matrix (it is aperiodic).
    """
    return np.array([[2, 2, 0],
                     [2, 2, 2],
                     [0, 2, 2]], dtype=float)


@pytest.fixture
def aperiodic_transmat(aperiodic_three_state_counts):
    return count_to_trans(scipy.sparse.coo_matrix(aperiodic_three_state_counts))


@pytest.fixture
def simple_csn(three_state_counts):
    return CSN(three_state_counts)


@pytest.fixture
def disconnected_csn(disconnected_counts):
    return CSN(disconnected_counts)


@pytest.fixture
def trimmed_csn(simple_csn):
    simple_csn.trim()
    return simple_csn


@pytest.fixture
def aperiodic_csn(aperiodic_three_state_counts):
    csn = CSN(aperiodic_three_state_counts)
    csn.trim()
    return csn
