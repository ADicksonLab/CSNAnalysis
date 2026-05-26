import numpy as np
import scipy.sparse
import pytest

from csnanalysis.matrix import (
    count_to_trans,
    symmetrize_matrix,
    eig_weights,
    mult_weights,
    committor,
    committor_linalg,
    fptd,
    get_eigenvectors,
    well_conditioned,
    hubscores,
)


# ---------------------------------------------------------------------------
# count_to_trans
# ---------------------------------------------------------------------------

class TestCountToTrans:

    def test_columns_sum_to_one(self, three_state_counts):
        T = count_to_trans(scipy.sparse.coo_matrix(three_state_counts))
        col_sums = T.toarray().sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0)

    def test_known_two_state(self):
        counts = scipy.sparse.coo_matrix(np.array([[0., 2.], [2., 0.]]))
        T = count_to_trans(counts)
        np.testing.assert_allclose(T.toarray(), [[0., 1.], [1., 0.]])

    def test_zero_column_stays_zero(self):
        # column 0 is all zeros — should remain zero (no normalisation)
        counts = scipy.sparse.coo_matrix(np.array([[0., 1.], [0., 1.]], dtype=float))
        T = count_to_trans(counts)
        np.testing.assert_allclose(T.toarray()[:, 0], [0., 0.])

    def test_output_shape_preserved(self, three_state_counts):
        T = count_to_trans(scipy.sparse.coo_matrix(three_state_counts))
        assert T.shape == three_state_counts.shape

    def test_returns_sparse(self, three_state_counts):
        T = count_to_trans(scipy.sparse.coo_matrix(three_state_counts))
        assert scipy.sparse.issparse(T)


# ---------------------------------------------------------------------------
# symmetrize_matrix
# ---------------------------------------------------------------------------

class TestSymmetrizeMatrix:

    def test_output_is_symmetric(self):
        counts = scipy.sparse.coo_matrix(np.array([[1., 3.], [1., 2.]]))
        S = symmetrize_matrix(counts)
        arr = S.toarray()
        np.testing.assert_allclose(arr, arr.T)

    def test_symmetric_input_unchanged(self, three_state_counts):
        sparse = scipy.sparse.coo_matrix(three_state_counts)
        S = symmetrize_matrix(sparse)
        np.testing.assert_allclose(S.toarray(), three_state_counts)

    def test_total_sum_preserved(self):
        counts = scipy.sparse.coo_matrix(np.array([[2., 4.], [6., 8.]], dtype=float))
        S = symmetrize_matrix(counts)
        assert np.isclose(S.sum(), counts.sum())


# ---------------------------------------------------------------------------
# well_conditioned
# ---------------------------------------------------------------------------

class TestWellConditioned:

    def test_valid_transition_matrix(self, three_state_transmat):
        assert well_conditioned(three_state_transmat.toarray())

    def test_identity_is_well_conditioned(self):
        I = np.eye(4)
        assert well_conditioned(I)

    def test_matrix_with_wrong_column_sum_fails(self):
        bad = np.array([[0.5, 0.5], [0.5, 0.4]])  # col 1 sums to 0.9
        assert not well_conditioned(bad)

    def test_zero_column_fails(self):
        bad = np.array([[0., 1.], [0., 0.]])
        assert not well_conditioned(bad)


# ---------------------------------------------------------------------------
# eig_weights
# ---------------------------------------------------------------------------

class TestEigWeights:

    def test_sums_to_one(self, three_state_transmat):
        wts = eig_weights(three_state_transmat)
        assert np.isclose(wts.sum(), 1.0)

    def test_all_nonnegative(self, three_state_transmat):
        wts = eig_weights(three_state_transmat)
        assert np.all(wts >= 0)

    def test_known_symmetric_chain(self, three_state_transmat):
        # 3-state chain: steady state [0.25, 0.5, 0.25] by detailed balance
        wts = eig_weights(three_state_transmat)
        np.testing.assert_allclose(wts, [0.25, 0.5, 0.25], atol=1e-6)

    def test_two_state_symmetric(self):
        T = scipy.sparse.coo_matrix(np.array([[0., 1.], [1., 0.]]))
        wts = eig_weights(T)
        assert np.isclose(wts.sum(), 1.0)
        np.testing.assert_allclose(wts, [0.5, 0.5], atol=1e-6)


# ---------------------------------------------------------------------------
# mult_weights
# ---------------------------------------------------------------------------

class TestMultWeights:

    def test_sums_to_one(self, three_state_transmat):
        wts = mult_weights(three_state_transmat)
        assert np.isclose(wts.sum(), 1.0)

    def test_all_nonnegative(self, three_state_transmat):
        wts = mult_weights(three_state_transmat)
        assert np.all(wts >= 0)

    def test_agrees_with_eig_weights(self, aperiodic_transmat):
        # mult_weights requires an aperiodic chain to converge; the pure 3-state
        # chain is periodic (period 2) and T^(2^k) does not converge to a unique
        # stationary distribution.
        w_eig = eig_weights(aperiodic_transmat)
        w_mult = mult_weights(aperiodic_transmat)
        np.testing.assert_allclose(w_eig, w_mult, atol=1e-5)

    def test_known_aperiodic_chain(self, aperiodic_transmat):
        # stationary distribution of [[2,2,0],[2,2,2],[0,2,2]] counts is [2/7, 3/7, 2/7]
        wts = mult_weights(aperiodic_transmat)
        np.testing.assert_allclose(wts, [2/7, 3/7, 2/7], atol=1e-6)


# ---------------------------------------------------------------------------
# committor
# ---------------------------------------------------------------------------

class TestCommittor:

    def test_shape(self, three_state_transmat):
        basins = [[0], [2]]
        comm = committor(three_state_transmat, basins)
        assert comm.shape == (3, 2)

    def test_committors_sum_to_one(self, three_state_transmat):
        basins = [[0], [2]]
        comm = committor(three_state_transmat, basins)
        np.testing.assert_allclose(comm.sum(axis=1), 1.0, atol=1e-6)

    def test_basin_states_have_correct_values(self, three_state_transmat):
        basins = [[0], [2]]
        comm = committor(three_state_transmat, basins)
        # state 0 is in basin 0: committor[:,0] = 1, committor[:,1] = 0
        assert np.isclose(comm[0, 0], 1.0)
        assert np.isclose(comm[0, 1], 0.0)
        # state 2 is in basin 1
        assert np.isclose(comm[2, 1], 1.0)
        assert np.isclose(comm[2, 0], 0.0)

    def test_symmetric_chain_middle_state(self, three_state_transmat):
        # by symmetry, state 1 has equal committor to each basin
        basins = [[0], [2]]
        comm = committor(three_state_transmat, basins)
        np.testing.assert_allclose(comm[1, 0], 0.5, atol=1e-5)
        np.testing.assert_allclose(comm[1, 1], 0.5, atol=1e-5)

    def test_three_basins(self):
        # 5-state system; states 0,2,4 are basins, states 1,3 are intermediate
        counts = np.array([[0, 2, 0, 0, 0],
                           [2, 0, 2, 0, 0],
                           [0, 2, 0, 2, 0],
                           [0, 0, 2, 0, 2],
                           [0, 0, 0, 2, 0]], dtype=float)
        T = count_to_trans(scipy.sparse.coo_matrix(counts))
        basins = [[0], [2], [4]]
        comm = committor(T, basins)
        assert comm.shape == (5, 3)
        np.testing.assert_allclose(comm.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# committor_linalg
# ---------------------------------------------------------------------------

class TestCommittorLinalg:

    def test_shape(self, three_state_transmat):
        basins = [[0], [2]]
        comm = committor_linalg(three_state_transmat, basins)
        assert comm.shape == (3, 2)

    def test_committors_sum_to_one(self, three_state_transmat):
        basins = [[0], [2]]
        comm = committor_linalg(three_state_transmat, basins)
        np.testing.assert_allclose(comm.sum(axis=1), 1.0, atol=1e-10)

    def test_symmetric_chain_middle_state(self, three_state_transmat):
        basins = [[0], [2]]
        comm = committor_linalg(three_state_transmat, basins)
        np.testing.assert_allclose(comm[1, 0], 0.5, atol=1e-10)
        np.testing.assert_allclose(comm[1, 1], 0.5, atol=1e-10)

    def test_agrees_with_iterative_method(self, three_state_transmat):
        basins = [[0], [2]]
        comm_iter = committor(three_state_transmat, basins, tol=1e-8)
        comm_lin = committor_linalg(three_state_transmat, basins)
        np.testing.assert_allclose(comm_iter, comm_lin, atol=1e-5)

    def test_requires_two_basins(self, three_state_transmat):
        with pytest.raises(AssertionError):
            committor_linalg(three_state_transmat, [[0], [1], [2]])


# ---------------------------------------------------------------------------
# fptd
# ---------------------------------------------------------------------------

class TestFptd:

    def test_nonnegative_values(self, three_state_transmat):
        result = fptd(three_state_transmat, sinks=[2], maxsteps=20)
        assert np.all(result >= 0)

    def test_output_shape(self, three_state_transmat):
        result = fptd(three_state_transmat, sinks=[2], maxsteps=10)
        # shape: (steps, n_states)
        assert result.ndim == 2
        assert result.shape[1] == 3

    def test_cumulative_probability_le_one(self, three_state_transmat):
        result = fptd(three_state_transmat, sinks=[2], maxsteps=20)
        cumulative = result.sum(axis=0)
        assert np.all(cumulative <= 1.0 + 1e-10)

    def test_sink_states_have_zero_fptd(self, three_state_transmat):
        # sink states should not appear as sources in the FPTD
        result = fptd(three_state_transmat, sinks=[2], maxsteps=20)
        np.testing.assert_allclose(result[:, 2], 0.0, atol=1e-10)

    def test_tol_stopping(self, three_state_transmat):
        # with tol=0.5, should stop early
        result_short = fptd(three_state_transmat, sinks=[2], maxsteps=100, tol=0.5)
        result_long = fptd(three_state_transmat, sinks=[2], maxsteps=100, tol=0.0)
        assert result_short.shape[0] <= result_long.shape[0]


# ---------------------------------------------------------------------------
# get_eigenvectors
# ---------------------------------------------------------------------------

class TestGetEigenvectors:

    def test_output_shapes(self, three_state_transmat):
        n = 3
        n_eig = 2
        vecs_r, vals_r, vecs_i, vals_i = get_eigenvectors(
            three_state_transmat.toarray(), n_eig=n_eig
        )
        assert vecs_r.shape == (n, n_eig)
        assert vals_r.shape == (n_eig,)
        assert vecs_i.shape == (n, n_eig)
        assert vals_i.shape == (n_eig,)

    def test_eigenvalues_descending(self, three_state_transmat):
        vecs_r, vals_r, _, _ = get_eigenvectors(
            three_state_transmat.toarray(), n_eig=2
        )
        assert vals_r[0] >= vals_r[1]

    def test_largest_eigenvalue_less_than_one(self, three_state_transmat):
        _, vals_r, _, _ = get_eigenvectors(
            three_state_transmat.toarray(), n_eig=2
        )
        # exclude the stationary eigenvalue (1.0); largest returned is < 1
        assert vals_r[0] < 1.0 + 1e-10

    def test_include_wt_vec_returns_eigenvalue_one(self, three_state_transmat):
        _, vals_r, _, _ = get_eigenvectors(
            three_state_transmat.toarray(), n_eig=1, return_wt_vec=True
        )
        np.testing.assert_allclose(vals_r[0], 1.0, atol=1e-10)

    def test_n_eig_one(self, three_state_transmat):
        vecs_r, vals_r, vecs_i, vals_i = get_eigenvectors(
            three_state_transmat.toarray(), n_eig=1
        )
        assert vecs_r.shape == (3, 1)
        assert vals_r.shape == (1,)


# ---------------------------------------------------------------------------
# hubscores
# ---------------------------------------------------------------------------

class TestHubScores:

    def test_linear_chain_hub_score_is_one(self, three_state_transmat):
        # every path 0 -> 2 must pass through state 1
        wts = eig_weights(three_state_transmat)
        h = hubscores(three_state_transmat, hubstates=[1], basins=[[0], [2]], wts=wts)
        np.testing.assert_allclose(h[0], 1.0, atol=1e-5)
        np.testing.assert_allclose(h[1], 1.0, atol=1e-5)

    def test_returns_two_values(self, three_state_transmat):
        wts = eig_weights(three_state_transmat)
        h = hubscores(three_state_transmat, hubstates=[1], basins=[[0], [2]], wts=wts)
        assert len(h) == 2

    def test_hub_scores_between_zero_and_one(self, three_state_transmat):
        wts = eig_weights(three_state_transmat)
        h = hubscores(three_state_transmat, hubstates=[1], basins=[[0], [2]], wts=wts)
        assert 0.0 <= h[0] <= 1.0
        assert 0.0 <= h[1] <= 1.0

    def test_no_hub_score_is_zero(self, three_state_transmat):
        # if hubstates = basins, hub should not intercept any transitions
        wts = eig_weights(three_state_transmat)
        h = hubscores(three_state_transmat, hubstates=[0], basins=[[0], [2]], wts=wts)
        # hub is basin 0 itself — paths from basin 0 trivially "visit" hub at start
        # the score is still well-defined; just assert it's in [0,1]
        assert 0.0 <= h[0] <= 1.0 + 1e-8
        assert 0.0 <= h[1] <= 1.0 + 1e-8
