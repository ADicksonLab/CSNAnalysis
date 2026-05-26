import numpy as np
import scipy.sparse
import pytest

from csnanalysis.csn import CSN


# ---------------------------------------------------------------------------
# CSN.__init__
# ---------------------------------------------------------------------------

class TestCSNInit:

    def test_from_numpy_array(self, three_state_counts):
        csn = CSN(three_state_counts)
        assert csn.nnodes == 3

    def test_from_list_of_lists(self):
        counts = [[0, 2, 0], [2, 0, 2], [0, 2, 0]]
        csn = CSN(counts)
        assert csn.nnodes == 3

    def test_from_sparse_coo(self, three_state_counts):
        sparse = scipy.sparse.coo_matrix(three_state_counts)
        csn = CSN(sparse)
        assert csn.nnodes == 3

    def test_from_sparse_csr(self, three_state_counts):
        sparse = scipy.sparse.csr_matrix(three_state_counts)
        csn = CSN(sparse)
        assert csn.nnodes == 3

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            CSN(np.array([[1., 2., 3.], [4., 5., 6.]]))

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            CSN("not a matrix")

    def test_trim_transmat_initially_none(self, simple_csn):
        assert simple_csn.trim_transmat is None

    def test_graph_has_correct_node_count(self, simple_csn):
        assert simple_csn.graph.number_of_nodes() == 3

    def test_graph_has_no_self_edges(self, simple_csn):
        for n in simple_csn.graph.nodes:
            assert not simple_csn.graph.has_edge(n, n)

    def test_symmetrize_flag(self):
        # asymmetric counts
        counts = np.array([[0., 3.], [1., 0.]])
        csn_sym = CSN(counts, symmetrize=True)
        arr = csn_sym.countmat.toarray()
        np.testing.assert_allclose(arr, arr.T)

    def test_node_count_attribute(self, simple_csn):
        # each node should carry the column-sum as 'count'
        for i in range(simple_csn.nnodes):
            assert 'count' in simple_csn.graph.nodes[i]


# ---------------------------------------------------------------------------
# CSN.trim
# ---------------------------------------------------------------------------

class TestTrim:

    def test_fully_connected_graph_unchanged(self, simple_csn):
        simple_csn.trim()
        assert simple_csn.trim_nnodes == 3

    def test_disconnected_graph_trimmed(self, disconnected_csn):
        disconnected_csn.trim()
        # MSN is node 0; only nodes 0 and 1 are reachable
        assert disconnected_csn.trim_nnodes == 2

    def test_trim_indices_subset_of_full(self, disconnected_csn):
        disconnected_csn.trim()
        assert set(disconnected_csn.trim_indices).issubset(set(range(4)))

    def test_trim_transmat_columns_sum_to_one(self, simple_csn):
        simple_csn.trim()
        col_sums = simple_csn.trim_transmat.toarray().sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_trim_graph_node_count(self, disconnected_csn):
        disconnected_csn.trim()
        assert disconnected_csn.trim_graph.number_of_nodes() == 2

    def test_min_count_removes_low_count_nodes(self):
        # node 2 has only 1 total count
        counts = np.array([[0., 5., 0.],
                           [5., 0., 1.],
                           [0., 1., 0.]], dtype=float)
        csn = CSN(counts)
        csn.trim(min_count=2)
        assert csn.trim_nnodes == 2
        assert 2 not in csn.trim_indices

    def test_trim_adds_is_trim_attribute(self, disconnected_csn):
        disconnected_csn.trim()
        # trimmed-away nodes should have is_trim=1
        for node in range(4):
            if node not in disconnected_csn.trim_indices:
                assert disconnected_csn.graph.nodes[node]['trim'] == 1


# ---------------------------------------------------------------------------
# CSN.add_attr / add_trim_attr
# ---------------------------------------------------------------------------

class TestAddAttr:

    def test_add_attr_stores_values(self, simple_csn):
        vals = [1.0, 2.0, 3.0]
        simple_csn.add_attr('test_attr', vals)
        for i, v in enumerate(vals):
            assert simple_csn.graph.nodes[i]['test_attr'] == v

    def test_add_trim_attr_stores_values(self, trimmed_csn):
        n_trim = trimmed_csn.trim_nnodes
        vals = list(range(n_trim))
        trimmed_csn.add_trim_attr('trim_test', vals)
        for i, idx in enumerate(trimmed_csn.trim_indices):
            assert trimmed_csn.graph.nodes[idx]['trim_test'] == vals[i]

    def test_add_trim_attr_default_for_nontrim_nodes(self, disconnected_csn):
        disconnected_csn.trim()
        n_trim = disconnected_csn.trim_nnodes
        vals = [99.0] * n_trim
        disconnected_csn.add_trim_attr('check_default', vals, default=-1)
        for node in range(4):
            if node not in disconnected_csn.trim_indices:
                assert disconnected_csn.graph.nodes[node]['check_default'] == -1


# ---------------------------------------------------------------------------
# CSN.calc_eig_weights
# ---------------------------------------------------------------------------

class TestCalcEigWeights:

    def test_weights_sum_to_one(self, trimmed_csn):
        wts = trimmed_csn.calc_eig_weights()
        assert np.isclose(wts.sum(), 1.0)

    def test_weights_nonnegative(self, trimmed_csn):
        wts = trimmed_csn.calc_eig_weights()
        assert np.all(wts >= 0)

    def test_weights_stored_as_node_attribute(self, trimmed_csn):
        trimmed_csn.calc_eig_weights(label='eig_weights')
        for node in trimmed_csn.graph.nodes:
            assert 'eig_weights' in trimmed_csn.graph.nodes[node]

    def test_known_steady_state(self, trimmed_csn):
        # 3-state chain steady state: [0.25, 0.5, 0.25]
        wts = trimmed_csn.calc_eig_weights()
        np.testing.assert_allclose(wts, [0.25, 0.5, 0.25], atol=1e-6)

    def test_without_trimming(self, simple_csn):
        wts = simple_csn.calc_eig_weights()
        assert np.isclose(wts.sum(), 1.0)


# ---------------------------------------------------------------------------
# CSN.calc_mult_weights
# ---------------------------------------------------------------------------

class TestCalcMultWeights:

    def test_weights_sum_to_one(self, trimmed_csn):
        wts = trimmed_csn.calc_mult_weights()
        assert np.isclose(wts.sum(), 1.0)

    def test_weights_nonnegative(self, trimmed_csn):
        wts = trimmed_csn.calc_mult_weights()
        assert np.all(wts >= 0)

    def test_agrees_with_eig_weights(self, aperiodic_csn):
        # mult_weights requires an aperiodic chain; the pure 3-state chain is
        # periodic so we use the aperiodic fixture here.
        wts_eig = aperiodic_csn.calc_eig_weights()
        wts_mult = aperiodic_csn.calc_mult_weights()
        np.testing.assert_allclose(wts_eig, wts_mult, atol=1e-5)

    def test_label_none_skips_graph_storage(self, trimmed_csn):
        trimmed_csn.calc_mult_weights(label=None)
        for node in trimmed_csn.graph.nodes:
            assert 'mult_weights' not in trimmed_csn.graph.nodes[node]


# ---------------------------------------------------------------------------
# CSN.calc_committors
# ---------------------------------------------------------------------------

class TestCalcCommittors:

    def test_returns_correct_shape(self, trimmed_csn):
        basins = [[0], [2]]
        comm = trimmed_csn.calc_committors(basins)
        assert comm.shape == (3, 2)

    def test_committors_sum_to_one(self, trimmed_csn):
        basins = [[0], [2]]
        comm = trimmed_csn.calc_committors(basins)
        np.testing.assert_allclose(comm.sum(axis=1), 1.0, atol=1e-5)

    def test_basin_states_correct(self, trimmed_csn):
        basins = [[0], [2]]
        comm = trimmed_csn.calc_committors(basins)
        assert np.isclose(comm[0, 0], 1.0, atol=1e-5)
        assert np.isclose(comm[2, 1], 1.0, atol=1e-5)

    def test_symmetric_middle_state(self, trimmed_csn):
        basins = [[0], [2]]
        comm = trimmed_csn.calc_committors(basins)
        np.testing.assert_allclose(comm[1, 0], 0.5, atol=1e-5)

    def test_linalg_method(self, trimmed_csn):
        basins = [[0], [2]]
        comm = trimmed_csn.calc_committors(basins, method='linalg')
        np.testing.assert_allclose(comm.sum(axis=1), 1.0, atol=1e-10)
        np.testing.assert_allclose(comm[1, 0], 0.5, atol=1e-10)

    def test_iter_and_linalg_agree(self, trimmed_csn):
        basins = [[0], [2]]
        comm_iter = trimmed_csn.calc_committors(basins, method='iter', tol=1e-8)
        comm_lin = trimmed_csn.calc_committors(basins, method='linalg')
        np.testing.assert_allclose(comm_iter, comm_lin, atol=1e-5)

    def test_committors_stored_as_attributes(self, trimmed_csn):
        basins = [[0], [2]]
        trimmed_csn.calc_committors(basins, labels=['p0', 'p1'])
        for node in trimmed_csn.graph.nodes:
            assert 'p0' in trimmed_csn.graph.nodes[node]
            assert 'p1' in trimmed_csn.graph.nodes[node]

    def test_invalid_method_raises(self, trimmed_csn):
        with pytest.raises(AssertionError):
            trimmed_csn.calc_committors([[0], [2]], method='bad_method')

    def test_add_basins_flag(self, trimmed_csn):
        basins = [[0], [2]]
        trimmed_csn.calc_committors(basins, add_basins=True,
                                    basin_labels=['basin_A', 'basin_B'])
        assert 'basin_A' in trimmed_csn.graph.nodes[0]
        assert 'basin_B' in trimmed_csn.graph.nodes[2]


# ---------------------------------------------------------------------------
# CSN.calc_mfpt
# ---------------------------------------------------------------------------

class TestCalcMfpt:

    def test_returns_mfpt_and_fptd(self, trimmed_csn):
        mfpt, fptd_out = trimmed_csn.calc_mfpt(sinks=[2], maxsteps=30, tol=1e-3)
        assert mfpt.shape == (3,)
        assert fptd_out.ndim == 2

    def test_sink_mfpt_is_zero(self, trimmed_csn):
        mfpt, _ = trimmed_csn.calc_mfpt(sinks=[2], maxsteps=30, tol=1e-3)
        assert mfpt[2] == 0.0

    def test_mfpt_nonnegative(self, trimmed_csn):
        mfpt, _ = trimmed_csn.calc_mfpt(sinks=[2], maxsteps=30, tol=1e-3)
        assert np.all(mfpt >= 0)

    def test_nonsink_states_have_positive_mfpt(self, trimmed_csn):
        # Both non-sink states must have a positive MFPT to reach state 2.
        # (The exponential time-stepping approximation can make states 0 and 1
        # numerically equal on this symmetric chain, so ordering is not tested.)
        mfpt, _ = trimmed_csn.calc_mfpt(sinks=[2], maxsteps=50, tol=1e-4)
        assert mfpt[0] > 0
        assert mfpt[1] > 0

    def test_sources_returns_averaged_mfpt(self, trimmed_csn):
        mfpt, fptd_out = trimmed_csn.calc_mfpt(
            sinks=[2], maxsteps=50, tol=1e-4, sources=[0, 1]
        )
        assert mfpt.shape == (1,)
        # fptd_out shape is (1, n_steps): first axis is the single averaged source
        assert fptd_out.shape[0] == 1

    def test_maxsteps_or_tol_required(self, trimmed_csn):
        with pytest.raises(AssertionError):
            trimmed_csn.calc_mfpt(sinks=[2], maxsteps=None, tol=None)


# ---------------------------------------------------------------------------
# CSN.calc_eigvectors
# ---------------------------------------------------------------------------

class TestCalcEigvectors:

    def test_output_shapes(self, trimmed_csn):
        n_eig = 2
        vr, valr, vi, vali = trimmed_csn.calc_eigvectors(n_eig=n_eig)
        assert vr.shape == (3, n_eig)
        assert valr.shape == (n_eig,)

    def test_eigenvalues_descending(self, trimmed_csn):
        vr, valr, vi, vali = trimmed_csn.calc_eigvectors(n_eig=2)
        assert valr[0] >= valr[1]

    def test_eigenvectors_stored_as_attributes(self, trimmed_csn):
        trimmed_csn.calc_eigvectors(n_eig=2, save_label='eig')
        for node in trimmed_csn.graph.nodes:
            assert 'eig0' in trimmed_csn.graph.nodes[node]
            assert 'eig1' in trimmed_csn.graph.nodes[node]

    def test_imaginary_parts_stored_when_requested(self, trimmed_csn):
        trimmed_csn.calc_eigvectors(n_eig=2, save_imag_to_graph=True, save_label='eig')
        for node in trimmed_csn.graph.nodes:
            assert 'eig0_imag' in trimmed_csn.graph.nodes[node]

    def test_include_wt_vec(self, trimmed_csn):
        vr, valr, vi, vali = trimmed_csn.calc_eigvectors(
            n_eig=1, include_wt_vec=True
        )
        np.testing.assert_allclose(valr[0], 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# CSN.idxs_to_trim
# ---------------------------------------------------------------------------

class TestIdxsToTrim:

    def test_converts_full_to_trim_indices(self, disconnected_csn):
        disconnected_csn.trim()
        trim_idxs = disconnected_csn.idxs_to_trim([0, 1])
        assert len(trim_idxs) == 2

    def test_ignores_untrimmed_nodes(self, disconnected_csn):
        disconnected_csn.trim()
        # nodes 2 and 3 were trimmed away; they should be ignored
        trim_idxs = disconnected_csn.idxs_to_trim([0, 1, 2, 3])
        assert 2 not in trim_idxs
        assert 3 not in trim_idxs


# ---------------------------------------------------------------------------
# CSN.colors_from_committors
# ---------------------------------------------------------------------------

class TestColorsFromCommittors:

    def test_returns_rgb_dict(self, simple_csn):
        comm = np.array([[1., 0.], [0.5, 0.5], [0., 1.]])
        rgb = simple_csn.colors_from_committors(comm)
        assert isinstance(rgb, dict)
        assert len(rgb) == 3

    def test_rgb_values_in_range(self, simple_csn):
        comm = np.array([[1., 0.], [0.5, 0.5], [0., 1.]])
        rgb = simple_csn.colors_from_committors(comm)
        for node, colors in rgb.items():
            for channel in colors.values():
                assert 0 <= channel <= 255
