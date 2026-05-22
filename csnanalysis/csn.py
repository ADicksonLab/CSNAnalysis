import itertools
from copy import deepcopy

import scipy
import networkx as nx
import numpy as np

from csnanalysis.matrix import (
    count_to_trans,
    symmetrize_matrix,
    eig_weights,
    mult_weights,
    committor,
    committor_linalg,
    get_eigenvectors,
    well_conditioned,
    fptd,
)

class CSN(object):

    def __init__(self, counts, symmetrize=False):
        """
        Initializes a CSN object using a counts matrix.  This can either be a numpy array,
        a scipy sparse matrix, or a list of lists. Indices: [to][from], (or, [row][column]).
        """
        if type(counts) is list:
            self.countmat = scipy.sparse.coo_matrix(counts)
        elif type(counts) is np.ndarray:
            self.countmat = scipy.sparse.coo_matrix(counts)
        elif type(counts) is scipy.sparse.coo.coo_matrix:
            self.countmat = counts
        else:
            try:
                self.countmat = counts.tocoo()
            except:
                raise TypeError("Count matrix is of unsupported type: ",type(counts))

        if self.countmat.shape[0] != self.countmat.shape[1]:
            raise ValueError("Count matrix is not square: ",self.countmat.shape)

        totcounts = self.countmat.sum(axis=1).tolist()

        self.symmetrize = symmetrize
        if self.symmetrize:
            self.countmat = symmetrize_matrix(self.countmat)

        self.nnodes = self.countmat.shape[0]
        self.transmat = count_to_trans(self.countmat)

        self.trim_transmat = None

        # initialize networkX directed graph
        self.graph = nx.DiGraph()
        labels = [{'label' : i, 'count' : int(totcounts[i][0])} for i in range(self.nnodes)]
        self.graph.add_nodes_from(zip(range(self.nnodes),labels))
        self.graph.add_weighted_edges_from(zip(self.transmat.col,self.transmat.row,100*self.transmat.data))

        # remove self edges from graph
        self_edges = [(i,i) for i in range(self.nnodes)]
        self.graph.remove_edges_from(self_edges)

    def to_gephi_csv(self, cols='all', node_name='node.csv', edge_name='edge.csv', directed=False):
        """
        Writes node and edge files for import into the Gephi network visualization program.

        cols  --  A list of columns that should be written to the node file.  ID and label are
                  included by default.  'all' will include every attribute attached to the
                  nodes in self.graph.

        """
        if cols == 'all':
            cols = list(self.graph.node[0].keys())
        else:
            if 'label' not in cols:
                cols = ['label'] + cols
            if 'ID' not in cols:
                cols = ['ID'] + cols

        with open(node_name,mode='w') as f:
            f.write(" ".join(cols)+"\n")
            for i in range(self.nnodes):
                data = [str(self.graph.node[i][c]) for c in cols]
                f.write(' '.join(data)+"\n")

        # compute edge weights
        if directed:
            with open(edge_name,mode='w') as f:
                f.write("source target type prob i_weight\n")
                for (from_ind, to_ind, weight_dict) in self.graph.edges.data():
                    wt = weight_dict['weight']
                    f.write("{0:d} {1:d} {2:s} {3:f} {4:d}\n".format(from_ind,to_ind,'Directed',wt,int(wt*100)))
        else:
            with open(edge_name,mode='w') as f:
                f.write("source target type prob i_weight\n")
                for (from_ind, to_ind, weight_dict) in self.graph.edges.data():
                    if from_ind <= to_ind:
                        if self.graph.has_edge(to_ind,from_ind):
                            back_wt = self.graph.edges[to_ind,from_ind]['weight']
                        else:
                            back_wt = 0
                        edge_weight = 0.5*(back_wt + weight_dict['weight'])
                        f.write("{0:d} {1:d} {2:s} {3:f} {4:d}\n".format(from_ind,to_ind,'Undirected',edge_weight,int(edge_weight*100)))

    def add_attr(self, name, values):
        """
        Adds an attribute to the set of nodes in the CSN.
        """
        attr = {}
        for i, v in enumerate(values):
            attr[i] = v

        nx.set_node_attributes(self.graph,values=attr,name=name)

    def add_trim_attr(self, name, values, default=0):
        """
        Adds an attribute to the set of nodes in the CSN.
        Values should be an iterable of the size of csn.trim_indices
        """
        attr = {}
        for i in range(self.nnodes):
            if i in self.trim_indices:
                trim_idx = self.trim_indices.index(i)
                attr[i] = values[trim_idx]
            else:
                attr[i] = default

        nx.set_node_attributes(self.graph,values=attr,name=name)
        
    def set_colors(self, rgb):
        """
        Adds colors to each node for gexf export of the graph.

        rgb: A dict that stores the rgb values of each node.

        Example: rgb['0']['r'] = 255
                 rgb['0']['g'] = 0
                 rgb['0']['b'] = 0
        """
        for node in rgb:
            if 'viz' not in self.graph.node[node]:
                self.graph.node[node]['viz'] = {}
            self.graph.node[node]['viz']['color'] = {'r': rgb[node]['r'], 'g': rgb[node]['g'], 'b': rgb[node]['b'], 'a': 0}

    def set_positions(self, xy):
        """
        Adds x,y positions to each node for gexf export of the graph.

        xy: A dict that stores the xy positions of each node.

        Example: xy[0]['x'] = 0.5
                 xy[0]['y'] = 1.6
        """
        for node in xy:
            if 'viz' not in self.graph.node[node]:
                self.graph.node[node]['viz'] = {}
            self.graph.node[node]['viz']['position'] = {'x': float(xy[node]['x']), 'y': float(xy[node]['y']), 'z': float(0)}


    def colors_from_committors(self,comm):
        """
        Returns rgb dict using values of committor probabilities.
        Very useful for 3-basin committors!

        comm:  Numpy array of committors, as returns from self.calc_committors
        """
        highc = 255
        nbasin = comm.shape[1]
        rgb = {}
        colors = ['r','g','b']
        for node in range(self.nnodes):
            maxc = comm[node,:].max()
            for i in range(min(3,nbasin)):
                if node not in rgb:
                    rgb[node] = {}
                if maxc == 0:
                    rgb[node][colors[i]] = 0
                else:
                    rgb[node][colors[i]] = int(highc*comm[node,i]/maxc)

        return rgb


    def trim(self, by_inflow=True, by_outflow=True, min_count=None):
        """
        Trims a graph to delete nodes that are not connected to the main
        component, which is the component containing the most-sampled node (MSN)
        by counts.

        by_inflow: whether to delete nodes that are not connected to the MSN by inflow

        by_outflow: whether to delete nodes that are not connected to the MSN by outflow

        min_count: nodes that do not have a count > min_count will be deleted

        Trimmed graph is saved as self.trim_graph. The trimmed transition matrix
        is saved as self.trim_transmat, and the count matrix is saved as
        self.trim_countmat.

        The mapping from the nodes in the trimmed set to the full set is given by
        self.trim_indices.
        """

        totcounts = self.countmat.toarray().sum(axis=0)
        msn = totcounts.argmax()

        mask = np.ones(self.nnodes,dtype=bool)
        oldmask = np.zeros(self.nnodes,dtype=bool)

        if min_count is not None:
            mask[[i for i in range(self.nnodes) if totcounts[i] < min_count]] = False
        else:
            mask[[i for i in range(self.nnodes) if totcounts[i] == 0]] = False

        itercount = 0
        diff = []
        while (mask != oldmask).any():

            oldmask = mask.copy()
            self.trim_indices = [i for i in range(self.nnodes) if mask[i] == True]
            self.trim_graph = self.graph.subgraph(self.trim_indices)

            print(f"Iteration {itercount}:",diff)
            itercount += 1
            
            if by_outflow:
                downstream = [i for i in self.trim_indices if nx.has_path(self.trim_graph,msn,i)]
                mask[[i for i in range(self.nnodes) if i not in downstream]] = False

            if by_inflow:
                upstream = [i for i in self.trim_indices if nx.has_path(self.trim_graph,i,msn)]
                mask[[i for i in range(self.nnodes) if i not in upstream]] = False

            diff = [i for i in range(self.nnodes) if mask[i] != oldmask[i]]

        # count all transitions to masked states and add these as self-transitions
        # rows = to, cols = from
        to_add = {}
        rows = self.countmat.row
        cols = self.countmat.col
        data = self.countmat.data

        for i in range(len(data)):
            if mask[rows[i]] == False and mask[cols[i]] == True:
                if cols[i] in to_add:
                    to_add[cols[i]] += data[i]
                else:
                    to_add[cols[i]] = data[i]

        tmp_arr = self.countmat.toarray()[mask,...][...,mask]

        for ind,full_ind in enumerate(self.trim_indices):
            if full_ind in to_add:
                tmp_arr[ind][ind] += to_add[full_ind]

        assert tmp_arr.sum(axis=0).min() > 0, 'Error! A state in the trimmed countmat has no transitions'
        self.trim_countmat = scipy.sparse.coo_matrix(tmp_arr)

        if self.symmetrize:
            self.trim_countmat = symmetrize_matrix(self.trim_countmat)

        self.trim_nnodes = self.trim_countmat.shape[0]
        self.trim_transmat = count_to_trans(self.trim_countmat)

        is_trim = np.zeros((self.nnodes))
        for i in range(self.nnodes):
            if i not in self.trim_indices:
                is_trim[i] = 1
        self.add_attr('trim',is_trim)

        if not well_conditioned(self.trim_transmat.toarray()):
            print("Warning: trimmed transition matrix is not well-conditioned.")
                
    def calc_eig_weights(self,label='eig_weights'):
        """
        Calculates weights of states using the highest Eigenvalue of the
        transition matrix.  By default it uses self.trim_transmat, but will
        use self.transmat if no trimming has been done.

        The weights are stored as node attributes in self.graph with the label
        'label', and are also returned from the function.
        """

        if self.trim_transmat is None:
            # use full transition matrix
            full_wts = eig_weights(self.transmat)
        else:
            # use trimmed transition matrix
            wts = eig_weights(self.trim_transmat)
            full_wts = np.zeros(self.nnodes,dtype=float)
            for i,ind in enumerate(self.trim_indices):
                full_wts[ind] = wts[i]

        fw_float = [float(i) for i in full_wts]
        self.add_attr(label, fw_float)

        return full_wts

    def calc_mult_weights(self,label='mult_weights',tol=1e-6):
        """
        Calculates weights of states using iterative multiplication of the
        transition matrix.  By default it uses self.trim_transmat, but will
        use self.transmat if no trimming has been done.

        The weights are stored as node attributes in self.graph with the label
        'label', and are also returned from the function.
        """

        if self.trim_transmat is None:
            # use full transition matrix
            full_wts = mult_weights(self.transmat,tol)
        else:
            # use trimmed transition matrix
            wts = mult_weights(self.trim_transmat,tol)
            full_wts = np.zeros(self.nnodes,dtype=float)
            for i,ind in enumerate(self.trim_indices):
                full_wts[ind] = wts[i]

        fw_float = [float(i) for i in full_wts]
        if label is not None:
            self.add_attr(label, fw_float)

        return full_wts

    def calc_committors(self, basins,
                        labels=None,
                        basin_labels=None,
                        add_basins=False,
                        tol=1e-6,
                        maxstep=20,
                        method='iter'):
        """
        Calculates committor probabilities between an arbitrary set of N basins.

        basins     -- A list of lists, describing which states make up the
                      basins of attraction.  There can be any number of basins.
                      e.g. [[basin1_a,basin1_b,...],[basin2_a,basin2_b,...]]
        labels     -- A list of labels given to the committors (one for each
                      basin) in the attribute list.
        add_basins -- Whether to add basin vectors to attribute list.
        basin_labels -- List of names of the basins.
        tol        -- Tolerance of iterative multiplication process
                      (see matrix.trans_mult_iter)
        maxstep    -- Maximum number of iteractions of multiplication process.
        method     -- 'iter' for iterative multiplication, 'linalg' for 
                      linear algebra solve (two-basin only)

        The committors are also returned from the function as a numpy array.
        """

        assert method in ['iter','linalg'], 'Error! method must be either iter or linalg'

        if self.trim_transmat is None:
            assert well_conditioned(self.transmat.toarray()), "Error: cannot calculate committors from transition matrix. Try trimming first."

            # use full transition matrix
            if method == 'iter':
                full_comm = committor(self.transmat,basins,tol=tol,maxstep=maxstep)
            elif method == 'linalg':
                full_comm = committor_linalg(self.transmat,basins)
                    
        else:
            # use trimmed transition matrix
            trim_basins = []
            for i,b in enumerate(basins):
                trim_basins.append([])
                for state in b:
                    if state in self.trim_indices:
                        trim_basins[i].append(self.trim_indices.index(state))

            if method == 'iter':
                comm = committor(self.trim_transmat,trim_basins,tol=tol,maxstep=maxstep)
            elif method == 'linalg':
                comm = committor_linalg(self.trim_transmat,trim_basins)

            full_comm = np.zeros((self.transmat.shape[0],len(basins)),dtype=float)
            for i,ind in enumerate(self.trim_indices):
                full_comm[ind] = comm[i]

        if labels is None:
            labels = ['p' + str(i) for i in range(len(basins))]

        for i in range(len(basins)):
            fc_float = [float(i) for i in full_comm[:,i]]
            self.add_attr(labels[i], fc_float)

        if add_basins:
            if basin_labels is None:
                basin_labels = [str(i) for i in range(len(basins))]
            for i,b in enumerate(basins):
                bvec = np.zeros(self.nnodes,dtype=int)
                bvec[b] = 1
                bv_int = [int(i) for i in bvec]
                self.add_attr(basin_labels[i],bv_int)

        return full_comm

    def calc_mfpt(self,sinks,maxsteps=None,tol=1e-3,sources=None):
        """
        Calculates the mean first passage time (MFPT) and the first passage time distribution (FPTD)
        from every state in the matrix to a set of "sinks".

        sinks -- (list of int) A list of states that will be used as sinks

        stepsize -- (int) The lagtime, in multiples of tau, that is used to compute the MFPT, which is 
                          also the resolution of the FPTD.

        maxsteps -- (int) The maximum number of steps used to compute the FPTD.

        tol -- (float) The quitting criteria for FPTD calculation.  The calculation will stop if the 
                       largest "un-sunk" probability is below tol.

        sources -- (None or list of int) List of source states to average over.  If None, will return
                   MFPT and FPTD of all states.
        """

        assert tol is not None or maxsteps is not None, "Error: either maxsteps or tol must be defined!"

        if tol is None:
            tol = 0.0
        if maxsteps is None:
            maxsteps = np.inf
        
        if self.trim_transmat is None:
            assert well_conditioned(self.transmat.toarray()), "Error: cannot calculate mfpt from transition matrix. Try trimming first."

            # use full transition matrix
            full_fptd = fptd(self.transmat,sinks,maxsteps=maxsteps,tol=tol)
            full_mfpt = np.zeros((self.transmat.shape[0]),dtype=float)
            for i in range(full_fptd.shape[0]):
                # loop over exponentially placed timepoints
                # this entry is the flux between lag*(2**[i]) and lag*(2**[i+1])
                # avg. of endpoints is (2**(i-1) + 2**(i))
                full_mfpt += full_fptd[i,:]*(2**(i-1) + 2**(i)) # in units of lagtime

        else:
            # use trimmed transition matrix
            trim_sinks = []
            for state in sinks:
                if state in self.trim_indices:
                    trim_sinks.append(self.trim_indices.index(state))

            trim_fptd = fptd(self.trim_transmat,trim_sinks,maxsteps=maxsteps,tol=tol)
            trim_mfpt = np.zeros((trim_fptd.shape[1]),dtype=float)
            for i in range(trim_fptd.shape[0]):
                # loop over exponentially placed timepoints
                # this entry is the flux between lag*(2**[i]) and lag*(2**[i+1])
                trim_mfpt += trim_fptd[i,:]*(2**(i-1) + 2**(i)) # in units of lagtime

            full_fptd = np.zeros((trim_fptd.shape[0],self.transmat.shape[0]),dtype=float)
            full_mfpt = np.zeros((self.transmat.shape[0]),dtype=float)
            for i,ind in enumerate(self.trim_indices):
                full_fptd[:,ind] = trim_fptd[:,i]
                full_mfpt[ind] = trim_mfpt[i]

        if sources is not None:
            wts = self.calc_mult_weights(label=None,tol=1e-6)
            wt_sum = wts[sources].sum()

            avg_mfpt = 0
            avg_fptd = np.zeros((full_fptd.shape[0]))
            for s in sources:
                avg_mfpt += full_mfpt[s]*wts[s]
                avg_fptd += full_fptd[:,s]*wts[s]

            return np.array([avg_mfpt/wt_sum]), np.array([avg_fptd/wt_sum])
        else:
            return full_mfpt, full_fptd

        

    def idxs_to_trim(self,idxs):
        """
        Converts a list of idxs to trim_idxs.

        idxs -- List of states in the transition matrix. Elements should be
                integers from 0 to nstates.
        """

        return [self.trim_indices.index(i) for i in idxs if i in self.trim_indices]

    def calc_eigvectors(self, n_eig=3,
                        include_wt_vec=False,
                        save_to_graph=True,
                        save_imag_to_graph=False,
                        save_label='eig'):
        """
        Calculates committor probabilities between an arbitrary set of N basins.

        n_eig    -- The number of eigenvectors to return

        include_wt_vec -- Whether or not to include the eigenvector with 
                          eigenvalue = 1.  Note that this is equal to the 
                          steady state weights.

        save_to_graph -- Whether or not to save the eigenvectors to the graph
                         (real part).

        save_imag_to_graph -- Whether or not to save the eigenvectors to the 
                              graph (imaginary part).

        save_label -- Labels given to each eigenvector when saving to the graph.
                      Indices are appended and counting starts at zero (e.g. 
                      eig0, eig1, ..).  If imaginary part is saved (eig0_imag, eig1_imag, ...)

        Output:
        eig_vecs -- A numpy array (N, n_eig) of eigenvector elements (real part only)

        eig_vals -- A numpy array of the n_eig eigenvalues (real part only)
        
        eig_vecs_imag -- A numpy array (N, n_eig) of eigenvector elements (imaginary part only)
        
        eig_vals_imag - A numpy array of the n_eig eigenvalues (imaginary part only)

        """

        if self.trim_transmat is None:
            # use full transition matrix
            vec_real, val_real, vec_imag, val_imag = get_eigenvectors(self.transmat.toarray(), n_eig=n_eig, return_wt_vec=include_wt_vec)
        else:
            # use trimmed transition matrix
            trim_vec_real, val_real, trim_vec_imag, val_imag = get_eigenvectors(self.trim_transmat.toarray(), n_eig=n_eig, return_wt_vec=include_wt_vec)

            vec_real = np.zeros((self.transmat.shape[0],n_eig),dtype=float)
            vec_imag = np.zeros((self.transmat.shape[0],n_eig),dtype=float)
            for i,ind in enumerate(self.trim_indices):
                vec_real[ind] = trim_vec_real[i]
                vec_imag[ind] = trim_vec_imag[i]

        # add eigenvectors as attributes
        if save_to_graph:
            for idx in range(n_eig):
                label = f'{save_label}{idx}'
                self.add_attr(label, vec_real[:,idx])

        if save_imag_to_graph:
            for idx in range(n_eig):
                label = f'{save_label}{idx}_imag'
                self.add_attr(label, vec_imag[:,idx])

        return vec_real, val_real, vec_imag, val_imag
