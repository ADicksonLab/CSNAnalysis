import scipy
import networkx as nx
import numpy as np
from csnanalysis.matrix import eig_weights, mult_weights

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
        
def symmetrize(countmat):
    """
    Symmetrizes a count matrix (in scipy sparse format).
    """
    return 0.5*(countmat + countmat.transpose())
    
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

        self.symmetrize = symmetrize
        if self.symmetrize:
            self.countmat = symmetrize(self.countmat)

        self.nnodes = self.countmat.shape[0]        
        self.transmat = count_to_trans(self.countmat)

        self.trim_transmat = None
            
        # initialize networkX directed graph
        self.graph = nx.DiGraph()
        labels = [{'ID' : i} for i in range(self.nnodes)]
        self.graph.add_nodes_from(zip(range(self.nnodes),labels))
        self.graph.add_weighted_edges_from(zip(self.transmat.col,self.transmat.row,self.transmat.data))

    def to_gephi(self, cols='all', node_name='node.csv', edge_name='edge.csv'):
        """
        Writes node and edge files for import into the Gephi network visualization program.
        """

    def add_attr(self, name, values):
        """
        Adds an attribute to the set of nodes in the CSN.
        """
        attr = {}
        for i, v in enumerate(values):
            attr[i] = v
            
        nx.set_node_attributes(self.graph,name,attr)
        
    def trim_graph(self, by_inflow=True, by_outflow=True, min_count=0):
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

        totcounts = self.countmat.sum(axis=1)
        msn = totcounts.argmax()

        to_cut = []
        mask = np.ones(self.nnodes,dtype=bool)
        if by_outflow:
            downstream = list(nx.dfs_predecessors(self.graph,msn).keys())
            mask[[i for i in range(self.nnodes) if i not in downstream]] = False

        if by_inflow:
            upstream = list(nx.dfs_successors(self.graph,msn).keys())
            mask[[i for i in range(self.nnodes) if i not in upstream]] = False

        if min_count > 0:
            mask[[i for i in range(self.nnodes) if totcounts[i] >= min_count]] = False

        self.trim_indices = [i for i in range(self.nnodes) if mask[i] is True]
        self.trim_graph = self.graph.subgraph(self.trim_indices)

        tmp_arr = self.countmat.toarray()[mask,...][...,mask]
        self.trim_countmat = scipy.sparse.coo_matrix(tmp_arr)
        if self.symmetrize:
            self.trim_countmat = symmetrize(self.trim_countmat)

        self.trim_nnodes = self.trim_countmat.shape[0]        
        self.trim_transmat = count_to_trans(self.trim_countmat)
                

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
            self.add_attr(label, full_wts)
        else:
            # use trimmed transition matrix
            wts = eig_weights(self.trim_transmat)
            full_wts = np.zeros(self.nnodes,dtype=float64)
            for i,ind in enumerate(self.trim_indices):
                full_wts[ind] = wts[i]
            self.add_attr(label, full_wts)

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
            self.add_attr(label, full_wts)
        else:
            # use trimmed transition matrix
            wts = mult_weights(self.trim_transmat,tol)
            full_wts = np.zeros(self.nnodes,dtype=float64)
            for i,ind in enumerate(self.trim_indices):
                full_wts[ind] = wts[i]
            self.add_attr(label, full_wts)

        return full_wts

