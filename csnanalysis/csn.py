import scipy
import networkx as nx
import numpy as np
from csnanalysis.matrix import *
import itertools
from copy import deepcopy

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
        
            
    def trim(self, by_inflow=True, by_outflow=True, min_count=0):
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

        mask = np.ones(self.nnodes,dtype=bool)
        oldmask = np.zeros(self.nnodes,dtype=bool)

        if min_count > 0:
            mask[[i for i in range(self.nnodes) if totcounts[i] < min_count]] = False

        while (mask != oldmask).any():

            oldmask = mask.copy()
            self.trim_indices = [i for i in range(self.nnodes) if mask[i] == True]
            self.trim_graph = self.graph.subgraph(self.trim_indices)

            if by_outflow:
                downstream = nx.dfs_successors(self.trim_graph,msn).values()
                dlist = list(itertools.chain(*downstream)) + [msn]
                mask[[i for i in range(self.nnodes) if i not in dlist]] = False

            if by_inflow:
                upstream = list(nx.dfs_predecessors(self.trim_graph,msn).keys()) + [msn]
                mask[[i for i in range(self.nnodes) if i not in upstream]] = False

        # count all transitions to masked states and add these as self-transitions
        to_add = {}
        rows = self.countmat.row
        cols = self.countmat.col
        data = self.countmat.data
        
        for i in range(len(data)):
            if mask[rows[i]] is False and mask[cols[i]] is True:
                if cols[i] in to_add:
                    to_add[cols[i]] += data[i]
                else:
                    to_add[cols[i]] = data[i]

        tmp_arr = self.countmat.toarray()[mask,...][...,mask]

        for ind,full_ind in enumerate(self.trim_indices):
            if full_ind in to_add:
                tmp_arr[ind][ind] += to_add[full_ind]
            
        self.trim_countmat = scipy.sparse.coo_matrix(tmp_arr)
        if self.symmetrize:
            self.trim_countmat = symmetrize_matrix(self.trim_countmat)

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
        self.add_attr(label, fw_float)
            
        return full_wts

    def calc_committors(self,basins,labels=None,basin_labels=None,add_basins=False,tol=1e-6,maxstep=20):
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

        The committors are also returned from the function as a numpy array.
        """

        if self.trim_transmat is None:
            # use full transition matrix
            full_comm = committor(self.transmat,basins,tol=tol,maxstep=maxstep)
        else:
            # use trimmed transition matrix
            trim_basins = []
            for i,b in enumerate(basins):
                trim_basins.append([])
                for j,state in enumerate(b):
                    try:
                        trim_basins[i].append(self.trim_indices.index(state))
                    except:
                        pass
            comm = committor(self.trim_transmat,trim_basins,tol=tol,maxstep=maxstep)
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

