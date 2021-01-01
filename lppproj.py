import torch 


from knn_cuda import KNN
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LocalityPreservingProjection(BaseEstimator, TransformerMixin):
    """Locality Preserving Projection
    Parameters
    ----------
    n_components : integer
        number of coordinates for the manifold
    n_neighbors : integer
        number of neighbors to consider for each point.
    weight : string ['adjacency'|'heat']
        Weight function to use for the mapping
    weight_width : float
        Width of the heat kernel for building the weight matrix.
        Only referenced if weights == 'heat'
    neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
        Algorithm to use for nearest neighbors search,
        passed to neighbors.NearestNeighbors instance.
    Attributes
    ----------
    projection_ : array-like, shape (n_features, n_components)
        Linear projection matrix for the embedding
    """
    def __init__(self, n_components=2, n_neighbors=5,
                 weight='adjacency', weight_width=1.0,
                 neighbors_algorithm='auto'):
        # TODO: allow radius neighbors
        # TODO: allow for precomputed weights
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.weight = weight
        self.weight_width = weight_width
        self.neighbors_algorithm = neighbors_algorithm

    def fit(self, X, y=None):
        #X = check_array(X)
        W = self._compute_weights(X)
        self.projection_ = self._compute_projection(X, W)
        return self

    def transform(self, X):
        #X = check_array(X)
        return torch.mm(X, self.projection_)

    def _compute_projection(self, X, W):
        """Compute the LPP projection matrix
        Parameters
        ----------
        X : array_like, (n_samples, n_features)
            The input data
        W : array_like or sparse matrix, (n_samples, n_samples)
            The precomputed adjacency matrix
        Returns
        -------
        P : ndarray, (n_features, self.n_components)
            The matrix encoding the locality preserving projection
        """
        # TODO: check W input; handle sparse case
        #X = check_array(X)

        D = torch.diag(W.sum(1))
        L = D - W
        #print(L.size())
        #print(X.size())
        evals, evecs = eigh_robust(torch.mm(X.t(), torch.mm(L, X)),
                                   torch.mm(X.t(), torch.mm(D, X)),
                                   eigvals=(0, self.n_components - 1))
        return evecs

    def _compute_weights(self, X):
        #X = check_array(X)
        #self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors,
        #                              algorithm=self.neighbors_algorithm)
        #self.nbrs_.fit(X)
        points_number,*para = X.size()
        knn = KNN(k=self.n_neighbors, transpose_mode=True) #points,neighbours..
        dist,col = knn(X.unsqueeze(0),X.unsqueeze(0))
        #knn_pc = torch.cat([X[0,col[0,:,i],:].unsqueeze(-1) for i in range(self.n_neighbours)],dim=-1).permute(0,2,1)
        W = torch.zeros([points_number,points_number],dtype=X.dtype, device=X.device)
        for i in range(5):
            W[:,col[0,:,i]]=1

        if self.weight == 'adjacency':
            W = W #kneighbors_graph(self.nbrs_, self.n_neighbors,
            #                     mode='connectivity', include_self=True)
        elif self.weight == 'heat':
            W = W #kneighbors_graph(self.nbrs_, self.n_neighbors,
            #                     mode='distance', include_self=True)
            W = torch.exp(-W ** 2 / self.weight_width ** 2)
        else:
            raise ValueError("Unrecognized Weight")

        # symmetrize the matrix
        # TODO: make this more efficient & keep sparse output
        #W = W.toarray()
        p = torch.cat([W.unsqueeze(-1),W.t().unsqueeze(-1)],dim=-1)
        W = torch.max(p, dim=-1,keepdim=False)[0]
        #print(W.size())
        return W


def eigh_robust(a, b=None, eigvals=None, eigvals_only=False,
                overwrite_a=False, overwrite_b=False,
                turbo=True, check_finite=True):
    """Robustly solve the Hermitian generalized eigenvalue problem
    This function robustly solves the Hermetian generalized eigenvalue problem
    ``A v = lambda B v`` in the case that B is not strictly positive definite.
    When B is strictly positive-definite, the result is equivalent to
    scipy.linalg.eigh() within floating-point accuracy.
    Parameters
    ----------
    a : (M, M) array_like
        A complex Hermitian or real symmetric matrix whose eigenvalues and
        eigenvectors will be computed.
    b : (M, M) array_like, optional
        A complex Hermitian or real symmetric matrix.
        If omitted, identity matrix is assumed.
    eigvals : tuple (lo, hi), optional
        Indexes of the smallest and largest (in ascending order) eigenvalues
        and corresponding eigenvectors to be returned: 0 <= lo <= hi <= M-1.
        If omitted, all eigenvalues and eigenvectors are returned.
    eigvals_only : bool, optional
        Whether to calculate only eigenvalues and no eigenvectors.
        (Default: both are calculated)
    turbo : bool, optional
        Use divide and conquer algorithm (faster but expensive in memory,
        only for generalized eigenvalue problem and if eigvals=None)
    overwrite_a : bool, optional
        Whether to overwrite data in `a` (may improve performance)
    overwrite_b : bool, optional
        Whether to overwrite data in `b` (may improve performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    Returns
    -------
    w : (N,) float ndarray
        The N (1<=N<=M) selected eigenvalues, in ascending order, each
        repeated according to its multiplicity.
    v : (M, N) complex ndarray
        (if eigvals_only == False)
    """
    kwargs = dict(eigvals=eigvals, eigvals_only=eigvals_only,
                  turbo=turbo, check_finite=check_finite,
                  overwrite_a=overwrite_a, overwrite_b=overwrite_b)
    #print("testtest1")
    # Check for easy case first:
    if b is None:
        return torch.symeig(a)

    # Compute eigendecomposition of b
    kwargs_b = dict(turbo=turbo, check_finite=check_finite,
                    overwrite_a=overwrite_b)  # b is a for this operation
    #print(b.size())
    #print("testtest2")
    S, U = torch.symeig(b,True)
    #print(S.size())
    #print(U.size())
    S=S.unsqueeze(-1)
    S=S[0,:]
    U=U[:,0].unsqueeze(-1)
    #print(U.size())
    #print(S.size())
    #print("testtest3")
    # Combine a and b on left hand side via decomposition of b
    #print(S.size())
    S[S <= 0] = np.inf
    Sinv = 1/ torch.sqrt(S)
    #print(Sinv.size())
    #print(U.size())
    #print(a.size())
    #print(Sinv.size())
    #print(U.size())
    #Sinv=Sinv.unsqueeze(0)

    #print(torch.mm(a,U).size())
    W = Sinv*torch.mm(U.t(), torch.mm(a, U))*Sinv
    output = torch.symeig(W,True)
    
    if eigvals_only:
        return output
    else:
        evals, evecs = output
        return evals, torch.mm(U, Sinv * evecs)
