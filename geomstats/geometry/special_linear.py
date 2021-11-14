"""Class for the group of special linear matrices."""

import geomstats.backend as gs
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.lie_group import MatrixLieGroup
from geomstats.geometry.lie_algebra import MatrixLieAlgebra


class SpecialLinear(MatrixLieGroup):
    """Class for the Special Linear group SL(n).

    This is the space of invertible matrices of size n and unit determinant.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """
    
    def __init__(self, n):
        super(SpecialLinear, self).__init__(
            dim=int((n * (n - 1)) / 2),
            n=n,
            lie_algebra=SpecialLinearLieAlgebra(n=n),
        )

        self.metric = InvariantMetric(self)

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the group.

        Check the size and the value of the determinant.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        ndim = point.ndim
        if ndim == 1:
            return False
        mat_dim1 , mat_dim2 = point.shape[-2:]
        if mat_dim1 != self.n or mat_dim2 != self.n:
            return False
        return gs.abs(gs.linalg.det(point) - 1) < atol

    def projection(self, point):
        """Project a point in embedding space to the group.

        This can be done by scaling the entire matrix by its determinant to
        the power 1/n.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., n, n]
            Projected point.
        """
        def projects(pt):
            if pt.ndim == 2:
                determinant = gs.linalg.det(pt)
                point_flipped = utils.flip_determinant(pt,determinant)
                return  ((1./gs.abs(determinant)) ** (1/self.n)) * point_flipped
            else:
                return gs.stack([projects(p) for p in pt])
        return projects(point)

    def random_point(self, n_samples=1, bound=1.0, n_iter=100):
        """Sample in the group.

        One may use a sample from the general linear group and project it
        down to the special linear group.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1.0
        n_iter : int
            Maximum number of trials to sample a matrix with non zero det.
            Optional, default: 100.

        Returns
        -------
        point : array-like, shape=[..., dim]
           Sample.
        """
        n = self.n
        sample = []
        n_accepted, iteration = 0, 0
        while n_accepted < n_samples and iteration < n_iter:
            raw_samples = gs.random.normal(size=(n_samples - n_accepted, n, n))
            dets = gs.linalg.det(raw_samples)
            criterion = gs.abs(dets) > gs.atol
            if gs.any(criterion):
                sample.append(raw_samples[criterion])
                n_accepted += gs.sum(criterion)
            iteration += 1
        if n_samples == 1:
            return self.projection(sample[0][0])
        return self.projection(gs.concatenate(sample))


class SpecialLinearLieAlgebra(MatrixLieAlgebra):
    """Class for the Lie algebra sl(n) of the Special Linear group.

    This is the space of matrices of size n with vanishing trace.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n):
        super(SpecialLinearLieAlgebra, self).__init__(
            dim=n**2 - 1, # I changed the dimension which was not correct
            n=n,
        )

    def basis_representation(self, matrix_representation):
        """Compute the coefficients of matrices in the given basis.

        Assume the basis is the one described in this answer on StackOverflow:
        https://math.stackexchange.com/a/1949395/920836

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        basis_representation : array-like, shape=[..., dim]
            Coefficients in the basis.
        """
        def computes_basis_representation(matrix,n):
            if matrix.ndim == 2:
                matrix_representation = gs.ones(n**2 - 1)
                for i in range(n):
                    for j in range(n):
                        if i != 0 or j != 0: matrix_representation[i*n+j-1] = matrix[i,j]
                return matrix_representation
            else:
                return gs.stack([computes_basis_representation(mt,n) for mt in matrix])

        return computes_basis_representation(matrix_representation,self.n)


    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the Lie algebra.

        This method checks the shape of the input point and its trace.

        Parameters
        ----------
        point : array-like, shape=[.., n, n]
            Point to test.
        atol : float
            Tolerance threshold for zero values.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        ndim = point.ndim
        if ndim == 1:
            return False
        mat_dim_1, mat_dim_2 = point.shape[-2:]
        has_right_dimension = (mat_dim_1 == self.n) and (mat_dim_2 == self.n)
        if has_right_dimension: return (gs.abs(gs.trace(point,axis1=-2,axis2= -1)) < atol)
        else: return gs.zeros(point.shape[:-2],dtype=bool)


    def projection(self, point):
        """Project a point to the Lie algebra.

        This can be done by removing the trace in the first entry of the matrix.

        Parameters
        ----------
        point: array-like, shape=[..., n, n]
            Point.

        Returns
        -------
        point: array-like, shape=[..., n, n]
            Projected point.
        """
        def projects(pt):
            if pt.ndim == 2:
                e00 = gs.zeros((self.n,self.n))
                e00[0,0] = 1.
                return pt - gs.trace(pt) * e00
            else:
                return gs.stack([projects(p) for p in pt])
        return projects(point)
