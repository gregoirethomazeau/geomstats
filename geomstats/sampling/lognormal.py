import geomstats.geometry.spd_matrices as spd

class LogNormal:
    """ LogNormal Sampler (Currently only done for SPDmanifold,EuclideanSpace)

    Parameters:
    ----------
    manifold: Name of Manifold
        

    """
    def __init__(self,manifold,mean,cov=None):
        self.manifold = manifold
        self.mean = mean
        self.n  = self.mean.shape[-1]
        self.cov_n = (self.n*(self.n+1))//2
        self.cov = cov
        if self.cov is None:
            self.cov = gs.eye(self.cov_n)
        if self.manifold == 'SPDmanifold':
            mean_SPDmanifold = spd.SPDMatrices(self.n)
            cov_SPDmanifold  = spd.SPDMatrices(self.cov_n)

            if (not mean_SPDmanifold.belongs(self.mean)):
                 raise ValueError("Invalid Value in mean, Not Symmetric Positive Definite" , mean)

    
    def _sample_spd(self,samples):
        SPDmanifold = spd.SPDMatrices(self.n)
        i,  = gs.diag_indices(self.n)
        j,k = gs.triu_indices(self.n,k=1)
        sym_matrix = SPDmanifold.logm(self.mean)

        mean_euclidean = gs.hstack((sym_matrix[i,i],gs.sqrt(2)*sym_matrix[j,k]))
        samples_spd = gs.zeros((samples,n,n))
        samples_euclidean = gs.random.multivariate_normal(mean_euclidean, cov, samples)
        samples_spd[:,i,i] = samples_euclidean[:,:n]
        samples_spd[:,j,k] = samples_euclidean[:,n:]/gs.sqrt(2)
        samples_spd = SPDmanifold.expm(samples_spd)
        return samples_spd

    def _sample_euclidean(self,samples):
        pass    

    def sample(self,samples=1):

        if self.manifold == 'Euclidean':
            return self._sample_euclidean(samples)

        if self.manifold == 'SPDmanifold':
            return self._sample_spd(samples)    
