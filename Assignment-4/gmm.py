import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
            means, membership, i = k_means.fit(x)
            N_k = np.bincount(membership)
            pi_k = N_k/N
            variances = np.zeros([self.n_cluster,D,D])
            for k in range(0,self.n_cluster):
                sample_index = [index for index, value in enumerate(membership) if value == k] 
                x_in_cluster_k = x[sample_index]
                diff = x_in_cluster_k - means[k]
                variances[k] = np.dot(diff.T,diff)/N_k[k]
            self.means = means
            self.variances = variances
            self.pi_k = pi_k
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            variances = np.zeros([self.n_cluster,D,D])
            for k in range(0,self.n_cluster):
                variances[k] = np.identity(D)
            self.means = np.random.rand(self.n_cluster,D)
            self.variances = variances
            self.pi_k = np.asarray([1/self.n_cluster]*self.n_cluster)
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        # Compute the log-likelihood
        log_likelihood_current = self.compute_log_likelihood(x)
        for update_iteration in range(0,self.max_iter):
            # E Step
            gamma_ik = np.zeros([N,self.n_cluster])
            for i in range(0,N):
                p_x = self.calculate_p_x(x[i])
                for k in range(0,self.n_cluster):
                    gamma_ik[i][k] = self.pi_k[k]*self.get_guassian_pro(x[i],self.means[k],self.variances[k])/p_x
            # M Step
            N_k = np.sum(gamma_ik,axis=0)
            self.means = np.divide((np.dot(gamma_ik.T,x)).T,N_k).T
            variances = np.zeros([self.n_cluster,D,D])
            for k in range(0,self.n_cluster):
                gamma_k = gamma_ik[:,k]
                diff = x - self.means[k]
                variances[k] = np.dot(gamma_k*diff.T,diff)/N_k[k]
            self.variances = variances
            self.pi_k = N_k/N
            # Compute the log-likelihood
            log_likelihood_new = self.compute_log_likelihood(x)
            if np.absolute(log_likelihood_current-log_likelihood_new)<self.e:
                break
            log_likelihood_current = log_likelihood_new
        return update_iteration
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        D = len(self.variances[0])
        samples = np.zeros([N,D])
        clusters = np.random.choice(self.n_cluster, size=N, p=self.pi_k)
        for i in range(0,N):
            k = clusters[i]
            samples[i] = np.random.multivariate_normal(self.means[k], self.variances[k])
        return samples
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        log_likelihood = 0.0
        for i in range(0,len(x)):
            p_x = self.calculate_p_x(x[i])
            log_likelihood += np.log(p_x)
        log_likelihood = float(log_likelihood)
        return log_likelihood
    def get_guassian_pro(self,xx,mu,sigma):
        inv = False
        while not inv:
            try:
                sigma_inv = np.linalg.inv(sigma)
                inv = True
            except np.linalg.LinAlgError:
                sigma += np.identity(len(xx))*0.001
        temp = np.dot(np.dot(xx-mu,sigma_inv),(xx-mu).T)
        return np.exp(-0.5*temp)/(np.sqrt(np.power(2*np.pi,len(xx))*np.linalg.det(sigma)))
    def calculate_p_x(self,xx):
        probabilities = np.zeros(self.n_cluster)
        for k in range(0,self.n_cluster):
                probabilities[k] = self.get_guassian_pro(xx,self.means[k],self.variances[k])
        return np.dot(self.pi_k,probabilities)
        # DONOT MODIFY CODE BELOW THIS LINE
