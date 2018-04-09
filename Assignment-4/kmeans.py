import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        randoms = np.random.choice(a=N, size=self.n_cluster, replace=False)
        means = np.zeros([self.n_cluster,D])
        J = -self.e
        for k in range(0,self.n_cluster):
            means[k] = x[randoms[k]]
        R = np.zeros(N)
        def get_membership(instance):
            return np.argmin(np.sum((means-instance)**2,axis=1))

        for number_of_updates in range(0,self.max_iter):
            #Update membership
            R = np.apply_along_axis(get_membership, 1, x)
            #Compute distortion measure
            J_new = 0
            for i in range(0,N):
                cluster = int(R[i])
                J_new += np.sum((means[cluster]-x[i])**2)
            J_new = J_new/N
            # print(np.absolute(J_new-J))
            if np.absolute(J_new-J)<=self.e:
                break
            J = J_new
            #Update means
            means_new = np.zeros([self.n_cluster,D])
            for i in range(0,N):
                cluster = int(R[i])
                means_new[cluster] += x[i]
            for k in range(0,self.n_cluster):
                count = sum(R==k)
                if count == 0:
                    means_new[k] = means[k]
                else:
                    means_new[k] = means_new[k]/count
            means = means_new
        return (means, R, number_of_updates)
        # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, i = k_means.fit(x)

        centroid_labels = np.zeros(self.n_cluster)
        for k in range(0,self.n_cluster):
            sample_index = [index for index, value in enumerate(membership) if value == k]
            if(sample_index!=[]):
                labels = np.take(y,sample_index);
                counts = np.bincount(labels)
                centroid_labels[k]=np.argmax(counts)
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        y = np.zeros(N)
        for i in range(0,N):
            k = np.argmin(np.sum((self.centroids-x[i])**2,axis=1))
            y[i] = self.centroid_labels[k]
        return y
        # DONOT CHANGE CODE BELOW THIS LINE
