import numpy as np
import sklearn.neighbors as neighbors
from scipy import stats


class KNNBasedDefense:
    """kNN-based Defense.

    Implementing kNN-based Defense by Paudice et. al. (2019).
    A kNN-based sanitization method to relabel the training data.

    Reference: http://link.springer.com/10.1007/s13042-016-0629-5

    Parameters
    ----------
    k : int, default=5
        k nearest neighbors.

    eta : float, default=0.5
        The confidence threshold. Range: [0.5, 1.0].
    """

    def __init__(self, k=5, eta=0.5) -> None:
        if k < 1:
            raise ValueError(f'k must be larger than 1. Received {k}.')
        if eta < 0.5 or eta > 1.:
            raise ValueError(f'Eta must be within [0.5, 1.0]. Received {eta}')

        self.k = k
        self.eta = eta

    def run(self, X_train, y_train):
        """Run the sanitization method to relabel y_train.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The training samples.

        y_train : array-like of shape (n_samples,)
            The corresponding training labels.

        Returns
        -------
        X_train : array-like of shape (n_samples, n_features)
            The training samples. Unchanged as the input, X_train.

        y_prime : array-like of shape (n_samples,)
            The corresponding sanitized training labels.
        """
        assert len(X_train) == len(y_train), \
            'X_train and y_train should have same length.'

        n = X_train.shape[0]
        tree = neighbors.KDTree(X_train)
        _, indices = tree.query(X_train, k=self.k + 1)
        # The 1st index is X itself, so exclude it.
        indices = indices[:, 1:]
        assert indices.shape == (n, self.k)

        y_prime = np.copy(y_train)
        for i in range(n):
            y_neighbors = y_train[indices[i]]
            mode_results = stats.mode(y_neighbors)
            y_mode = mode_results.mode[0]
            y_conf = mode_results.count[0] / self.k
            if y_conf >= self.eta:
                y_prime[i] = y_mode

        return X_train, y_prime

    def eval(self, y_original, y_sanitized):
        """Evaluate the results between original and sanitized labels.

        Parameters
        ----------
        y_original : array-like of shape (n_samples,)
            The received training labels.


        y_sanitized : array-like of shape (n_samples,)
            The training labels after sanitization.

        Returns
        -------
        ratio : float
            The percentage of unmatched labels.
        """
        assert y_original.shape == y_sanitized.shape, \
            'Two inputs must be in the same shape.'

        return np.mean(y_original == y_sanitized)


if __name__ == '__main__':
    """Demo"""
    rng = np.random.RandomState(0)
    knn_defense = KNNBasedDefense()
    n = 20
    X = rng.random_sample((n, 4))
    y = rng.randint(0, 2, size=n)
    _, y_prime = knn_defense.run(X, y)

    print('Original:\n', y)
    print('Sanitized:\n', y_prime)

    ratio = knn_defense.eval(y, y_prime)
    print('Difference ratio:', ratio)
