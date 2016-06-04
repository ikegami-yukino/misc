import numpy as np
from scipy.sparse import issparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted


class NegationNB(MultinomialNB):
    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")
        fc = safe_sparse_dot(Y.T, X)
        self.feature_count_ += fc.sum(axis=0) - fc
        self.class_count_ += Y.sum(axis=0)

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X
        1 / (1 - P(c)) * P(1 / P(w_i | ¬c))"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse='csr')
        return np.array(safe_sparse_dot(X, -self.feature_log_prob_.T) +
                        np.log(1 - np.exp(self.class_log_prior_)) * -1)
