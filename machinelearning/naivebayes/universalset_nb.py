import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted


class UniversalSetNB(MultinomialNB):
    def _update_feature_log_prob(self):
        smoothed_fc = self.feature_count_ + self.alpha
        negation_fc = smoothed_fc.sum(axis=0) - smoothed_fc
        smoothed_cc = negation_fc.sum(axis=1)

        numerator = np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1))
        denominator = np.log(negation_fc) - np.log(smoothed_cc.reshape(-1, 1))
        self.feature_log_prob_ = numerator - denominator

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X
        P(c) / P(¬c) * (P(P(w_i | c) / P(w_i | ¬c)))"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse='csr')
        negation_class_log_prior_ = self.class_log_prior_.sum(axis=0) - self.class_log_prior_
        return np.array(safe_sparse_dot(X, self.feature_log_prob_.T) +
                        (self.class_log_prior_ - negation_class_log_prior_))
