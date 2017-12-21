import numpy as np
from scipy.sparse import issparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.validation import check_is_fitted


class UniversalSetNB(MultinomialNB):
    complement_class_prior = None

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")
        fc = safe_sparse_dot(Y.T, X)
        self.feature_count_ += fc
        cc = Y.sum(axis=0)
        self.class_count_ += cc
        self.complement_feature_count_ += fc.sum(axis=0) - fc
        self.complement_class_count_ += cc

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))

        smoothed_fc = self.complement_feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.complement_feature_log_prob_ = (np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1)))

    def _update_class_log_prior(self, class_prior=None):
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError("Number of priors must match number of"
                                 " classes.")
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            # empirical prior, with sample_weight taken into account
            self.class_log_prior_ = (np.log(self.class_count_) -
                                     np.log(self.class_count_.sum()))
        else:
            self.class_log_prior_ = np.zeros(n_classes) - np.log(n_classes)

        if self.complement_class_prior is not None:
            self.complement_class_log_prior_ = np.log(self.complement_class_prior)
        elif self.fit_prior:
            # empirical prior, with sample_weight taken into account
            self.complement_class_log_prior_ = (np.log(self.complement_class_count_) -
                                              np.log(self.complement_class_count_.sum()))
        else:
            self.complement_class_log_prior_ = np.zeros(n_classes) - np.log(n_classes)

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X
        P(c) / P(¬c) * (Π(P(w_i | c) / ΠP(w_i | ¬c)))"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse='csr')

        features_doc_logprob = safe_sparse_dot(X, self.feature_log_prob_.T)
        numerator = features_doc_logprob + self.class_log_prior_

        denominator = safe_sparse_dot(X, self.complement_feature_log_prob_.T) + self.complement_class_log_prior_
        return np.array(numerator - denominator)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incremental fit on a batch of samples.
        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.
        This is especially useful when the whole dataset is too big to fit in
        memory at once.
        This method has some performance overhead hence it is better to call
        partial_fit on chunks of data that are as large as possible
        (as long as fitting in the memory budget) to hide the overhead.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        classes : array-like, shape = [n_classes] (default=None)
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.
        sample_weight : array-like, shape = [n_samples] (default=None)
            Weights applied to individual samples (1. for unweighted).
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        _, n_features = X.shape

        if _check_partial_fit_first_call(self, classes):
            # This is the first call to partial_fit:
            # initialize various cumulative counters
            n_effective_classes = len(classes) if len(classes) > 1 else 2
            self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
            self.feature_count_ = np.zeros((n_effective_classes, n_features),
                                           dtype=np.float64)
            self.complement_class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
            self.complement_feature_count_ = np.zeros((n_effective_classes, n_features),
                                           dtype=np.float64)
        elif n_features != self.coef_.shape[1]:
            msg = "Number of features %d does not match previous data %d."
            raise ValueError(msg % (n_features, self.coef_.shape[-1]))

        Y = label_binarize(y, classes=self.classes_)
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        n_samples, n_classes = Y.shape

        if X.shape[0] != Y.shape[0]:
            msg = "X.shape[0]=%d and y.shape[0]=%d are incompatible."
            raise ValueError(msg % (X.shape[0], y.shape[0]))

        # label_binarize() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently
        Y = Y.astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= check_array(sample_weight).T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        self._count(X, Y)

        # XXX: OPTIM: we could introduce a public finalization method to
        # be called by the user explicitly just once after several consecutive
        # calls to partial_fit and prior any call to predict[_[log_]proba]
        # to avoid computing the smooth log probas at each call to partial fit
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self

    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        sample_weight : array-like, shape = [n_samples], (default=None)
            Weights applied to individual samples (1. for unweighted).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, 'csr')
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        Y = Y.astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= check_array(sample_weight).T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features),
                                       dtype=np.float64)
        self.complement_class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.complement_feature_count_ = np.zeros((n_effective_classes, n_features),
                                                dtype=np.float64)
        self._count(X, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self
