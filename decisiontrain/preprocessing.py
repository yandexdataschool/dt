from __future__ import division, print_function, absolute_import

import numpy
from sklearn.base import BaseEstimator, TransformerMixin

__author__ = 'Alex Rogozhnikov'


class BinTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_bins=128):
        """
        Bin transformer transforms all features (which are expected to be numerical)
        to small integers.

        :param int max_bins: maximal number of bins along each axis.
        """
        self.max_bins = max_bins

    def fit(self, X, y=None, sample_weight=None):
        """Prepare transformation rule, compute bin edges.

        :param X: array-like with data
        :param y: labels, ignored
        :param sample_weight: weights, ignored
        :return: self
        """
        assert self.max_bins < 255, 'Too high number of bins!'
        X = numpy.require(X, dtype='float32')
        self.percentiles_ = []
        for column in range(X.shape[1]):
            values = numpy.array(X[:, column])
            if len(numpy.unique(values)) < self.max_bins:
                self.percentiles_.append(numpy.unique(values)[:-1])
            else:
                targets = numpy.linspace(0, 100, self.max_bins + 1)[1:-1]
                self.percentiles_.append(numpy.percentile(values, targets))
        return self

    def transform(self, X, extend_to=1):
        """
        :param X: array-like with data
        :param int extend_to: extends number of samples to be divisible by extend_to
        :return: numpy.array with transformed features, dtype is 'uint8' for space efficiency.
        """
        X = numpy.require(X, dtype='float32')
        assert X.shape[1] == len(self.percentiles_), 'Wrong names of columns'
        n_samples = len(X)
        extended_length = ((n_samples + extend_to - 1) // extend_to) * extend_to
        bin_indices = numpy.zeros([extended_length, X.shape[1]], dtype='uint8', order='F')
        for i, percentiles in enumerate(self.percentiles_):
            bin_indices[:n_samples, i] = numpy.searchsorted(percentiles, X[:, i])
        return bin_indices
