"""
Using losses from hep_ml.losses is possible, but those are quite slow.
"""
from __future__ import division, print_function, absolute_import

import numpy
from sklearn.base import BaseEstimator


def log_loss_grad_numpy(y_signed, pred, n_threads=None):
    return y_signed - numpy.tanh(pred / 2)


log_loss_grad = log_loss_grad_numpy


class LogLoss(BaseEstimator):
    def __init__(self, n_threads=2):
        self.n_threads = n_threads

    def fit(self, X, y, sample_weight=None):
        assert set(y) == {0, 1}
        self.sample_weight = numpy.require(sample_weight, dtype='float32')
        self.y_signed = numpy.require(2 * y - 1, dtype='float32')
        return self

    def prepare_tree_params(self, pred):
        return log_loss_grad(self.y_signed, pred, n_threads=self.n_threads), self.sample_weight


class MSELoss(BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        self.sample_weight = numpy.require(sample_weight, dtype='float32')
        self.y = numpy.require(y, dtype='float32')
        return self

    def prepare_tree_params(self, pred):
        return self.y - pred, self.sample_weight
