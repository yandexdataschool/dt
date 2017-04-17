from __future__ import division, print_function, absolute_import

import numpy
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state

from . import losses
from .preprocessing import BinTransformer
from hep_ml.commonutils import check_xyw, take_last, score_to_proba

__author__ = 'Alex Rogozhnikov'
__all__ = ['DecisionTrainClassifier', 'DecisionTrainRegressor']


class _Bootstrapper(object):
    def __init__(self, random_state, n_samples, bootstrap=True, dtype='float32'):
        """Tricky fast substitute to bagging"""
        self.random_state = random_state
        self.n_samples = n_samples
        if bootstrap:
            self.part = self.generate_weights_reference(n_samples=2 * n_samples + 1000).astype(dtype=dtype)
        else:
            self.part = numpy.ones(2 * n_samples + 1000)

    def generate_weights(self):
        shift = self.random_state.randint(0, len(self.part) - self.n_samples)
        return self.part[shift:shift + self.n_samples]

    def generate_weights_reference(self, n_samples):
        return numpy.bincount(self.random_state.randint(0, n_samples, size=n_samples), minlength=n_samples)


def chunks(sequence, step):
    """Yield successive chunks from sequence."""
    for i in range(0, len(sequence), step):
        yield sequence[i:i + step]


class DecisionTrainBase(BaseEstimator):
    _max_thresholds = 64
    _l2_regularization = 3.
    _is_classifier = None
    _indices_type = numpy.uint32
    _skip_preprocessing = False

    def __init__(self, n_estimators=10000, depth=6, loss=None, learning_rate=0.05, max_features=0.8,
                 l2_regularization=100., use_friedman_mse=True, update_step=4,
                 bootstrap=True, train_features=None, n_threads=4):
        """
        Decision Train is ultra-fast modification of gradient boosting over decision trees.
        :param n_estimators: number of trees
        :param depth: depth of trees
        :param loss: loss function, python object or None. MSE or LogLoss is used if None is passed.
        :param l2_regularization: regularization on values in leaves.
        :param learning_rate: float, shrinkage of step.
        :param max_features: int (amount) or float (portion) of features tested to select new split.
        :param use_friedman_mse: if True, uses modified MSE to select splits.
        :param update_step: gradients are recomputed once in `update_step` iterations.
        :param bootstrap: if True, uses bagging during split selection
        :param train_features: list with names of features used to build trees.
            All features are used if None was passed.
            Some features can be used in loss (like query id), but not participate in building trees.
        :param n_threads: number of threads
        """
        self.update_step = update_step
        self.n_estimators = n_estimators
        self.depth = depth
        self.loss = loss
        self.l2_regularization = l2_regularization
        self.learning_rate = learning_rate
        self.max_features = max_features
        self.use_friedman_mse = use_friedman_mse
        self.update_step = update_step
        self.bootstrap = bootstrap
        self.train_features = train_features
        self.n_threads = n_threads

        self.estimators = []
        self.transformer = None
        self.random_state = check_random_state(42)
        self.n_features_ = None

    def _transform(self, X, extend_to=1):
        # taking needed features
        if self.train_features is not None:
            X = X.loc[:, self.train_features]
        if self._skip_preprocessing:
            if (X.dtype == 'uint8') and (len(X) % extend_to == 0):
                return numpy.require(X, dtype='uint8', requirements=['A', 'W', 'F', 'O', 'E'])
            else:
                raise ValueError('please use BinTransformer to compress X.')
        if self.transformer is None:
            self.transformer = BinTransformer(max_bins=self._max_thresholds)
            self.transformer.fit(X)
        return self.transformer.transform(X, extend_to=extend_to)

    def fit(self, X, y, sample_weight=None):
        if self._is_classifier:
            self.classes_, y = numpy.unique(y, return_inverse=True)
            assert len(self.classes_) == 2, 'only binary classification supported'

        X, y, sample_weight = check_xyw(X, y, sample_weight=sample_weight, classification=self._is_classifier)

        if self.loss is None:
            if self._is_classifier:
                self.loss = losses.LogLoss(n_threads=self.n_threads)
            else:
                self.loss = losses.MSELoss()

        self.loss.fit(X, y, sample_weight=sample_weight)

        X = self._transform(X)
        n_samples, self.n_features_ = X.shape

        if isinstance(self.max_features, int):
            used_features = self.max_features
        else:
            assert isinstance(self.max_features, float)
            used_features = int(numpy.ceil(self.max_features * self.n_features_))
        assert 0 < used_features <= self.n_features_, 'wrong max_features: {}'.format(self.max_features)

        assert numpy.max(X) < 128, 'bin indices should be smaller than 128'
        n_thresholds = int(numpy.max(X)) + 1

        self.estimators = []
        current_indices = numpy.zeros(n_samples, dtype=self._indices_type)
        pred = numpy.zeros(n_samples, dtype='float32')
        self.initial_bias_ = self.compute_optimal_step(pred)
        pred += self.initial_bias_

        bootstrapper = _Bootstrapper(self.random_state, bootstrap=self.bootstrap, n_samples=n_samples)

        targets, weights = self.loss.prepare_tree_params(pred)
        for stage in range(self.n_estimators):
            bootstrap_weights = bootstrapper.generate_weights()

            columns_to_test = numpy.sort(self.random_state.choice(self.n_features_, size=used_features, replace=False))

            feature, cut, best_improvements, best_cuts = build_decision(
                X, targets=targets, weights=weights, bootstrap_weights=bootstrap_weights,
                current_indices=current_indices, columns_to_test=columns_to_test, depth=self.depth,
                n_thresh=n_thresholds, reg=self._l2_regularization,
                use_friedman_mse=self.use_friedman_mse, n_threads=self.n_threads
            )

            leaf_values_placeholder = numpy.zeros(2 ** self.depth, dtype='float32')
            self.estimators.append([feature, cut, leaf_values_placeholder])

            if (self.n_estimators - 1 - stage) % self.update_step == 0:
                self._update_leaves_and_predictions(current_indices, pred, target=targets, hessians=weights,
                                                    stage=stage, n_stages=min(self.update_step, len(self.estimators)))
                # computing new tree parameters
                targets, weights = self.loss.prepare_tree_params(pred)

        return self

    def compute_optimal_step(self, y_pred):
        step = 0.
        for _ in range(10):
            target, weights = self.loss.prepare_tree_params(y_pred + step)
            step += 0.5 * numpy.average(target, weights=weights)
        return step

    def _update_leaves_and_predictions(self, current_indices, current_predictions, target, hessians, stage, n_stages):
        gradients = target * hessians
        n_hybrid_leaves = 2 ** (self.depth + n_stages - 1)
        # spoiling indices, but those are not expected to be needed anymore
        current_indices &= n_hybrid_leaves - 1
        hybrid_grads, hybrid_hesss = parallel_bincount(current_indices, gradients, hessians, n_bins=n_hybrid_leaves)
        hybrid_leaf_values = numpy.zeros(n_hybrid_leaves, dtype='float32')
        help_indices = numpy.arange(n_hybrid_leaves)
        for i in reversed(range(n_stages)):
            processed_stage = stage - i
            mapping = (help_indices >> i) & (2 ** self.depth - 1)
            grads = numpy.bincount(mapping, weights=hybrid_grads - hybrid_hesss * hybrid_leaf_values)
            hesss = numpy.bincount(mapping, weights=hybrid_hesss)
            leaf_values = self._compute_learning_rate(processed_stage) * grads / (hesss + self.l2_regularization)
            leaf_values = leaf_values.astype('float32')
            feature, cut, _ = self.estimators[processed_stage]
            self.estimators[processed_stage] = feature, cut, leaf_values
            hybrid_leaf_values += leaf_values[mapping]

        current_predictions += hybrid_leaf_values[current_indices]

    def _compute_learning_rate(self, stage):
        return self.learning_rate * (stage + 1.) / self.n_estimators

    def _staged_decision_function_naive(self, X):
        mask = 2 ** self.depth - 1
        current_indices = numpy.zeros(len(X), dtype=self._indices_type)
        X = self._transform(X)
        result = numpy.zeros(len(X), dtype='float32') + self.initial_bias_
        for feature, cut, leaf_values in self.estimators:
            current_indices <<= 1
            current_indices |= (X[:, feature] > cut).astype(self._indices_type)
            result += leaf_values[current_indices & mask]
            yield result

    def staged_decision_function(self, X, step=100):
        n_samples = len(X)
        X = self._transform(X, extend_to=8)
        X_64 = X.T.view('uint64').T
        current_indices = numpy.zeros(len(X), dtype=self._indices_type)
        result = numpy.zeros(len(X), dtype='float32') + self.initial_bias_
        chunk_size = min(7, 16 + 1 - self.depth)
        for estimators_group in chunks(self.estimators, step):
            for estimator_subgroup in chunks(estimators_group, chunk_size):
                features, thresholds, leaf_values_array = zip(*estimator_subgroup)
                predict_several_trees(X_64, current_indices, result, self.depth, features, thresholds,
                                      leaf_values_array)
            yield result[:n_samples]

    @property
    def feature_importances_(self):
        """Feature importances (only for the variables used in training) """
        result = numpy.bincount([feature for feature, split, values in self.estimators], minlength=self.n_features_)
        return result / result.sum()

    def decision_function(self, X):
        return take_last(self.staged_decision_function(X))


class DecisionTrainClassifier(DecisionTrainBase, ClassifierMixin):
    _is_classifier = True

    def predict_proba(self, X):
        return score_to_proba(self.decision_function(X))

    def staged_predict_proba(self, X, step=100):
        for score in self.staged_decision_function(X, step=step):
            yield score_to_proba(score)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class DecisionTrainRegressor(DecisionTrainBase, RegressorMixin):
    _is_classifier = False

    def predict(self, X):
        return self.decision_function(X)

    def staged_predict(self, X, step=100):
        for pred in self.staged_decision_function(X, step=step):
            yield pred


# region Computationally expensive code

def build_decision_numpy(X, targets, weights, bootstrap_weights, current_indices, columns_to_test,
                         depth, n_thresh=128, reg=5., use_friedman_mse=True, n_threads=None):
    """ Builds one more split and updates indices """
    n_leaves = 2 ** (depth - 1)
    minlength = n_thresh * n_leaves
    best_improvements = []
    best_cuts = []

    leaf_indices = current_indices & (n_leaves - 1)
    assert leaf_indices.max() < n_leaves

    hessians = weights * bootstrap_weights
    gradients = targets * hessians

    for column_index, column in enumerate(columns_to_test):
        indices = leaf_indices.copy()
        indices |= X[:, column].astype(leaf_indices.dtype) << (depth - 1)

        bin_gradients = numpy.bincount(indices, weights=gradients, minlength=minlength).reshape([n_thresh, n_leaves]).T
        bin_hessians = numpy.bincount(indices, weights=weights, minlength=minlength).reshape([n_thresh, n_leaves]).T

        bin_gradients = numpy.cumsum(bin_gradients, axis=1)
        bin_gradients_op = bin_gradients[:, [-1]] - bin_gradients
        bin_hessians = numpy.cumsum(bin_hessians, axis=1)
        bin_hessians_op = bin_hessians[:, [-1]] - bin_hessians

        if not use_friedman_mse:
            # x1 ** 2 / w1 + x2 ** 2 / w2
            improvements = bin_gradients ** 2 / (bin_hessians + reg)
            improvements += bin_gradients_op ** 2 / (bin_hessians_op + reg)
        else:
            # (w1 x2 - x1 w2) ** 2 / w1 / w2 / (w1 + w2)
            improvements = (bin_gradients * bin_hessians_op - bin_gradients_op * bin_hessians) ** 2.
            improvements /= (bin_hessians + reg) * (bin_hessians_op + reg) * (bin_hessians[:, [-1]] + reg)

        improvements = improvements.sum(axis=0)

        best_improvement = numpy.max(improvements)
        best_cut = numpy.argmax(improvements)

        best_improvements.append(best_improvement)
        best_cuts.append(best_cut)

    selected_feature_id = numpy.argmax(best_improvements)
    feature = columns_to_test[selected_feature_id]
    cut = best_cuts[selected_feature_id]
    # updating indices after step
    current_indices <<= 1
    current_indices |= (X[:, feature] > cut).astype(current_indices.dtype)

    return feature, cut, numpy.array(best_improvements), numpy.array(best_cuts)


def predict_several_trees_numpy(X_64, indices, current_predictions, depth, features, thresholds, leaf_values_array):
    assert len(features) == len(thresholds) == len(leaf_values_array)
    X = X_64.T.view('uint8').T
    hybrid_leaf_values = numpy.zeros(2 ** (depth + len(features) - 1), dtype='float32')
    help_indices = numpy.arange(len(hybrid_leaf_values))
    mask = 2 ** depth - 1
    for i, leaf_values in enumerate(reversed(leaf_values_array)):
        assert len(leaf_values) == 2 ** depth
        hybrid_leaf_values += leaf_values[(help_indices >> i) & mask]
    for feature, threshold in zip(features, thresholds):
        indices <<= 1
        indices |= X[:, feature] > threshold
    current_predictions += hybrid_leaf_values[indices & (len(hybrid_leaf_values) - 1)]


def parallel_bincount_numpy(indices, grads, hesss, n_bins):
    bin_gradients = numpy.bincount(indices, weights=grads, minlength=n_bins)
    bin_hessians = numpy.bincount(indices, weights=hesss, minlength=n_bins)
    return bin_gradients, bin_hessians

# endregion

build_decision = build_decision_numpy
predict_several_trees = predict_several_trees_numpy
parallel_bincount = parallel_bincount_numpy

