from __future__ import division, print_function, absolute_import

__author__ = 'Alex Rogozhnikov'
__version__ = '0.1.0'

try:
    from ._dtrain import dtrain
except:
    raise ImportError('impossible to use fortran-optimized decision train')

from . import decisiontrain, losses
from .decisiontrain import DecisionTrainClassifier, DecisionTrainRegressor, BinTransformer

# substituting functions in fortran
decisiontrain.build_decision = decisiontrain.build_decision_fortran = dtrain.build_decision_fortran
decisiontrain.predict_several_trees = decisiontrain.predict_several_trees_fortran = dtrain.predict_several_trees_fortran
losses.log_loss_grad = losses.log_loss_grad_fortran = dtrain.log_loss_grad
decisiontrain.parallel_bincount = decisiontrain.parallel_bincount_fortran = dtrain.parallel_bincount_fortran
