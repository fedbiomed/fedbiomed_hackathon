# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""TrainingPlan definitions for the scikit-learn ML framework.

This module implements the base class for all implementations of
Fed-BioMed training plans wrapping scikit-learn models.
"""

from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union

import numpy as np
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.data import NPDataLoader
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.models import SkLearnModel
from fedbiomed.common.optimizers import Optimizer, SklearnOptimizer
from fedbiomed.common.training_args import TrainingArgs

from ._base_training_plan import BaseTrainingPlan


class SKLearnTrainingPlan(BaseTrainingPlan, metaclass=ABCMeta):
    """Base class for Fed-BioMed wrappers of sklearn classes.

    Classes that inherit from this abstract class must:
    - Specify a `_model_cls` class attribute that defines the type
      of scikit-learn model being wrapped for training.
    - Implement a `set_init_params` method that:
      - sets and assigns the model's initial trainable weights attributes.
      - populates the `_param_list` attribute with names of these attributes.
    - Implement a `_training_routine` method that performs a training round
      based on `self.train_data_loader` (which is a `NPDataLoader`).

    Attributes:
        dataset_path: The path that indicates where dataset has been stored
        pre_processes: Preprocess functions that will be applied to the
            training data at the beginning of the training routine.
        training_data_loader: Data loader used in the training routine.
        testing_data_loader: Data loader used in the validation routine.

    !!! info "Notes"
        The trained model may be exported via the `export_model` method,
        resulting in a dump file that may be reloded using `joblib.load`
        outside of Fed-BioMed.
    """

    # Class attributes (defined or overridden by children classes).
    _model_cls: ClassVar[Type[BaseEstimator]]        # wrapped model class
    _model_dep: ClassVar[Tuple[str, ...]] = tuple()  # model-specific dependencies

    # Type annotations for post-init instance attributes.
    _model: SkLearnModel
    _aggregator_args: Dict[str, Any]
    _optimizer_args: Dict[str, Any]

    def __init__(self) -> None:
        """Initialize the SKLearnTrainingPlan."""
        super().__init__()
        self._training_args = {}  # type: Dict[str, Any]
        self.__type = TrainingPlans.SkLearnTrainingPlan
        self._batch_maxnum = 0
        self.dataset_path: Optional[str] = None
        self.add_dependency([
            "import inspect",
            "import numpy as np",
            "import pandas as pd",
            "from fedbiomed.common.training_plans import SKLearnTrainingPlan",
            "from fedbiomed.common.data import DataManager",
        ])
        self.add_dependency(list(self._model_dep))

    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: TrainingArgs,
            aggregator_args: Optional[Dict[str, Any]] = None,
        ) -> None:
        """Process model, training and optimizer arguments.

        Args:
            model_args: Arguments defined to instantiate the wrapped model.
            training_args: Arguments that are used in training routines
                such as epoch, dry_run etc.
                Please see [`TrainingArgs`][fedbiomed.common.training_args.TrainingArgs]
            aggregator_args: Arguments managed by and shared with the
                researcher-side aggregator.
        """
        # Add model dependencies (to enable saving this plan's code).
        self._configure_dependencies()
        # Instantiate the wrapped scikit-learn model.
        self._model = SkLearnModel(self._model_cls)
        # Parse input arguments.
        self._aggregator_args = aggregator_args or {}
        self._optimizer_args = training_args.optimizer_arguments() or {}
        self._training_args = training_args.pure_training_arguments()
        self._batch_maxnum = self._training_args.get('batch_maxnum', self._batch_maxnum)
        # Override default model parameters based on `self._model_args`.
        model_args.setdefault("verbose", 1)
        params = {
            key: model_args.get(key, val)
            for key, val in self._model.get_params().items()
        }
        self._model.set_params(**params)
        # Set up initial model weights (normally created by `self._model.fit`).
        self._model.set_init_params(model_args)
        # Get the optional Optimizer that users may have defined.
        optimizer = self.init_optimizer()
        # Wrap it and the model into a `SklearnOptimizer` instance.
        self._optimizer = SklearnOptimizer(self._model, optimizer)

    def set_data_loaders(
            self,
            train_data_loader: Union[DataLoader, NPDataLoader, None],
            test_data_loader: Union[DataLoader, NPDataLoader, None]
        ) -> None:
        """Sets data loaders

        Args:
            train_data_loader: Data loader for training routine/loop
            test_data_loader: Data loader for validation routine
        """
        args = (train_data_loader, test_data_loader)
        if not all(isinstance(data, NPDataLoader) for data in args):
            msg = (
                f"{ErrorNumbers.FB310.value}: SKLearnTrainingPlan expects "
                "NPDataLoader instances as training and testing data "
                f"loaders, but received {type(train_data_loader)} "
                f"and {type(test_data_loader)} respectively."
            )
            logger.error(msg)
            raise FedbiomedTrainingPlanError(msg)
        self.training_data_loader = train_data_loader
        self.testing_data_loader = test_data_loader

    def model_args(self) -> Dict[str, Any]:
        """Retrieve model arguments.

        Returns:
            Model arguments
        """
        return self._model.model_args

    def training_args(self) -> Dict[str, Any]:
        """Retrieve training arguments.

        Returns:
            Training arguments
        """
        return self._training_args

    def model(self) -> Optional[BaseEstimator]:
        """Retrieve the wrapped scikit-learn model instance.

        Returns:
            Scikit-learn model instance
        """
        if self._model is not None:
            return self._model.model
        else:
            return self._model

    def init_optimizer(self) -> Optional[Optimizer]:
        """Return an optional Optimizer to use for model training.

        Researchers may override this method to return an
        [Optimizer][fedbiomed.common.optimizers.Optimizer] instance, which
        will replace the default scikit-learn internal vanilla SGD optimizer.

        The `self.optimizer_args()` method may be used to access parameters
        received from the researcher so as to configure the returned optimizer.

        Returns:
            None, resulting in the scikit-learn model's internal optimizer
                being used. Subclasses may return an `Optimizer` instead.
        """
        return None

    def optimizer_args(self) -> Dict[str, Any]:
        """Retrieves optimizer arguments

        Returns:
            Optimizer arguments
        """
        return self._optimizer_args

    def training_routine(
            self,
            history_monitor: Optional['HistoryMonitor'] = None,
            node_args: Optional[Dict[str, Any]] = None
        ) -> None:
        """Training routine, to be called once per round.

        Args:
            history_monitor: optional HistoryMonitor
                instance, recording training metadata. Defaults to None.
            node_args: command line arguments for node.
                These arguments can specify GPU use; however, this is not
                supported for scikit-learn models and thus will be ignored.
        """
        if self._optimizer is None:
            raise FedbiomedTrainingPlanError(
                'Optimizer is None, please run `post_init` beforehand.'
            )
        if not isinstance(self.training_data_loader, NPDataLoader):
            msg = (
                f"{ErrorNumbers.FB310.value}: SKLearnTrainingPlan cannot "
                "be trained without a NPDataLoader as `training_data_loader`."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        # Run preprocessing operations.
        self._preprocess()
        # Warn if GPU-use was expected (as it is not supported).
        if node_args is not None and node_args.get('gpu_only', False):
            logger.warning(
                'Node would like to force GPU usage, but sklearn training '
                'plan does not support it. Training on CPU.'
            )
        # Run the model-specific training routine.
        try:
            return self._training_routine(history_monitor)
        except Exception as exc:
            msg = (
                f"{ErrorNumbers.FB605.value}: error while fitting "
                f"the model: {exc}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg) from exc

    @abstractmethod
    def _training_routine(
            self,
            history_monitor: Optional['HistoryMonitor'] = None
        ) -> None:
        """Model-specific training routine backend.

        Args:
            history_monitor: optional HistoryMonitor
                instance, recording the loss value during training.

        This method needs to be implemented by SKLearnTrainingPlan
        child classes, and is called as part of `training_routine`
        (that notably enforces preprocessing and exception catching).
        """

    def testing_routine(
            self,
            metric: Optional[MetricTypes],
            metric_args: Dict[str, Any],
            history_monitor: Optional['HistoryMonitor'],
            before_train: bool
        ) -> None:
        """Evaluation routine, to be called once per round.

        !!! info "Note"
            If the training plan implements a `testing_step` method
            (the signature of which is func(data, target) -> metrics)
            then it will be used rather than the input metric.

        Args:
            metric: The metric used for validation.
                If None, use MetricTypes.ACCURACY.
            history_monitor: HistoryMonitor instance,
                used to record computed metrics and communicate them to
                the researcher (server).
            before_train: Whether the evaluation is being performed
                before local training occurs, of afterwards. This is merely
                reported back through `history_monitor`.
        """
        # Check that the testing data loader is of proper type.
        if not isinstance(self.testing_data_loader, NPDataLoader):
            msg = (
                f"{ErrorNumbers.FB310.value}: SKLearnTrainingPlan cannot be "
                "evaluated without a NPDataLoader as `testing_data_loader`."
            )
            logger.error(msg)
            raise FedbiomedTrainingPlanError(msg)
        # If required, make up for the lack of specifications regarding target
        # classification labels.
        if self._model.is_classification and not hasattr(self.model(), 'classes_'):
            classes = self._classes_from_concatenated_train_test()
            setattr(self.model(), 'classes_', classes)
        # If required, select the default metric (accuracy or mse).
        if metric is None:
            if self._model.is_classification:
                metric = MetricTypes.ACCURACY
            else:
                metric = MetricTypes.MEAN_SQUARE_ERROR
        # Delegate the actual evalation routine to the parent class.
        super().testing_routine(
            metric, metric_args, history_monitor, before_train
        )

    def _classes_from_concatenated_train_test(self) -> np.ndarray:
        """Return unique target labels from the training and testing datasets.

        Returns:
            Numpy array containing the unique values from the targets wrapped
            in the training and testing NPDataLoader instances.
        """
        loaders = (self.training_data_loader, self.testing_data_loader)
        return np.unique([t for loader in loaders for d, t in loader])

    def type(self) -> TrainingPlans:
        """Getter for training plan type """
        return self.__type
