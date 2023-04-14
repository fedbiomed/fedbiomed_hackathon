# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""API and wrappers to interface framework-specific and generic optimizers."""

from abc import ABCMeta, abstractmethod
from typing import Any, Generic, Type, TypeVar, Union

import declearn
import declearn.model.torch
import torch

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedOptimizerError
from fedbiomed.common.models import Model, SkLearnModel, TorchModel
from fedbiomed.common.optimizers.optimizer import Optimizer


MT = TypeVar("MT", Model, SkLearnModel)
"""Generic TypeVar for framework-specific Model types."""

OT = TypeVar("OT")  # generic type-annotation for wrapped optimizers
"""Generic TypeVar for framework-specific Optimizer types"""


class ModelOptimizer(Generic[MT, OT], metaclass=ABCMeta):
    """Abstract base class for Optimizer and Model wrappers."""

    # Private class attributes, used for type-checking in `__init__`.
    _model_cls: Type[MT]
    _optim_cls: Type[Any]

    def __init__(
            self,
            model: MT,
            optimizer: Union[OT, Optimizer],
        ) -> None:
        """Instantiate the ModelOptimizer, wrapping a model and its optimizer.

        Args:
            model: model to train, interfaced via a framework-specific Model.
            optimizer: optimizer that will be used for optimizing the model,
                which may either be a declearn-based Fed-BioMed Optimizer or
                a framework-specific object.

        Raises:
            FedbiomedOptimizerError:
                Raised if model is not an instance of `_model_cls` (which may
                be a subset of the generic Model type).
        """
        if not isinstance(model, self._model_cls):
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621_b.value}, in `model` argument, expected "
                f"an instance of {self._model_cls} but got an object of type "
                f"{type(model)}."
            )
        if not isinstance(optimizer, (Optimizer, self._optim_cls)):
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621_b.value}, in `optimizer` argument, "
                f"expected an instance of {self._optim_cls} or Optimizer"
                f" but got an object of type {type(model)}."
            )
        self._model: MT = model
        self.optimizer: Union[OT, Optimizer] = optimizer

    def enter_training(self) -> None:
        """Set up the model and call any internal callback prior to training.

        This method is expected to be called at the start of each and every
        training round (which may comprise any number of steps or epochs).
        """
        self._model.init_training()
        if isinstance(self.optimizer, Optimizer):
            self.optimizer.init_round()

    def step(
            self,
            inputs: Any,
            target: Any,
            **kwargs: Any,
        ) -> None:
        """Perform a training and optimization step, updating the model."""
        try:
            self._model.train(inputs, target, **kwargs)
            if isinstance(self.optimizer, Optimizer):
                self._declearn_optim_step()
            else:
                self._native_optim_step()
        except Exception as exc:
            raise FedbiomedOptimizerError(
                f"{ErrorNumbers.FB621_b.value}: an error occured as part of the"
                f" training and optimization `step`: {repr(exc)}"
            ) from exc

    def _declearn_optim_step(self) -> None:
        """Perform an optimization step using a declearn-based Optimizer.

        This method should only be called as part of the `step` one,
        so that the wrapped model is guaranteed to have been trained
        (i.e. its gradients are ready to fetch and process).
        """
        grads = declearn.model.api.Vector.build(self._model.get_gradients())
        weights = declearn.model.api.Vector.build(self._model.get_weights())
        updates = self.optimizer.step(grads, weights)  # type: ignore
        self._model.apply_updates(updates.coefs)

    @abstractmethod
    def _native_optim_step(self) -> None:
        """Perform an optimization step using framework-specific code.

        This method should only be called as part of the `step` one,
        so that the wrapped model is guaranteed to have been trained
        (i.e. its gradients are ready to fetch and process), and the
        `self.optimizer` object is known to be framework-specific.
        """

    def exit_training(self) -> None:
        """Call any internal callback after training.

        This method is expected to be called at the end of each and every
        training round (which may comprise any number of steps or epochs).
        """


class SklearnOptimizer(ModelOptimizer):
    """ModelOptimizer subclass for Scikit-Learn SGD-based models.

    This `ModelOptimizer` subclass is tailored to wrap a `SkLearnModel`
    and train it using either its internal optimizer, or a Fed-BioMed
    `Optimizer`.
    """

    _model_cls = SkLearnModel
    _optim_cls = type(None)

    def enter_training(self) -> None:
        """Set up the model and call any internal callback prior to training.

        This method is expected to be called at the start of each and every
        training round (which may comprise any number of steps or epochs).
        """
        super().enter_training()
        if isinstance(self.optimizer, Optimizer):
            self._model.disable_internal_optimizer()

    def _native_optim_step(self) -> None:
        """Perform an optimization step using framework-specific code.

        This method should only be called as part of the `step` one,
        so that the wrapped model is guaranteed to have been trained
        (i.e. its gradients are ready to fetch and process) and the
        `self.optimizer` object is known to be None.
        """
        updates = self._model.get_gradients()
        self._model.apply_updates(updates)

    def exit_training(self) -> None:
        """Call any internal callback after training.

        This method is expected to be called at the end of each and every
        training round (which may comprise any number of steps or epochs).
        """
        if isinstance(self.optimizer, Optimizer):
            self._model.enable_internal_optimizer()


class TorchOptimizer(ModelOptimizer):
    """ModelOptimizer subclass for the Torch model framework.

    This `ModelOptimizer` subclass is tailored to wrap a `TorchModel`
    and train it using either a `torch.optim.Optimizer` or a Fed-BioMed
    `Optimizer`.
    """

    _model_cls = TorchModel
    _optim_cls = torch.optim.Optimizer

    def __init__(
            self,
            model: TorchModel,
            optimizer: Union[torch.optim.Optimizer, Optimizer],
        ) -> None:
        """Instantiate the TorchOptimizer, wrapping a model and its optimizer.

        Args:
            model: TorchModel to train.
            optimizer: optimizer that will be used for optimizing the model,
                which may either be a declearn-based Fed-BioMed Optimizer or
                a `torch.optim.Optimizer`.

        Raises:
            FedbiomedOptimizerError:
                Raised if model is not a `TorchModel` instance.
        """
        super().__init__(model, optimizer)

    def _native_optim_step(self) -> None:
        """Perform an optimization step using framework-specific code.

        This method should only be called as part of the `step` one,
        so that the wrapped model is guaranteed to have been trained
        (i.e. its gradients are ready to fetch and process), and the
        `self.optimizer` object is known to be a torch optimizer.
        """
        self.optimizer.step()  # type: ignore
