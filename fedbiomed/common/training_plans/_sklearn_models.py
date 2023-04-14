# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

# coding: utf-8

"""SKLearnTrainingPlan subclasses for models implementing `partial_fit`."""

import functools
from abc import ABCMeta
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor, Perceptron

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import SKLearnTrainingPlan
from fedbiomed.common.training_plans._training_iterations import (
    MiniBatchTrainingIterationsAccountant
)


__all__ = [
    "FedPerceptron",
    "FedSGDClassifier",
    "FedSGDRegressor",
]


class SKLearnTrainingPlanPartialFit(SKLearnTrainingPlan, metaclass=ABCMeta):
    """Base SKLearnTrainingPlan for models implementing `partial_fit`."""

    def __init__(self) -> None:
        super().__init__()
        if not hasattr(self._model_cls, 'partial_fit'):
            raise FedbiomedTrainingPlanError(
                f"{ErrorNumbers.FB302.value}: SKLearnTrainingPlanPartialFit"
                "requires the target scikit-learn model class to expose a"
                "`partial_fit` method."
            )

    def _training_routine(
            self,
            history_monitor: Optional['HistoryMonitor'] = None
        ) -> int:
        """Backend training routine for scikit-learn models with `partial_fit`.

        Args:
            history_monitor (HistoryMonitor, None): optional HistoryMonitor
                instance, recording the loss value during training.

        Returns:
            Number of data processed during training. This should be sent back
            to the reasearcher, for instance to weight the nodes' updates based
            on their relative contributions as part of the FedAvg aggregation.
        """
        # Set the number of training steps to perform.
        iterations_accountant = MiniBatchTrainingIterationsAccountant(self)
        # Optionally set up a func and arguments to report the training loss.
        report = False
        if (history_monitor is not None) and hasattr(self.model(), "verbose"):
            report = True
            loss_name = getattr(self.model(), "loss", "")
            loss_name = "Loss" + (f" {loss_name}" if loss_name else "")
            record_loss = functools.partial(
                history_monitor.add_scalar,
                train=True,
            )
            # force verbose = 1 to print losses
            verbose = self._model.get_params("verbose")
            self._model.set_params(verbose=1)
        # Run pre-training optimizer operations.
        self._optimizer.enter_training()
        # Iterate over epochs.
        for _ in iterations_accountant.iterate_epochs():
            training_data_iter: Iterator = iter(self.training_data_loader)
            # Iterate over data batches.
            for _ in iterations_accountant.iterate_batches():
                # Gather the data batch and keep track of iterations.
                inputs, target = next(training_data_iter)
                batch_size = self._infer_batch_size(inputs)
                iterations_accountant.increment_sample_counters(batch_size)
                # Run the training step.
                loss = self._train_over_batch(inputs, target, report)
                # Optionally report on the batch training loss.
                if (
                    report and not np.isnan(loss)
                    and iterations_accountant.should_log_this_batch()
                ):
                    self._report_training_loss(
                        loss, iterations_accountant, record_loss, loss_name, batch_size
                    )
        # Reset model verbosity to its initial value.
        if report:
            self._model.set_params(verbose=verbose)
        # Run post-training optimizer operations.
        self._optimizer.exit_training()
        # Return the total number of samples used during this round.
        return iterations_accountant.num_samples_observed_in_total

    def _train_over_batch(
            self,
            inputs: np.ndarray,
            target: np.ndarray,
            report: bool
        ) -> float:
        """Perform gradient descent over a single data batch.

        This method also resets the n_iter_ attribute of the
        scikit-learn model, such that n_iter_ will always equal
        1 at the end of the execution.

        Args:
            inputs: 2D-array of batched input features.
            target: 2D-array of batched target labels.
            report: Whether to capture and parse the training
                loss printed out to the console by the scikit-learn
                model. If False, or if parsing fails, return a nan.
        """
        # Run a training step, and collect standard outputs.
        stdout = []  # type: List[List[str]]
        self._optimizer.step(inputs, target, stdout=stdout if report else None)
        # Optionally report the training loss over this batch.
        if report:
            try:
                return self._parse_batch_loss(stdout, inputs, target)
            except Exception as exc:
                msg = (
                    f"{ErrorNumbers.FB605.value}: error while parsing "
                    f"training losses from stdout: {exc}"
                )
                logger.error(msg)
        # Otherwise, return nan as a fill-in value.
        return float('nan')

    def _parse_batch_loss(
            self,
            stdout: List[List[str]],
            inputs: np.ndarray,  # NOTE: these args are used by some children
            target: np.ndarray,
        ) -> float:
        """Parse logged loss values from captured stdout lines.

        Args:
            stdout: Captured stdout outputs from calling
                the model's partial fit, with one list per batched sample.
            inputs: Batched input features.
            target: Batched target labels.
        """
        values = [self._parse_sample_losses(sample) for sample in stdout]
        losses = np.array(values)
        return float(np.mean(losses))

    @staticmethod
    def _parse_sample_losses(
            stdout: List[str]
        ) -> List[float]:
        """Parse logged loss values from captured stdout lines."""
        losses = []  # type: List[float]
        for row in stdout:
            split = row.rsplit("loss: ", 1)
            if len(split) == 1:  # no "loss: " in the line
                continue
            try:
                losses.append(float(split[1]))
            except ValueError as exc:
                logger.error(f"Value error during monitoring: {exc}")
        return losses

    @staticmethod
    def _report_training_loss(
        loss: float,
        accountant: MiniBatchTrainingIterationsAccountant,
        record_loss: Callable[..., None],
        loss_name: str,
        batch_size: int,
    ) -> None:
        """Backend method to log and report training loss.

        This static method is merely a refactoring of code from the training
        routine and should never be called in any other context.
        """
        # Retrieve reporting information.
        # Semantics differ whether num_updates or epochs were specified.
        num_samples, num_samples_max = accountant.reporting_on_num_samples()
        num_iter, num_iter_max = accountant.reporting_on_num_iter()
        epoch = accountant.reporting_on_epoch()
        # Log the information.
        info = (
            "Train" + ("" if epoch is None else f" Epoch: {epoch}") + " | "
            + f"Iteration {num_iter}/{num_iter_max} | "
            + f"Samples  {num_samples}/{num_samples_max} "
            + f"({100 * num_iter / num_iter_max:.0f}%)\t"
            + f"Loss: {loss:.6f}"
        )
        logger.debug(info)
        # Record it into the HistoryMonitor.
        record_loss(
            metric={loss_name: loss},
            iteration=num_iter,
            epoch=epoch,
            num_samples_trained=num_samples,
            num_batches=num_iter_max,
            total_samples=num_samples_max,
            batch_samples=batch_size,
        )


class FedSGDRegressor(SKLearnTrainingPlanPartialFit):
    """Fed-BioMed training plan for scikit-learn SGDRegressor models."""

    _model_cls = SGDRegressor
    _model_dep = (
        "from sklearn.linear_model import SGDRegressor",
        "from fedbiomed.common.training_plans import FedSGDRegressor"
    )


class FedSGDClassifier(SKLearnTrainingPlanPartialFit):
    """Fed-BioMed training plan for scikit-learn SGDClassifier models."""

    _model_cls = SGDClassifier
    _model_dep = (
        "from sklearn.linear_model import SGDClassifier",
        "from fedbiomed.common.training_plans import FedSGDClassifier"
    )

    def _parse_batch_loss(
            self,
            stdout: List[List[str]],
            inputs: np.ndarray,
            target: np.ndarray
        ) -> float:
        """Parse logged loss values from captured stdout lines."""
        # Delegate binary classification case to parent class.
        if self.model_args()["n_classes"] == 2:
            return super()._parse_batch_loss(stdout, inputs, target)
        # Handle multilabel classification case.
        # Compute and batch-average sample-wise label-wise losses.
        values = [self._parse_sample_losses(sample) for sample in stdout]
        losses = np.array(values).mean(axis=0)
        # Compute the support-weighted average of label-wise losses.
        # NOTE: this assumes a (n, 1)-shaped targets array.
        classes = getattr(self.model(), "classes_")
        support = (target == classes).sum(axis=0)
        return float(np.average(losses, weights=support))


class FedPerceptron(FedSGDClassifier):
    """Fed-BioMed training plan for scikit-learn Perceptron models.

    This class inherits from FedSGDClassifier, and forces the wrapped
    scikit-learn SGDClassifier model to use a "perceptron" loss, that
    makes it equivalent to an actual scikit-learn Perceptron model.
    """

    _model_dep = (
        "from sklearn.linear_model import SGDClassifier",
        "from fedbiomed.common.training_plans import FedPerceptron"
    )

    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: TrainingArgs,
            aggregator_args: Optional[Dict[str, Any]] = None,
        ) -> None:
        # get default values of Perceptron model (different from SGDClassifier model default values)
        perceptron_default_values = Perceptron().get_params()
        sgd_classifier_default_values = SGDClassifier().get_params()
        # make sure loss used is perceptron loss - can not be changed by user
        model_args["loss"] = "perceptron"
        super().post_init(model_args, training_args)
        self._model.set_params(loss="perceptron")

        # collect default values of Perceptron and set it to the model FedPerceptron
        model_hyperparameters = self._model.get_params()
        for hyperparameter_name, val in perceptron_default_values.items():
            if model_hyperparameters[hyperparameter_name] == sgd_classifier_default_values[hyperparameter_name]:
                # this means default parameter of SGDClassifier has not been changed by user
                self._model.set_params(**{hyperparameter_name: val})
