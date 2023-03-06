from abc import abstractmethod
from typing import Callable, Dict, Union

from fedbiomed.common. models import Model, SkLearnModel
from fedbiomed.common.exceptions import FedbiomedOptimizerError
from fedbiomed.common.optimizers.optimizer import Optimizer

import declearn
from declearn.model.api import Vector
import torch
import numpy as np

class GenericOptimizer:
    model: Model
    optimizer: Union[Optimizer, None]
    _step_method: Callable = NotImplemented
    _return_type: Union[None, Callable]
    def __init__(self, model: Model, optimizer: Union[Optimizer, None], return_type: Union[None, Callable] = None):
        self.model = model
        self.optimizer = optimizer
        self._return_type = return_type
        if isinstance(optimizer, declearn.optimizer.Optimizer):
            self._step_method = optimizer.step_modules
            if hasattr(model, 'gradients_computation'):
                self.model.gradients_computation: Callable = self.model._declearn_gradients_computation
        else:
            if hasattr(optimizer,'step_native'):
                self._step_method = optimizer.step_native
            else:
                raise FedbiomedOptimizerError(f"Optimizer {optimizer} has not `step_native` method, can not proceed")
        
            
    def step(self) -> Callable:
        if self._step_method is NotImplemented:
            raise FedbiomedOptimizerError("Error, method used for step not implemeted yet")
        return self._step_method()
    
    def step_modules(self):
        grad: Vector = self.model.get_gradients(self._return_type)
        weights: Vector = self.model.get_weights(self._return_type)
        updates = self.optimizer.step(grad, weights)
        self.model.apply_updates(updates)

    @classmethod
    def load_state(cls, state):
        # state: breakpoint content for optimizer
        return cls
    def save_state(self):
        pass
    def set_aux(self):
        pass
    def get_aux(self):
        pass
    def init_training(self):
        self.init_training()

    @abstractmethod
    def step_native(selfs):
        """_summary_

        Raises:
            FedbiomedOptimizerError: _description_
            FedbiomedOptimizerError: _description_
        """


class TorchOptimizer(GenericOptimizer):
    def __init__(self, model, optimizer: torch.optim, return_type=None):
        if not isinstance(optimizer, torch.optim):
            raise FedbiomedOptimizerError(f"Error, expected a `torch.optim` optimizer, but got {type(optimizer)}")
        super().__init__(model, optimizer, return_type=None)

    def setp_native(self):
        self.optimizer.step()
        
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def fed_prox(self, loss: torch.float, mu: float) -> torch.float:
        loss += float(self._fedprox_mu) / 2 * self.__norm_l2()
    
    def scaffold(self):
        pass # FIXME: should we implement scaffold here?

    def __norm_l2(self) -> float:
        """Regularize L2 that is used by FedProx optimization

        Returns:
            L2 norm of model parameters (before local training)
        """
        norm = 0

        for current_model, init_model in zip(self.model.model().parameters(), self.model.init_params):
            norm += ((current_model - init_model) ** 2).sum()
        return norm
    
    
class SkLearnOptimizer(GenericOptimizer):
    
    def __init__(self, model: SkLearnModel, optimizer, return_type=None):
        if not isinstance(model, SkLearnModel):
            raise FedbiomedOptimizerError(f"Error in model argument: expected a `SkLearnModel` object, but got {type(model)}")
        super().__init__(model, optimizer, return_type=None)
        self.model.gradients_computation: Callable = self.model._native_gradients_computation
    def step_native(self):
        gradients: Dict[str, np.ndarray] = self.model.get_gradients()
        self.model.apply_updates(gradients)
