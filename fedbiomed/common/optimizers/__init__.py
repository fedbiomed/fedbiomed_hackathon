# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Submodule providing with 'ModelOptimizer' and 'Optimizer' classes.

- [Optimizer][fedbiomed.common.optimizers.Optimizer]
    is a generic and modular SGD-core optimizer that can be used with all
    fedbiomed-supported model frameworks.
- [ModelOptimizer][fedbiomed.common.optimizers.ModelOptimizer]
    is an abstract base class that defines an API to wrap together a fedbiomed
    `Model` and an optimizer, that may either be an `Optimizer` instance or an
    object of framework-specific type (e.g. `torch.optim.Optimizer`).
- [SklearnOptimizer][fedbiomed.common.optimizers.SklearnOptimizer]
    is a `ModelOptimizer` subclass tailored for `SkLearnModel` models.
- [TorchOptimizer][fedbiomed.common.optimizers.TorchOptimizer]
    is a `ModelOptimizer` subclass tailored for `TorchModel` models.
"""

from .optimizer import Optimizer
from .model_optim import ModelOptimizer, SklearnOptimizer, TorchOptimizer
