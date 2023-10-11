import logging
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Hashable,
    Mapping,
    Optional,
    Sequence,
    Union,
    Tuple,
)

import torch

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.learner.learner import Learner, LearnerHyperparameters
from ray.rllib.core.rl_module.rl_module import (
    RLModule,
    ModuleID,
    SingleAgentRLModuleSpec,
)
from ray.rllib.utils.annotations import (
    override,
    OverrideToImplementCustomLogic,
)
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import Optimizer, Param, ParamDict, TensorType

from rlmoduleTest import MyTorchPPORLModule
from myScheduler import MyScheduler

logger = logging.getLogger("ray")

class MyPPOTorchLearner(PPOTorchLearner):

    @override(PPOTorchLearner)
    def configure_optimizers_for_module(
        self,
        module_id: str,
        hps: LearnerHyperparameters
    ) -> None:
        logger.warning("MyTorchLearner " + module_id)
        module = self._module[module_id]
        optimizer = torch.optim.Adam(self.get_parameters(module))
        params = self.get_parameters(module)

        self.register_optimizer(
            module_id=module_id,
            optimizer=optimizer,
            params=params,
            lr_or_lr_schedule=0.1234,
        )

        scheduler = MyScheduler(
            fixed_value_or_schedule=hps.learning_rate,
            framework="torch",
            device=self._device,
        )

        self._optimizer_lr_schedules[optimizer] = scheduler
        cur_value = scheduler.get_current_value()
        logger.warning("Cur LR " + str(cur_value))
        # Set the optimizer to the current (first) learning rate.
        self._set_optimizer_lr(
            optimizer=optimizer,
            lr=cur_value,
        )


MyPPOTorchLearner(module_spec=SingleAgentRLModuleSpec(module_class=MyTorchPPORLModule,
        model_config_dict={ "fcnet_hiddens" : [64],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [32],
            "post_fcnet_activation": "relu",
        })
)
