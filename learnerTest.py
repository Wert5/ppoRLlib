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

logger = logging.getLogger("ray")

class MyPPOTorchLearner(PPOTorchLearner):

    @override(PPOTorchLearner)
    def configure_optimizers_for_module(
        self,
        module_id: str,
        hps: LearnerHyperparameters
    ) -> None:
        logger.warning("MyTorchLearner " + module_id)
        super().configure_optimizers_for_module(module_id, hps)
        module = self._module[module_id]
        optimizer = torch.optim.Adam(self.get_parameters(module))
        params = self.get_parameters(module)

        self.register_optimizer(
            module_id=module_id,
            optimizer=optimizer,
            params=params,
            lr_or_lr_schedule=None,
        )

        scheduler = Scheduler(
            fixed_value_or_schedule=hps.learning_rate,
            framework="torch",
            device=self._device,
        )

        self._optimizer_lr_schedules[optimizer] = scheduler
        cur_value = scheduler.get_current_value()
        logger.warning("Cur LR " + str(cur_value))
        if type(cur_value) is torch.Tensor:
            cur_value = cur_value.item()
        logger.warning("Cur LR Convert " + str(cur_value))
        # Set the optimizer to the current (first) learning rate.
        self._set_optimizer_lr(
            optimizer=optimizer,
            lr=cur_value,
        )
        logger.warning("Cur LR Optimizer Get " + str(self._get_optimizer_lr(optimizer=optimizer)))

        for g in optimizer.param_groups:
            logger.warning("G " + str(g))


    @override(PPOTorchLearner)
    def additional_update_for_module(
        self,
        *,
        module_id: ModuleID,
        hps: LearnerHyperparameters,
        timestep: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Apply additional non-gradient based updates for a single module.

        See `additional_update` for more details.

        Args:
            module_id: The id of the module to update.
            hps: The LearnerHyperparameters specific to the given `module_id`.
            timestep: The current global timestep (to be used with schedulers).
            **kwargs: Keyword arguments to use for the additional update.

        Returns:
            A dictionary of results from the update
        """
        results = super().additional_update_for_module(
            module_id=module_id,
            hps=hps,
            timestep=timestep,
            **kwargs
        )

        # Only cover the optimizer mapped to this particular module.
        for optimizer_name, optimizer in self.get_optimizers_for_module(module_id):
            # Only update this optimizer's lr, if a scheduler has been registered
            # along with it.
            if optimizer in self._optimizer_lr_schedules:
                new_lr = self._optimizer_lr_schedules[optimizer].update(
                    timestep=timestep
                )
                if type(new_lr) is torch.Tensor:
                    new_lr = new_lr.item()
                logger.warning("New LR Convert" + str(new_lr))
                self._set_optimizer_lr(optimizer, lr=new_lr)
                # Make sure our returned results differentiate by optimizer name
                # (if not the default name).
                stats_name = LEARNER_RESULTS_CURR_LR_KEY
                if optimizer_name != DEFAULT_OPTIMIZER:
                    stats_name += "_" + optimizer_name
                results.update({stats_name: new_lr})

        return results

MyPPOTorchLearner(module_spec=SingleAgentRLModuleSpec(module_class=MyTorchPPORLModule,
        model_config_dict={ "fcnet_hiddens" : [64],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [32],
            "post_fcnet_activation": "relu",
        })
)
