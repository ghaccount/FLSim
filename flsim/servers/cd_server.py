#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Algorithm 6:
    Server with partial participation (do not store every vt for every client)
    Randomized coordinate descent for personalized FL (without variance reduction)
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from flsim.active_user_selectors.simple_user_selector import (
    UniformlyRandomActiveUserSelectorConfig,
)
from flsim.channels.base_channel import (
    IdentityChannel,
    IFLChannel,
)
from flsim.channels.message import Message
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.model import IFLModel
from flsim.optimizers.server_optimizers import (
    ServerOptimizerConfig,
    FedAvgOptimizerConfig,
    OptimizerType,
)
from flsim.servers.aggregator import AggregationType, Aggregator
from flsim.servers.sync_servers import (
    ISyncServer,
    SyncServerConfig,
)
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf


class CDServer(ISyncServer):
    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IFLChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=CDServerConfig,
            **kwargs,
        )
        self._optimizer = OptimizerType.create_optimizer(
            # pyre-fixme[16]: `SyncServer` has no attribute `cfg`.
            config=self.cfg.server_optimizer,
            model=global_model.fl_get_module(),
        )
        self._global_model: IFLModel = global_model
        self._aggregator: Aggregator = Aggregator(
            module=global_model.fl_get_module(),
            aggregation_type=self.cfg.aggregation_type,
            only_federated_params=self.cfg.only_federated_params,
        )
        self._active_user_selector = instantiate(self.cfg.active_user_selector)
        self._channel: IFLChannel = channel or IdentityChannel()
        self.vt = deepcopy(global_model.fl_get_module())
        self.users_per_round = 0
        # Set the averaged model to be all zeros for now.
        FLModelParamUtils.zero_weights(self.vt)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.active_user_selector, "_target_"):
            cfg.active_user_selector = UniformlyRandomActiveUserSelectorConfig()
        if OmegaConf.is_missing(cfg.server_optimizer, "_target_"):
            cfg.server_optimizer = FedAvgOptimizerConfig()

    @property
    def global_model(self):
        return self._global_model

    def select_clients_for_training(
        self,
        num_total_users,
        users_per_round,
        data_provider: Optional[IFLDataProvider] = None,
        epoch: Optional[int] = None,
    ):
        self.num_users = num_total_users
        self.users_per_round = users_per_round
        return self._active_user_selector.get_user_indices(
            num_total_users=num_total_users,
            users_per_round=users_per_round,
            data_provider=data_provider,
            global_model=self.global_model,
            epoch=epoch,
        )

    def init_round(self):
        self._aggregator.zero_weights()
        self._optimizer.zero_grad()

    def receive_update_from_client(self, message: Message):
        message = self._channel.client_to_server(message)
        self._aggregator.apply_weight_to_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )

    def step(self):
        # sum of v_i^{t+1}
        sum_current_vi = self._aggregator.aggregate()

        # (1 / m) * sum of v_i^{t+1}
        FLModelParamUtils.multiply_model_by_weight(
            sum_current_vi, 1.0 / self.num_users, sum_current_vi
        )

        FLModelParamUtils.substract_model(self.average_v, sum_current_vi, self.vt)

        delta = deepcopy(self.vt)
        FLModelParamUtils.substract_model(
            self._global_model.fl_get_module(), self.vt, delta
        )

        FLModelParamUtils.set_gradient(
            model=self._global_model.fl_get_module(),
            reference_gradient=delta,
        )
        self.optimizer.step()


@dataclass
class CDServerConfig(SyncServerConfig):
    _target_: str = fullclassname(CDServer)
    aggregation_type: AggregationType = AggregationType.SUM
    server_optimizer: ServerOptimizerConfig = ServerOptimizerConfig()
    lambda_: float = 1.0
