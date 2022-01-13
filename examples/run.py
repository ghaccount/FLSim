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

import json
from copy import deepcopy
from subprocess import Popen

BASE_CONFIG_PATH = "papaya/toolkit/simulation/examples/configs"


def command(config_file):
    return [
        "./buck-out/gen/papaya/toolkit/simulation/examples/femnist_example.par",
        "--config-file",
        config_file,
    ]


def change_value(section_path, config, value):
    new_config = deepcopy(config)
    curr = new_config
    for path in section_path[:-1]:
        curr = curr[path]
    curr[section_path[-1]] = value
    return new_config


def wait(processes):
    for p in processes:
        result = p.wait()
        if result == 1:
            print(f"Processed failed {p.args}")
    return []


server_lr = (
    ["config", "trainer", "server", "server_optimizer", "lr"],
    [1.0, 0.1, 0.01, 0.001],
)
momentum = (["config", "trainer", "server", "server_optimizer", "momentum"], [0, 0.9])
lambda_ = (
    ["config", "trainer", "client", "optimizer", "lambda_"],
    [10, 1.0, 0.1, 0.01, 0],
)
local_lr = (["config", "trainer", "client", "optimizer", "lr"], [1.0, 0.1, 0.01, 0.001])


processes = []
for config_file in ["femnist_cd.json", "femnist_sarah.json", "femnist_bilevel.json"]:
    with open(f"{BASE_CONFIG_PATH}/{config_file}", "r+") as f:
        config = json.load(f)
    configs = [server_lr, momentum, lambda_, local_lr]
    for params in configs:
        path, param_values = params
        for hp in param_values:
            new_config = change_value(path, config, hp)

            new_config_path = f"{BASE_CONFIG_PATH}/config.json"
            with open(new_config_path, "w+") as f:
                json.dump(new_config, f)
            cmd = command(new_config_path)
            print(cmd)
            with open(f"{BASE_CONFIG_PATH}/log.txt", "a") as log:
                processes.append(Popen(cmd, stdout=log))

            processes = wait(processes)
