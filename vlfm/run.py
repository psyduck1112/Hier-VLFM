# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os

# The following imports require habitat to be installed, and despite not being used by
# this script itself, will register several classes and make them discoverable by Hydra.
# This run.py script is expected to only be used when habitat is installed, thus they
# are hidden here instead of in an __init__.py file. This avoids import errors when used
# in an environment without habitat, such as when doing real-world deployment. noqa is
# used to suppress the unused import and unsorted import warnings by ruff.
import frontier_exploration  # noqa
import hydra  # noqa
from habitat import get_config  # noqa
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig

import vlfm.measurements.traveled_stairs  # noqa: F401
import vlfm.obs_transformers.resize  # noqa: F401
import vlfm.policy.action_replay_policy  # noqa: F401
import vlfm.policy.habitat_policies  # noqa: F401
import vlfm.utils.vlfm_trainer  # noqa: F401


class HabitatConfigPlugin(SearchPathPlugin):
    """
    自定义配置搜索路径插件，用于添加Habitat配置文件的搜索路径
    
    该插件通过继承SearchPathPlugin类，实现将Habitat配置文件路径添加到Hydra配置搜索路径中，
    使得Hydra能够在指定路径中查找配置文件。
    """
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="habitat", path="config/")
        """
        操作配置搜索路径，添加Habitat配置路径
        参数:
            search_path (ConfigSearchPath): 配置搜索路径对象，用于管理配置文件的搜索路径
        """   

register_hydra_plugin(HabitatConfigPlugin)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="experiments/vlfm_objectnav_hm3d",
)
def main(cfg: DictConfig) -> None:
    """
    主函数，用于运行VLFM实验
    该函数首先检查必要的数据文件是否存在，然后根据配置执行训练或评估任务。
    参数:
        cfg (DictConfig): Hydra配置对象，包含实验的所有配置参数
    """
    assert os.path.isdir("data"), "Missing 'data/' directory!"
    if not os.path.isfile("data/dummy_policy.pth"):
        print("Dummy policy weights not found! Please run the following command first:")
        print("python -m vlfm.utils.generate_dummy_policy")
        exit(1)
    # 修补配置并移除语义传感器配置（如果存在）
    cfg = patch_config(cfg)
    with read_write(cfg):
        try:
            cfg.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor")
        except KeyError:
            pass
    # 根据配置决定执行评估还是训练任务
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train") # 重点 # DictConfig类重写了getattr方法，使得可以.语法访问嵌套配置项，就像访问对象属性一样


if __name__ == "__main__":
    main()
