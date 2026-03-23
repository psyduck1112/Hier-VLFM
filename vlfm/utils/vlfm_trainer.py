# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Modifications Copyright (c) 2026 Yikang.

import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    extract_scalars_from_info as extract_scalars_from_info_habitat,
)
from omegaconf import OmegaConf


# 它会过滤 info 字典，移除所有值是列表的项，然后把结果交给 Habitat自带的标准函数
def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    info_filtered = {k: v for k, v in info.items() if not isinstance(v, list)}
    return extract_scalars_from_info_habitat(info_filtered)


@baseline_registry.register_trainer(name="vlfm")
# 这里使用了注册模式，将VLFMTrainer注册为名为"vlfm"的训练器，可以在配置文件中通过这个名字引用此类。
class VLFMTrainer(PPOTrainer):
    envs: VectorEnv

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:  # 如果启用分布式模式
            raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        # 加载检查点
        if self.config.habitat_baselines.eval.should_load_ckpt:  # 如果需要加载检查点中的模型权重
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            # 从检查点文件加载模型权重
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            step_id = ckpt_dict["extra_state"]["step"]  # 读取训练步数
            print(step_id)
        else:
            ckpt_dict = {"config": None}  # 不加载检查点时，创建一个空字典
        # 选择并读取配置
        config = self._get_resume_state_config_or_new_config(ckpt_dict["config"])

        # 数据集分割
        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        # ======================== 视频录制和传感器配置 ========================
        # 如果启用了视频录制选项（如"disk"或"tensorboard"），需要配置额外的传感器
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            # 1. 获取智能体的传感器配置
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors  # 获取现有的传感器列表

            # 2. 获取评估时需要的额外传感器（如第三人称视角、鸟瞰图等）
            extra_sensors = config.habitat_baselines.eval.extra_sim_sensors

            # 3. 将额外传感器添加到智能体传感器配置中
            with read_write(agent_sensors):  # 启用配置修改模式
                agent_sensors.update(extra_sensors)  # 合并传感器配置

            # 4. 更新环境配置，确保新传感器的数据能被正确处理
            with read_write(config):  # 启用配置修改模式
                # 确保所有额外的传感器视图都被包含在observation keys中
                if config.habitat.gym.obs_keys is not None:
                    for render_view in extra_sensors.values():
                        # 检查传感器UUID是否已在观察键列表中
                        if render_view.uuid not in config.habitat.gym.obs_keys:
                            # 如果不在，则添加到观察键列表，确保数据能被收集
                            config.habitat.gym.obs_keys.append(render_view.uuid)

                # 启用调试渲染模式，生成高质量的视觉输出用于视频录制
                config.habitat.simulator.debug_render = True

        # ======================== 配置信息输出 ========================
        # 如果启用了详细模式，打印完整的环境配置信息用于调试
        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        # 环境初始化
        self._init_envs(config, is_eval=True)

        # 创建agent
        self._agent = self._create_agent(None)  # SingleAgentAccessMgr实例,初始化时创建了 self._agent._actor_critic实例
        action_shape, discrete_actions = get_action_space_info(self._agent.policy_action_space)

        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)  # 加载模型状态

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        # 初始化当前episode奖励、先前动作存储、未完成掩码
        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                self.config.habitat_baselines.num_environments,
                *self._agent.hidden_state_shape,
            ),
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        # 存储每个情节的统计信息与评估次数
        stats_episodes: Dict[Any, Any] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        # 初始化RGB帧存储，用于视频录制 如果启用了视频选项，则创建视频目录
        rgb_frames: List[List[np.ndarray]] = [[] for _ in range(self.config.habitat_baselines.num_environments)]
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)
        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes, dataset only has {{total_num_eps}}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert number_of_eval_episodes > 0, "You must specify a number of evaluation episodes with test_episode_count"

        # 初始化进度条和设置评估模式
        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self._agent.eval()  # 返回的是self._agent._actor_critic.eval()

        # 导入可视化工具
        from vlfm.utils.habitat_visualizer import HabitatVis

        num_successes = 0
        num_total = 0
        hab_vis = HabitatVis()
        # 主评估循环：持续运行直到完成指定数量的评估回合
        while len(stats_episodes) < (number_of_eval_episodes * evals_per_ep) and self.envs.num_envs > 0:
            # 获取当前环境中的情节信息
            current_episodes_info = self.envs.current_episodes()

            # 在推理模式下执行动作预测
            with inference_mode():  # 使用推理模式，不计算梯度
                # 获取动作数据
                action_data = self._agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                # 如果设置了动作记录目录，则将动作ID记录到文件中
                if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
                    action_id = action_data.actions.cpu()[0].item()
                    filepath = os.path.join(
                        os.environ["VLFM_RECORD_ACTIONS_DIR"],
                        "actions.txt",
                    )
                    # 如果文件不存在则创建
                    if not os.path.exists(filepath):
                        open(filepath, "w").close()
                    # 将动作ID追加写入文件
                    with open(filepath, "a") as f:
                        f.write(f"{action_id}\n")

                # 更新RNN隐藏状态和先前动作
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = action_data.rnn_hidden_states
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    # 根据should_inserts标志选择性更新隐藏状态和动作
                    for i, should_insert in enumerate(action_data.should_inserts):
                        if should_insert.item():
                            test_recurrent_hidden_states[i] = action_data.rnn_hidden_states[i]
                            prev_actions[i].copy_(action_data.actions[i])  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # 处理连续动作空间：裁剪动作值到合法范围内
            if is_continuous_action_space(self._env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                # 离散动作空间：直接提取动作值
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            # 在环境中执行动作步骤
            outputs = self.envs.step(step_data)

            # 解包环境输出
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            # 获取策略额外信息并更新到infos中
            policy_infos = self._agent.actor_critic.get_extra(action_data, infos, dones)
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])
            # 将观察结果批处理并应用观察变换
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            # 更新未完成掩码
            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            # 累积奖励
            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            # 获取下一步的情节信息
            next_episodes_info = self.envs.current_episodes()
            # 记录需要暂停的环境索引
            envs_to_pause = []
            n_envs = self.envs.num_envs
            # 遍历所有环境，处理已完成的回合
            for i in range(n_envs):
                # 检查是否已达到每个情节的评估次数上限
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                # 特殊情节ID需要暂停
                elif int(next_episodes_info[i].episode_id) == 123123123:
                    envs_to_pause.append(i)

                # 如果启用了视频选项，收集可视化数据
                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    hab_vis.collect_data(batch, infos, action_data.policy_info)

                # 处理已完成的回合
                if not not_done_masks[i].item():
                    pbar.update()
                    # 收集回合统计数据
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # 使用scene_id + episode_id作为唯一ID存储统计信息
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    # 更新成功率统计
                    if episode_stats["success"] == 1:
                        num_successes += 1
                    num_total += 1
                    print(f"Success rate: {num_successes / num_total * 100:.2f}% ({num_successes} out of {num_total})")

                    # 记录回合统计信息
                    from vlfm.utils.episode_stats_logger import (
                        log_episode_stats,
                    )

                    try:
                        failure_cause = log_episode_stats(
                            current_episodes_info[i].episode_id,
                            current_episodes_info[i].scene_id,
                            infos[i],
                        )
                    except Exception:
                        failure_cause = "Unknown"

                    # 如果启用了视频选项，生成视频
                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        rgb_frames[i] = hab_vis.flush_frames(failure_cause)
                        generate_video(
                            video_option=self.config.habitat_baselines.eval.video_option,
                            video_dir=self.config.habitat_baselines.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes_info[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(infos[i]),
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        rgb_frames[i] = []

                    # 处理gfx回放数据
                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            # 将未完成掩码移到指定设备
            not_done_masks = not_done_masks.to(device=self.device)
            # 暂停已完成评估的环境
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()

        # 如果设置了完成路径，在评估完成后创建标记文件
        if "ZSOS_DONE_PATH" in os.environ:
            # Create an empty file at ZSOS_DONE_PATH to signal that the
            # evaluation is done
            done_path = os.environ["ZSOS_DONE_PATH"]
            with open(done_path, "w") as f:
                f.write("")

        # 验证评估的情节数量是否符合预期
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        # 计算聚合统计信息
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean([v[stat_key] for v in stats_episodes.values()])

        # 记录平均统计信息
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        # 获取检查点步数
        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        # 将奖励和指标写入Tensorboard
        writer.add_scalar("eval_reward/average_reward", aggregated_stats["reward"], step_id)

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        # 关闭环境
        self.envs.close()
