# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size

from .resnet import resnet18
from .rnn_state_encoder import LSTMStateEncoder


class ResNetEncoder(nn.Module): 
    visual_keys = ["depth"] # 定义类属性，指定输入的视觉键为 "depth"，表示该编码器处理深度图像数据

    def __init__(self) -> None:
        super().__init__()
        self.running_mean_and_var = nn.Sequential() # 创建一个空的 Sequential 容器，用于可能的均值和方差归一化处理，但当前为空
        self.backbone = resnet18(1, 32, 16) # 创建 ResNet-18 主干网络
        self.compression = nn.Sequential(   # 创建压缩层序列
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.GroupNorm(1, 128, eps=1e-05, affine=True), # 分组归一化层 把128个通道分成一组进行归一化
            nn.ReLU(inplace=True),
        )
    # 前向传播函数，接收观测字典作为输入，返回张量
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        cnn_input = []
        for k in self.visual_keys:
            obs_k = observations[k]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH] 转换形状
            # 便于输入GroupNorm
            obs_k = obs_k.permute(0, 3, 1, 2)
            cnn_input.append(obs_k) # 加入输入张量列表
        
        x = torch.cat(cnn_input, dim=1)
        # 对输入进行 2x2 平均池化，将图像尺寸缩小一半
        x = F.avg_pool2d(x, 2)
        # 应用均值方差归一化
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class PointNavResNetNet(nn.Module):
    def __init__(self, discrete_actions: bool = False, no_fwd_dict: bool = False):  # discrete_actions: 是否使用离散动作空间 no_fwd_dict: 是否在前向传播中返回额外字典
        super().__init__()
        if discrete_actions: # 如果是离散动作空间
            self.prev_action_embedding_discrete = nn.Embedding(4 + 1, 32) # 使用嵌入层将离散动作（4种基本动作+1个起始标记）映射到32维向量
        else:
            self.prev_action_embedding_cont = nn.Linear(in_features=2, out_features=32, bias=True) # 连续动作：使用线性层将2维连续动作映射到32维向量
        
        # 创建目标位置嵌入层，将3维目标位置信息（距离、cos角度、sin角度）映射到32维向量。
        self.tgt_embeding = nn.Linear(in_features=3, out_features=32, bias=True)
        # 创建视觉编码器实例，用于处理深度图像
        self.visual_encoder = ResNetEncoder()
        self.visual_fc = nn.Sequential(  # 创建视觉特征处理序列
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.ReLU(inplace=True),
        )
        # 创建LSTM状态编码器，输入维度576，隐藏层维度512，2层LSTM
        self.state_encoder = LSTMStateEncoder(576, 512, 2)
        # 保存相关属性，供外部使用
        self.num_recurrent_layers = self.state_encoder.num_recurrent_layers
        self.discrete_actions = discrete_actions
        self.no_fwd_dict = no_fwd_dict

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        visual_feats = self.visual_encoder(observations) # 通过视觉编码器提取视觉特征
        visual_feats = self.visual_fc(visual_feats) # 通过视觉全连接层进一步处理
        x.append(visual_feats)  # 将处理后的视觉特征添加到特征列表

        goal_observations = observations["pointgoal_with_gps_compass"] # 从观测中提取目标位置信息（距离和角度）
        goal_observations = torch.stack(   # 重新组织为目标表示：[距离, cos(-角度), sin(-角度)]
            [
                goal_observations[:, 0],
                torch.cos(-goal_observations[:, 1]),
                torch.sin(-goal_observations[:, 1]),
            ],
            -1,
        )

        x.append(self.tgt_embeding(goal_observations))  # 将目标位置信息通过嵌入层处理并添加到特征列表

        # 处理历史动作信息
        # 离散和连续
        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1) # 去除多余的维度
            start_token = torch.zeros_like(prev_actions) # 创建与前一个动作相同形状的零张量，用作序列开始的特殊标记
            # The mask means the previous action will be zero, an extra dummy action
            # 将离散动作索引转换为密集向量表示
            # torch.where(condition, x, y)
            # .view(-1)：将张量重塑为一维
            prev_actions = self.prev_action_embedding_discrete(
                torch.where(masks.view(-1), prev_actions + 1, start_token) # 使用嵌入层处理
                # 当mask为1时（继续序列）：使用prv_actions + 1，加1是为了避免与起始标记（0）冲突，所以embedding层输入为5
                # 当mask为0时（新序列开始）：使用start_token（值为0）, 表示这是序列的开始
            )
        else:
            prev_actions = self.prev_action_embedding_cont(masks * prev_actions.float()) # 通过线性层处理，并应用掩码

        x.append(prev_actions) # 将处理后的历史动作信息添加到特征列表。

        out = torch.cat(x, dim=1) # 在特征维度上拼接所有特征
        # 通过LSTM状态编码器处理时序信息
        out, rnn_hidden_states = self.state_encoder(out, rnn_hidden_states, masks, rnn_build_seq_info)


        if self.no_fwd_dict:
            return out, rnn_hidden_states  # type: ignore
        # 返回处理后的特输出、RNN隐藏状态和额外的字典
        return out, rnn_hidden_states, {}
        # out: 处理后的特征张量，维度为 [batch_size, 512]
        # rnn_hidden_states: 更新后的RNN隐藏状态，[batch_size, num_recurrent_layers, 512]


class CustomNormal(torch.distributions.normal.Normal):
    def sample(self, sample_shape: Size = torch.Size()) -> torch.Tensor:
        return self.rsample(sample_shape)


class GaussianNet(nn.Module):
    min_log_std: int = -5
    max_log_std: int = 2
    log_std_init: float = 0.0

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()
        num_linear_outputs = 2 * num_outputs

        self.mu_maybe_std = nn.Linear(num_inputs, num_linear_outputs)
        nn.init.orthogonal_(self.mu_maybe_std.weight, gain=0.01)
        nn.init.constant_(self.mu_maybe_std.bias, 0)
        nn.init.constant_(self.mu_maybe_std.bias[num_outputs:], self.log_std_init)

    def forward(self, x: torch.Tensor) -> CustomNormal:
        mu_maybe_std = self.mu_maybe_std(x).float()
        mu, std = torch.chunk(mu_maybe_std, 2, -1)

        mu = torch.tanh(mu) #  tanh激活函数将输出限制在[-1,1]

        std = torch.clamp(std, self.min_log_std, self.max_log_std)
        std = torch.exp(std)

        return CustomNormal(mu, std, validate_args=False)


class PointNavResNetPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = PointNavResNetNet()
        self.action_distribution = GaussianNet(512, 2)

    def act(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, rnn_hidden_states, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mean
        else:
            action = distribution.sample()

        return action, rnn_hidden_states


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("state_dict_path", type=str, help="Path to state_dict file")
    args = parser.parse_args()

    ckpt = torch.load(args.state_dict_path, map_location="cpu")
    policy = PointNavResNetPolicy()
    print(policy)
    current_state_dict = policy.state_dict()
    policy.load_state_dict({k: v for k, v in ckpt.items() if k in current_state_dict})
    print("Loaded model from checkpoint successfully!")

    policy = policy.to(torch.device("cuda"))
    print("Successfully moved model to GPU!")

    observations = {
        "depth": torch.ones(1, 212, 240, 1, device=torch.device("cuda")),
        "pointgoal_with_gps_compass": torch.zeros(1, 2, device=torch.device("cuda")),
    }
    mask = torch.zeros(1, 1, device=torch.device("cuda"), dtype=torch.bool)

    rnn_state = torch.zeros(1, 4, 512, device=torch.device("cuda"), dtype=torch.float32)

    action = policy.act(
        observations,
        rnn_state,
        torch.zeros(1, 2, device=torch.device("cuda"), dtype=torch.float32),
        mask,
        deterministic=True,
    )

    print("Forward pass successful!")
    print(action[0].detach().cpu().numpy())
