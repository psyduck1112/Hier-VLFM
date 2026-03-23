# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Tuple

import torch
from torch import Tensor


def image_resize(
    img: Tensor,
    size: Tuple[int, int],
    channels_last: bool = False,
    interpolation_mode: str = "area",
) -> torch.Tensor:
    """Resizes an img.
    确保输入是张量，并统一成 batch 格式 
    如果输入是 channels_last （HWC/NHWC 格式），就转成 channels_first，方便 PyTorch 处理
    用插值方法缩放到目标大小。
    如果需要，再把通道顺序转换回 channels_last
    如果原来是单张图片，去掉 batch 维度，保证输入输出形式一致
    Args:
        img: the array object that needs to be resized (HWC) or (NHWC)
        size: the size that you want
        channels: a boolean that channel is the last dimension
    Returns:
        The resized array as a torch tensor.
    """
    img = torch.as_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if no_batch_dim:
        img = img.unsqueeze(0)  # Adds a batch dimension
    if channels_last:
        if len(img.shape) == 4:
            # NHWC -> NCHW
            img = img.permute(0, 3, 1, 2)
        else:
            # NDHWC -> NDCHW
            img = img.permute(0, 1, 4, 2, 3)

    img = torch.nn.functional.interpolate(img.float(), size=size, mode=interpolation_mode).to(dtype=img.dtype)
    if channels_last:
        if len(img.shape) == 4:
            # NCHW -> NHWC
            img = img.permute(0, 2, 3, 1)
        else:
            # NDCHW -> NDHWC
            img = img.permute(0, 1, 3, 4, 2)
    if no_batch_dim:
        img = img.squeeze(dim=0)  # Removes the batch dimension
    return img
