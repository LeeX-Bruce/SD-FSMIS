from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureToConditioningMLP(ModelMixin, ConfigMixin):
    """
    将聚合后的视觉特征 (如 DINOv2 masked features) 映射到
    扩散模型的条件嵌入空间 (类似 text embedding)。
    """

    @register_to_config
    def __init__(self,
                 input_dim: int = 768,  # 输入特征维度 (例如 DINOv2 的 embed_dim, 768)
                 output_dim: int = 768,  # 输出条件嵌入维度 (需要匹配扩散模型的 U-Net cross-attention dim)
                 hidden_dim: int = 768 * 2,  # MLP 中间隐藏层维度
                 num_hidden_layers: int = 2,  # MLP 隐藏层数量
                 use_layernorm: bool = True):  # 是否使用 LayerNorm
        super().__init__()

        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2  # 如果未指定，取输入输出维度的平均

        activation = nn.GELU

        layers = []
        # Input Layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation())

        # Hidden Layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())

        # Output Layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): 聚合后的支持集特征, shape: (batch_size, input_dim)

        Returns:
            torch.Tensor: 映射后的条件嵌入, shape: (batch_size, 1, output_dim)
                          (增加了一个 sequence length 维度，以匹配 diffusers 的 cross-attention 输入)
        """
        conditioning_embedding = self.mlp(features)  # shape: (batch_size, output_dim)

        # 为 U-Net 的 cross-attention 增加 sequence length 维度 (通常为 1)
        conditioning_embedding = conditioning_embedding.unsqueeze(1)  # shape: (batch_size, 1, output_dim)

        return conditioning_embedding


def get_diffusion_conditioning(
        feat_s: torch.Tensor,  # 支持集特征 [nshot, 257, 768]
        mask_s: torch.Tensor,  # 支持集掩码 [nshot, 256, 256]
        mapper_mlp: FeatureToConditioningMLP,  # 实例化的 MLP 映射器
        patch_grid_size: int = 16  # DINOv2 输出的 patch 网格大小 (例如 16x16=256)
) -> torch.Tensor:
    """
    从 DINOv2 支持特征和掩码计算扩散模型的条件嵌入。

    Args:
        feat_s: DINOv2 输出的支持集特征 (包括 CLS token)。
        mask_s: 支持集掩码 (原始分辨率或接近原始分辨率)。
        mapper_mlp: 用于映射聚合特征的 MLP 模块。
        patch_grid_size: DINOv2 输出特征对应的网格边长 (例如 16 for 16x16=256 patches)。

    Returns:
        聚合后的条件嵌入, shape: (1, 1, mapper_mlp.output_dim)
        注意：假设这 nshot 个 support 对应一个 query (即 batch_size=1 的 few-shot 任务)
    """
    nshot, _, embed_dim = feat_s.shape
    device = feat_s.device
    dtype = feat_s.dtype  # 保持与输入特征相同的数据类型

    # 1. 分离 CLS token 和 Patch tokens
    # CLS token 通常在索引 0
    patch_feat_s = feat_s[:, 1:, :]  # Shape: [nshot, 256, 768]

    # 2. 将 Patch tokens 调整为空间形状
    # 确保 patch_grid_size * patch_grid_size == patch_feat_s.shape[1]
    num_patches = patch_feat_s.shape[1]
    if patch_grid_size * patch_grid_size != num_patches:
        raise ValueError(
            f"patch_grid_size ({patch_grid_size}) squared does not match number of patches ({num_patches})")
    # Reshape to [nshot, H', W', C] then permute to [nshot, C, H', W']
    patch_feat_s_spatial = patch_feat_s.reshape(
        nshot, patch_grid_size, patch_grid_size, embed_dim
    ).permute(0, 3, 1, 2).contiguous()  # Shape: [nshot, 768, 16, 16]

    # 3. 调整 Support Mask 大小以匹配 Patch 网格
    # 需要将 [nshot, 256, 256] -> [nshot, 1, 16, 16]
    # 使用 F.interpolate。需要 mask 是 float 类型，并添加 channel 维度。
    mask_s_resized = F.interpolate(
        mask_s.unsqueeze(1).float(),  # Add channel dim: [nshot, 1, 256, 256]
        size=(patch_grid_size, patch_grid_size),
        mode='bilinear',  # 或者 'nearest'，bilinear 可能更平滑
        align_corners=False
    )  # Shape: [nshot, 1, 16, 16]

    # 可选项：如果希望 mask 更接近二值，可以加一个阈值
    mask_s_resized = (mask_s_resized > 0.5).float()

    # 4. 执行 Masked Averaging
    # 逐元素相乘，然后在空间维度上求和，再除以 mask 的和
    masked_features = patch_feat_s_spatial * mask_s_resized  # Broadcasting mask: [nshot, 768, 16, 16]

    # Sum over spatial dimensions (H', W')
    summed_masked_features = masked_features.sum(dim=[2, 3])  # Shape: [nshot, 768]

    # Sum the mask values over spatial dimensions to get the normalization factor
    # 添加一个小的 epsilon 防止除以零
    mask_sum = mask_s_resized.sum(dim=[2, 3]) + 1e-6  # Shape: [nshot, 1]

    # Calculate the average feature vector for the foreground of each support image
    target_feature_vectors = summed_masked_features / mask_sum  # Shape: [nshot, 768]

    # 5. 聚合 Support Set 特征 (跨 nshot 维度)
    # 对所有 support 样本的特征向量求平均
    # 保持 batch 维度 (即使现在是 1)，以便与 MLP 输入兼容
    aggregated_support_features = torch.mean(
        target_feature_vectors, dim=0, keepdim=True
    )  # Shape: [1, 768]

    # 6. 通过 MLP 映射到条件嵌入空间
    # 确保数据类型匹配 MLP 的权重类型
    conditioning_embeds = mapper_mlp(aggregated_support_features.to(dtype=mapper_mlp.mlp[0].weight.dtype))
    # Output shape: [1, 1, output_dim]

    return conditioning_embeds
    