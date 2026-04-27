import torch
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from torch import Tensor
import torch.nn.functional as F


class Conv_1x1(ModelMixin, ConfigMixin):
    """
    将输入特征图的通道数从 input_dim 转换为 output_dim
    """

    @register_to_config
    def __init__(self,
                 input_dim: int = 8,
                 output_dim: int = 4,
                 use_active: bool = True):
        super().__init__()

        self.conv = torch.nn.Conv2d(input_dim, output_dim, kernel_size=1)
        if use_active:
            self.active = torch.nn.SiLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.conv(features)
        if hasattr(self, 'active'):
            x = self.active(x)
        return x
        

def extract_vae_supp_prototype(
        support_features,  # [B, C, h, w]
        mask: torch.Tensor  # [B, H, W]
) -> tuple[Tensor, Tensor]:
    """
    support_features: vae 的中间特征
    mask: 原始分辨率二进制掩码

    fg_prototypes: [B, C] 每个样本的前景原型
    bg_prototypes: [B, C] 每个样本的背景原型
    """
    B, C, h, w = support_features.shape

    mask_resized = F.interpolate(mask.float().unsqueeze(1), size=(h, w), mode='bilinear', align_corners=True).squeeze(1)  # [B, h, w]

    support_features_flat = support_features.view(B, C, -1)  # [B, C, h*w]
    mask_flat = mask_resized.view(B, -1)  # [B, h*w]

    fg_sum = torch.sum(support_features_flat * mask_flat.unsqueeze(1), dim=2)  # [B, C]
    fg_count = torch.sum(mask_flat, dim=1)  # [B], 每个样本的前景像素数
    fg_prototypes = fg_sum / (fg_count.unsqueeze(1) + 1e-6)  # [B, C]

    bg_mask_flat = 1.0 - mask_flat  # [B, h*w]
    bg_sum = torch.sum(support_features_flat * bg_mask_flat.unsqueeze(1), dim=2)  # [B, C]
    bg_count = torch.sum(bg_mask_flat, dim=1)  # [B], 每个样本的背景像素数
    bg_prototypes = bg_sum / (bg_count.unsqueeze(1) + 1e-6)  # [B, C]

    return fg_prototypes, bg_prototypes


def extract_vae_query_prototype(
        query_feature,  # [1, C, h, w]
        supp_fg_prototypes: Tensor,  # [B, C] 支持集前景原型
        supp_bg_prototypes: Tensor  # [B, C] 支持集背景原型
) -> tuple[Tensor, Tensor]:
    # 确保查询特征的批次大小为1
    assert query_feature.shape[0] == 1, "Query feature batch size should be 1"

    # 处理支持集原型：如果 B > 1，取平均值
    if supp_fg_prototypes.shape[0] > 1:
        supp_fg = supp_fg_prototypes.mean(dim=0, keepdim=True)  # [1, C]
        supp_bg = supp_bg_prototypes.mean(dim=0, keepdim=True)  # [1, C]
    else:
        supp_fg = supp_fg_prototypes  # [1, C]
        supp_bg = supp_bg_prototypes  # [1, C]

    # 扩展支持集原型为 [1, C, 1, 1] 以便广播
    supp_fg = supp_fg.unsqueeze(-1).unsqueeze(-1)  # [1, C, 1, 1]
    supp_bg = supp_bg.unsqueeze(-1).unsqueeze(-1)  # [1, C, 1, 1]

    # 计算查询特征与支持集原型的余弦相似度
    similarity_fg = F.cosine_similarity(query_feature, supp_fg, dim=1)  # [1, h, w]
    similarity_bg = F.cosine_similarity(query_feature, supp_bg, dim=1)  # [1, h, w]

    # 堆叠相似度图并应用 softmax 得到概率图
    out = torch.stack((similarity_bg, similarity_fg), dim=1)  # [1, 2, h, w]
    prob = F.softmax(out, dim=1)  # [1, 2, h, w]

    # 展平空间维度，便于掩码操作
    prob_flat = prob.view(1, 2, -1)  # [1, 2, N], N = h * w
    query_feature_flat = query_feature.view(1, query_feature.shape[1], -1)  # [1, C, N]

    # 定义高置信度阈值和 top-k 备选
    fg_thres = 0.7  # 前景阈值
    bg_thres = 0.6  # 背景阈值
    top_k = int(query_feature.shape[-1] * 0.1875)  # 若高置信度像素不足，取 top-k 个

    # 处理批次大小为1的情况，直接取 epi=0
    prob_fg = prob_flat[0, 1]  # [N] 前景概率
    prob_bg = prob_flat[0, 0]  # [N] 背景概率
    query_feats = query_feature_flat[0]  # [C, N]

    # 提取前景原型
    mask_fg = prob_fg > fg_thres
    if mask_fg.sum() > 0:
        fg_feats = query_feats[:, mask_fg]  # [C, num_fg_pixels]
    else:
        # 若无高置信度像素，取 top-k
        _, topk_indices = torch.topk(prob_fg, top_k)
        fg_feats = query_feats[:, topk_indices]  # [C, top_k]

    # 提取背景原型
    mask_bg = prob_bg > bg_thres
    if mask_bg.sum() > 0:
        bg_feats = query_feats[:, mask_bg]  # [C, num_bg_pixels]
    else:
        # 若无高置信度像素，取 top-k
        _, topk_indices = torch.topk(prob_bg, top_k)
        bg_feats = query_feats[:, topk_indices]  # [C, top_k]

    # 计算自支持原型
    query_fg_proto = fg_feats.mean(dim=1).unsqueeze(0)  # [1, C]
    query_bg_proto = bg_feats.mean(dim=1).unsqueeze(0)  # [1, C]

    return query_fg_proto, query_bg_proto
