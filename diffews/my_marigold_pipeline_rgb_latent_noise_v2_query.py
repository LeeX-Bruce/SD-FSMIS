# Author: Bingxin Ke
# Last modified: 2023-12-15

from typing import Dict, Union

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from PIL import Image

from diffusers import (
    DiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    ControlNetModel
)

from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer, AutoModel, AutoImageProcessor

from diffews.models.FeatureToConditioningMLP import FeatureToConditioningMLP, get_diffusion_conditioning
from diffews.models.proto import extract_vae_supp_prototype, extract_vae_query_prototype
from marigold.util.image_util import chw2hwc, colorize_depth_maps, resize_max_res, norm_to_rgb
from marigold.util.ensemble import ensemble_depths
from marigold.models import DPTHead, CustomUNet2DConditionModel

from marigold.util.scheduler_customized import DDIMSchedulerCustomized as DDIMScheduler


class MarigoldAlbedoShadingOutput(BaseOutput):

    albedo: np.ndarray
    shading: np.ndarray
    albedo_colored: Image.Image
    shading_colored: Image.Image
    uncertainty: Union[None, np.ndarray]

class MarigoldNormalOutput(BaseOutput):

    normal_np: np.ndarray
    normal_colored: Image.Image
    uncertainty: Union[None, np.ndarray]

class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (np.ndarray):
            Predicted depth map, with depth values in the range of [0, 1]
        depth_colored (PIL.Image.Image):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1]
        uncertainty (None` or `np.ndarray):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class MarigoldSegOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (np.ndarray):
            Predicted depth map, with depth values in the range of [0, 1]
        depth_colored (PIL.Image.Image):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1]
        uncertainty (None` or `np.ndarray):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    # seg_np: np.ndarray
    seg_colored: Image.Image
    uncertainty: Union[None, np.ndarray]

class MarigoldSrOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (np.ndarray):
            Predicted depth map, with depth values in the range of [0, 1]
        depth_colored (PIL.Image.Image):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1]
        uncertainty (None` or `np.ndarray):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    # sr_np: np.ndarray
    sr_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class MarigoldPipelineRGBLatentNoise(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://arxiv.org/abs/2312.02145.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (UNet2DConditionModel):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (AutoencoderKL):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (DDIMScheduler):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (CLIPTextModel):
            Text-encoder, for empty text embedding.
        tokenizer (CLIPTokenizer):
            CLIP tokenizer.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215
    seg_latent_scale_factor = 0.18215
    sr_latent_scale_factor = 0.18215
    normal_latent_scale_factor = 0.18215

    def __init__(
            self,
            unet: Union[UNet2DConditionModel, CustomUNet2DConditionModel],
            vae: AutoencoderKL,
            scheduler: DDIMScheduler,
            tokenizer: Union[CLIPTokenizer, None],
            text_embeds: Union[torch.Tensor, None],
            text_encoder: Union[CLIPTextModel, None],
            image_encoder: Union[AutoModel, None],
            image_projector: Union[FeatureToConditioningMLP, None],
            processor: Union[AutoImageProcessor, None],
            controlnet: Union[ControlNetModel, None],
            customized_head: Union[DPTHead, None],
    ):
        super().__init__()

        # 注册模块的字典构建
        register_dict = dict(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
        )

        # 使用图像编码器还是文本编码器(二选一)
        if image_encoder is not None:
            self.text_embed_flag = False
            self.vision_embed_flag = True
            register_dict['image_encoder'] = image_encoder
            register_dict['processor'] = processor
            register_dict['text_encoder'] = None
            register_dict['tokenizer'] = None
            register_dict['text_embeds'] = None
        elif text_encoder is not None:
            self.text_embed_flag = True
            self.vision_embed_flag = False
            register_dict['image_encoder'] = None
            register_dict['processor'] = None
            register_dict['text_encoder'] = text_encoder
            register_dict['tokenizer'] = tokenizer
            register_dict['text_embeds'] = text_embeds
            self.empty_text_embed = text_embeds
        else:
            raise ValueError

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.processor = processor

        # 是否使用controlnet
        if isinstance(controlnet, ControlNetModel):
            register_dict['controlnet'] = controlnet
        else:
            self.controlnet = None
            register_dict['controlnet'] = None

        # 是否使用customized_head
        if customized_head is None:
            self.customized_head = None
        else:
            self.customized_head = customized_head
        register_dict['customized_head'] = self.customized_head

        # 图像投影器
        register_dict['image_projector'] = image_projector
        self.image_projector = image_projector

        self.register_modules(**register_dict)


    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0  # 对输入值进行放大，以便后续计算中更好地分布于正弦和余弦函数中

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb


    @property
    def guidance_scale(self):
        return self._guidance_scale


    # def extract_supp_prototype(
    #         self,
    #         dinov2_output,  # odict_keys(['last_hidden_state', 'pooler_output', 'hidden_states'])
    #         mask: torch.Tensor  # [B, H, W] binary mask
    # ):
    #     """
    #     dinov2_output: DINOv2 output
    #     mask: 原始分辨率二进制掩码，需与输入图像尺寸一致
    #
    #     fg_prototypes: [B, 768] 每个样本的前景原型
    #     bg_prototypes: [B, 768] 每个样本的背景原型
    #     """
    #     # 提取支持集特征
    #     features = (dinov2_output.hidden_states[4] * 0.4 +
    #                 dinov2_output.hidden_states[8] * 0.3 +
    #                 dinov2_output.hidden_states[12] * 0.3)
    #     features = features[:, 1:, :]  # [B, 256, 768]
    #
    #     B, num_patches, dim = features.shape
    #     spatial_size = int(num_patches ** 0.5)  # 特征图尺寸 16x16
    #
    #     # 掩码对齐
    #     mask = mask.unsqueeze(1).float()  # [B, 1, H, W]
    #     mask_down = F.interpolate(mask,
    #                               size=(spatial_size, spatial_size),
    #                               mode='bilinear',
    #                               align_corners=False
    #                               ).squeeze(1)  # [B, 16, 16]
    #
    #     # 二值化处理
    #     fg_mask = (mask_down > 0.5).view(B, -1)  # [B, 256]
    #     bg_mask = ~fg_mask  # [B, 256]
    #
    #     # 计算原型
    #     fg_prototypes = []
    #     bg_prototypes = []
    #     for i in range(B):
    #         # 前景
    #         fg_mask_i = fg_mask[i]  # [256]
    #         fg_features_i = features[i, fg_mask_i, :]  # [N_fg, 768]
    #         if fg_features_i.size(0) > 0:
    #             fg_prototype_i = fg_features_i.mean(dim=0)  # [768]
    #         else:
    #             fg_prototype_i = torch.zeros(768).to(features.device)
    #         fg_prototypes.append(fg_prototype_i)
    #
    #         # 背景
    #         bg_mask_i = bg_mask[i]  # [256]
    #         bg_features_i = features[i, bg_mask_i, :]  # [N_bg, 768]
    #         if bg_features_i.size(0) > 0:
    #             bg_prototype_b = bg_features_i.mean(dim=0)  # [768]
    #         else:
    #             bg_prototype_b = torch.zeros(768).to(features.device)
    #         bg_prototypes.append(bg_prototype_b)
    #
    #     # 堆叠为[B, 768]
    #     fg_prototypes = torch.stack(fg_prototypes, dim=0)  # [B, 768]
    #     bg_prototypes = torch.stack(bg_prototypes, dim=0)  # [B, 768]
    #
    #     return fg_prototypes, bg_prototypes
    #
    #
    # def extract_query_prototype(
    #         self,
    #         dinov2_output,  # odict_keys(['last_hidden_state', 'pooler_output', 'hidden_states'])
    #         supp_fg_prototypes,
    #         supp_bg_prototypes
    # ):
    #     """
    #     dinov2_output: DINOv2 output
    #     supp_fg_prototypes: [B, 768] 支持集前景原型
    #     supp_bg_prototypes: [B, 768] 支持集背景原型
    #
    #     prototypes: [B, 768] 每个样本的前景原型
    #     """
    #     # 提取查询集特征
    #     features = (dinov2_output.hidden_states[4] * 0.4 +
    #                 dinov2_output.hidden_states[8] * 0.3 +
    #                 dinov2_output.hidden_states[12] * 0.3)
    #     assert features.shape[0] == 1, "Only one query"  # 一次只处理一个query
    #     features = features[0, 1:, :]  # [256, 768]
    #
    #     # 聚合support原型
    #     supp_fg_prototype = supp_fg_prototypes.mean(dim=0)  # [768]
    #     supp_bg_prototype = supp_bg_prototypes.mean(dim=0)  # [768]
    #
    #     # 自支持原型生成（SSP核心）
    #     def generate_self_support(feature, supp_proto, sim_threshold=0.7):
    #         # 计算初始相似度
    #         sim_map = F.cosine_similarity(feature, supp_proto, dim=-1)  # [256]
    #         confidence_mask = (sim_map > sim_map.quantile(sim_threshold))  # 取前20%高置信度区域
    #
    #         if confidence_mask.sum() > 0:
    #             self_proto = feature[confidence_mask].mean(dim=0)  # [768]
    #         else:
    #             self_proto = supp_proto  # 退化到支持原型
    #
    #         return self_proto
    #
    #     # 生成自支持前景原型
    #     self_fg = generate_self_support(features, supp_fg_prototype)
    #     # 生成自支持背景原型
    #     self_bg = generate_self_support(features, supp_bg_prototype, 0.5)
    #
    #     # # 原型融合（自适应加权）
    #     # final_fg = 0.6 * self_fg + 0.4 * supp_fg  # 自支持原型权重更高[6](@ref)
    #     # final_bg = supp_bg  # 保持原始背景原型
    #
    #     return self_fg, self_bg


    @torch.no_grad()
    def __call__(
        self,
        input_images: list[Union[Image.Image, torch.Tensor]],
        denoising_steps: int = 10,
        # num_inference_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        match_input_res: bool = True,
        batch_size: int = 0,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        mode: str = 'depth',
        seed = None,
    ) -> BaseOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (Image):
                Input RGB (or gray-scale) image.
            processing_res (int, optional):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
                Defaults to 768.
            match_input_res (bool, optional):
                Resize depth prediction to match input resolution.
                Only valid if `limit_input_res` is not None.
                Defaults to True.
            denoising_steps (int, optional):
                Number of diffusion denoising steps (DDIM) during inference.
                Defaults to 10.
            ensemble_size (int, optional):
                Number of predictions to be ensembled.
                Defaults to 10.
            batch_size (int, optional):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
                Defaults to 0.
            show_progress_bar (bool, optional):
                Display a progress bar of diffusion denoising.
                Defaults to True.
            color_map (str, optional):
                Colormap used to colorize the depth map.
                Defaults to "Spectral".
            ensemble_kwargs ()
        Returns:
            `MarigoldDepthOutput`
        """

        # time1 = time.time()

        device = self.device

        # 设置任务通道数
        if mode == 'depth':
            task_channel_num = 1
        elif mode == 'seg' or mode == 'semseg':
            task_channel_num = 3
        elif mode == 'sr':
            task_channel_num = 3
        elif mode == 'normal':
            task_channel_num = 3
        elif mode == 'feature':
            task_channel_num = 4
        else:
            raise ValueError

        if not match_input_res:
            assert (
                processing_res is not None
            ), "Value error: `resize_output_back` is only valid with "
        assert processing_res >= 0
        assert denoising_steps >= 1
        assert ensemble_size >= 1

        # ----------------- Image Preprocess -----------------
        input_rgbs_norm = []
        for input_image in input_images:
            rgb_norm = input_image.to(device)

            assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
            input_rgbs_norm.append(rgb_norm)  # [bs, 3, H, W]

        input_size = input_images[0].shape[2:]
        bs = batch_size
        if mode == 'seg' or mode == 'semseg':
            input_size = input_images[1].shape[-2:]     # Query的尺寸
            bs = input_images[1].shape[0]               # Query的batch size


        # ----------------- Predicting -----------------
        # 重复图像用于集成预测
        duplicated_rgb_ref = torch.stack([input_rgbs_norm[0]] * ensemble_size)  # [ensemble_size, n_shot, 3, H, W]
        duplicated_rgb_tag = torch.stack([input_rgbs_norm[1]] * ensemble_size)  # [ensemble_size, bs, 3, H, W]
        duplicated_gt_ref = torch.stack([input_rgbs_norm[2]] * ensemble_size)   # [ensemble_size, n_shot, 3, H, W]

        # 将第一个维度（ensemble_size）与批量大小合并
        duplicated_rgb_ref = duplicated_rgb_ref.view(-1, 3, duplicated_rgb_ref.shape[-2], duplicated_rgb_ref.shape[-1])  # [ensemble_size * n_shot, 3, H, W]
        duplicated_rgb_tag = duplicated_rgb_tag.view(-1, 3, duplicated_rgb_tag.shape[-2], duplicated_rgb_tag.shape[-1])  # [ensemble_size * bs, 3, H, W]
        duplicated_gt_ref = duplicated_gt_ref.view(-1, 3, duplicated_gt_ref.shape[-2], duplicated_gt_ref.shape[-1])      # [ensemble_size * n_shot, 3, H, W]

        # print('batch_size :', bs)

        # if show_progress_bar:
        #     iterable = tqdm(desc=" " * 2 + "Inference batches", leave=False)
        # else:
        #     iterable = duplicated_rgb_tag

        # 按批次进行推理Predict (batched)
        depth_pred_ls = []
        (batched_img_ref, batched_img_tag, batched_gt_ref) = (duplicated_rgb_ref, duplicated_rgb_tag, duplicated_gt_ref)

        depth_pred_raw = self.single_infer(
            rgb_in_ref=batched_img_ref,  # [ensemble_size * n_shot, 3, H, W]
            rgb_in_tag=batched_img_tag,  # [ensemble_size * bs, 3, H, W]
            gt_in_ref=batched_gt_ref,    # [ensemble_size * n_shot, 3, H, W]
            num_inference_steps=denoising_steps,
            show_pbar=show_progress_bar,
            mode=mode,
            seed=seed,
        )  # [ensemble_size * bs, 3, H, W], value in (0, 255)
        depth_pred_ls.append(depth_pred_raw.detach().clone())

        # 将预测结果拼接到一起
        depth_preds = torch.concat(depth_pred_ls, axis=0).squeeze()

        if mode != 'feature':
            depth_preds = depth_preds.view(ensemble_size, bs, task_channel_num, depth_preds.shape[-2], depth_preds.shape[-1]) # [ensemble_size, bs, task_channel_num, H, W]
        else:
            depth_preds = depth_preds.view(ensemble_size, bs, denoising_steps, task_channel_num, depth_preds.shape[-2], depth_preds.shape[-1]) # [ensemble_size, bs, steps, task_channel_num, H, W]


        # ----------------- Test-time ensembling -----------------
        if mode == 'depth':
            if ensemble_size > 1:
                depth_pred_list = []
                pred_uncert_list = []
                for i in range(bs):
                    depth_pred_i, pred_uncert_i = ensemble_depths(
                        depth_preds[:, i, 0], **(ensemble_kwargs or {})
                    )
                    depth_pred_list.append(depth_pred_i)
                    pred_uncert_list.append(pred_uncert_i)
                depth_preds = torch.stack(depth_pred_list, dim=0)[:, None] # [bs, task_channel_num, H, W]
                pred_uncert = torch.stack(pred_uncert_list, dim=0)[:, None].squeeze()
            else:
                depth_preds = depth_preds.mean(dim=0) # [bs, task_channel_num, H, W]
                pred_uncert = None
        else:
            depth_preds = depth_preds.mean(dim=0) # [bs, task_channel_num, H, W] or [bs, steps, task_channel_num, H, W]
        
        if match_input_res:
            if mode == 'depth' or mode == 'normal':
                depth_preds = F.interpolate(depth_preds, input_size, mode='bilinear')
            elif mode == 'seg' or 'semseg':
                depth_preds = F.interpolate(depth_preds, input_size, mode='nearest')
            elif mode == 'sr':
                depth_preds = F.interpolate(depth_preds, input_size, mode='nearest')
            elif mode == 'feature':
                pass
            else:
                raise NotImplementedError

        # ----------------- Post processing -----------------
        if mode == 'depth':
            depth_preds = depth_preds[:, 0] # [bs, H, W]
            # Scale prediction to [0, 1]
            min_d = depth_preds.min(dim=-1)[0].min(dim=-1)[0]
            max_d = depth_preds.max(dim=-1)[0].max(dim=-1)[0]
            depth_preds = (depth_preds - min_d[:, None, None]) / (max_d[:, None, None] - min_d[:, None, None])

            # Convert to numpy
            depth_preds = depth_preds.cpu().numpy().astype(np.float32)

            # Clip output range
            depth_preds = depth_preds.clip(0, 1)

            # Colorize
            depth_colored_img_list = []
            for i in range(depth_preds.shape[0]):
                depth_colored_i = colorize_depth_maps(
                    depth_preds[i], 0, 1, cmap=color_map
                ).squeeze()  # [3, H, W], value in (0, 1)
                depth_colored_i = (depth_colored_i * 255).astype(np.uint8)
                depth_colored_i_hwc = chw2hwc(depth_colored_i)
                depth_colored_i_image = Image.fromarray(depth_colored_i_hwc)
                depth_colored_img_list.append(depth_colored_i_image)

            return MarigoldDepthOutput(
                depth_np=np.squeeze(depth_preds),
                depth_colored=depth_colored_img_list[0] if len(depth_colored_img_list) == 1 else depth_colored_img_list,
                uncertainty=pred_uncert,
            )

        elif mode == 'seg' or mode == 'semseg':
            # 限制图像数值范围
            seg_colored = depth_preds.clip(0, 255).cpu().numpy().astype(np.uint8)

            # 将张量转换为 PIL 图像
            seg_colored_img_list = []
            for i in range(seg_colored.shape[0]):
                seg_colored_hwc_i = chw2hwc(seg_colored[i])
                seg_colored_img_i = Image.fromarray(seg_colored_hwc_i).resize((input_size[1], input_size[0]))
                seg_colored_img_list.append(seg_colored_img_i)

            return MarigoldSegOutput(
                # 如果只有一幅图像，则直接返回该图像；如果有多幅图像，则返回整个列表
                seg_colored=seg_colored_img_list[0] if len(seg_colored_img_list) == 1 else seg_colored_img_list,
                uncertainty=None,
            )
        
        elif mode == 'sr':
            # Clip output range
            sr_colored = depth_preds.clip(0, 255).cpu().numpy().astype(np.uint8)

            sr_colored_img_list = []
            for i in range(sr_colored.shape[0]):
                sr_colored_hwc_i = chw2hwc(sr_colored[i])
                sr_colored_img_i = Image.fromarray(sr_colored_hwc_i).resize((input_size[1], input_size[0]))
                sr_colored_img_list.append(sr_colored_img_i)

            return MarigoldSrOutput(
                sr_colored=sr_colored_img_list[0] if len(sr_colored_img_list) == 1 else sr_colored_img_list,
                uncertainty=None,
            )

        elif mode == 'normal':
            normal = depth_preds.clip(-1, 1).cpu().numpy() # [-1, 1]

            normal_colored_img_list = []
            for i in range(normal.shape[0]):
                normal_colored_i = norm_to_rgb(normal[i])
                normal_colored_hwc_i = chw2hwc(normal_colored_i)
                normal_colored_img_i = Image.fromarray(normal_colored_hwc_i).resize((input_size[1], input_size[0]))
                normal_colored_img_list.append(normal_colored_img_i)

            return MarigoldNormalOutput(
                normal_np=np.squeeze(normal),
                normal_colored=normal_colored_img_list[0] if len(normal_colored_img_list) == 1 else normal_colored_img_list,
                uncertainty=None,
            )
        
        elif mode == 'feature':
            # depth_preds: [B, steps, 4, H/8, W/8] for when mode is 'feature'
            return np.squeeze(depth_preds.detach().cpu().numpy())

        else:
            raise NotImplementedError
    
    def encode_hidden_states(self, rgb_in_ref=None, rgb_in_tag=None, gt_in_ref=None):
        """
        Encode text embedding for empty prompt
        """
        # 使用空字符串作为prompt生成嵌入
        if self.text_encoder is not None:
            prompt = ""
            text_inputs = self.tokenizer(
                prompt,
                padding="do_not_pad",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
            text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)
            return text_embed

        # 使用图像编码器生成嵌入
        if self.image_encoder is not None and rgb_in_ref is not None and gt_in_ref is not None:
            # 处理输入图像
            imgs_ref = (rgb_in_ref + 1) / 2  # -> [0, 1] [n_shot, 3, H, W]
            cond_images_ref = (gt_in_ref[:, 0, :, :] + 1) / 2  # -> [0, 1] [n_shot, H, W]

            # 将Support images用dinov2编码
            inputs_ref = self.processor(images=imgs_ref, return_tensors="pt", do_rescale=False)
            inputs_ref = inputs_ref.to(device=self.unet.device, dtype=self.dtype)  # [n_shot, 3, 224, 224]
            imgs_ref_output = self.image_encoder(output_hidden_states=True, **inputs_ref).last_hidden_state
            conditioning_embeds = get_diffusion_conditioning(imgs_ref_output,
                                                             cond_images_ref,
                                                             self.image_projector)  # [1, 1, 768]

            return conditioning_embeds

    @torch.no_grad()
    def single_infer(
        self, 
        rgb_in_ref: torch.Tensor,
        rgb_in_tag: torch.Tensor,
        gt_in_ref: torch.Tensor,
        num_inference_steps: int, 
        show_pbar: bool,
        mode: str = 'depth',
        seed = None,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (torch.Tensor):
                Input RGB image.
            num_inference_steps (int):
                Number of diffusion denoising steps (DDIM) during inference.
            show_pbar (bool):
                Display a progress bar of diffusion denoising.

        Returns:
            torch.Tensor: Predicted depth map.
        """
        #  初始化并设置时间步
        device = rgb_in_tag.device
        self.scheduler.set_timesteps(num_inference_steps, device=device)  # num_inference_steps = 1
        timesteps = self.scheduler.timesteps  # [T], 1步时, 为: tensor([1])

        # 编码图像
        rgb_latent_ref = self.encode_rgb(rgb_in_ref)  # [n-shot, 4, 64, 64]
        rgb_latent_tag = self.encode_rgb(rgb_in_tag)  # [bs, 4, 64, 64]
        gt_latent_ref= self.encode_rgb(gt_in_ref)     # [n-shot, 4, 64, 64]
        

        # 设置随机种子
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        # 将Support构成为条件输入，用于指导Query的去噪过程
        latents_rgb_cond_ref = torch.cat([rgb_latent_ref, gt_latent_ref], dim=1)  # [n-shot, 8, 64, 64]
        depth_latent = rgb_latent_tag.clone()  # [bs, 4, 64, 64]

        # 指导嵌入生成
        if self.text_embed_flag:
            if self.empty_text_embed is not None:
                batch_embed = self.empty_text_embed  # [1, 2, 1024]
            else:
                batch_embed = self.encode_hidden_states()
            batch_embed_ref = batch_embed.repeat(rgb_in_ref.shape[0], 1, 1).to(device)  # [n_shot, 16, 768]
            batch_embed = batch_embed.repeat(rgb_in_tag.shape[0], 1, 1).to(device)  # [bs, 16, 768]
        elif self.vision_embed_flag:
            batch_embed = self.encode_hidden_states(rgb_in_ref, rgb_in_tag, gt_in_ref)
            batch_embed_ref = batch_embed.repeat(rgb_in_ref.shape[0], 1, 1).to(device)  # [n_shot, 1, 768]
            batch_embed = batch_embed.repeat(rgb_in_tag.shape[0], 1, 1).to(device)  # [bs, 1, 768]
        else:
            raise ValueError

        # 设置是否显示进度条
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        # 循环去噪
        scheduler_z0_list = []
        for i, t in iterable:
            if self.controlnet is None:
                # 将Query作为unet的输入
                unet_input = depth_latent  # query image latent

                # 获得support原型
                support_fg_proto, support_bg_proto = extract_vae_supp_prototype(rgb_latent_ref,
                                                                                (gt_in_ref[:, 0, :, :] + 1) / 2)  # [n-shot, C]

                # 获取query原型
                query_fg_protos = []
                for j in range(rgb_latent_tag.shape[0]):
                    query_fg_proto, _ = extract_vae_query_prototype(rgb_latent_tag[[j]], support_fg_proto,
                                                                    support_bg_proto)  # [1, 4]
                    query_fg_protos.append(query_fg_proto)  # [bs, 4]
                query_fg_protos = torch.cat(query_fg_protos, dim=0)

                unet_input = torch.cat([unet_input,
                                        query_fg_protos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, unet_input.shape[2],
                                                                                           unet_input.shape[3])],
                                       dim=1)  # [bs, 2C, H, W]

                self.unet.clear_attn_bank()

                # 对Support进行一次前向传播，用于更新 UNet 内部状态
                self.unet(latents_rgb_cond_ref, (t * self.test_timestep).repeat(rgb_in_ref.shape[0]),
                          encoder_hidden_states=batch_embed_ref,
                          is_target=False)  # latents_rgb_cond_ref: [n-shot, 8, 32, 32]



                noise_pred_list = []

                # unet_input: [bs, 8, 32, 32]
                for b in range(rgb_in_tag.shape[0]):  # bs 次循环
                    # 取出当前这个 query 的输入
                    unet_input_b = unet_input[b:b+1]           # [1, 8, 32, 32]
                    embed_b      = batch_embed[b:b+1]          # [1, 1, 768]

                    unet_output_b = self.unet(
                        unet_input_b,
                        t * self.test_timestep,
                        encoder_hidden_states=embed_b,
                        is_target=True                     # 使用已填充的 KV bank
                    )

                    noise_pred_b = unet_output_b.sample        # [1, 4, 64, 64]
                    noise_pred_list.append(noise_pred_b)

                # 合并回 batch 维度
                noise_pred = torch.cat(noise_pred_list, dim=0)  # [bs, 4, 64, 64]



                
                ## 对Query进行一次前向传播，生成分割预测
                #unet_output = self.unet(
                #    unet_input, (t * self.test_timestep).repeat(rgb_in_tag.shape[0]), encoder_hidden_states=batch_embed,
                #    is_target=True
                #)
                #noise_pred = unet_output.sample  # [B, 4, 64, 64]

                self.unet.clear_attn_bank()

                if self.customized_head:
                    assert isinstance(self.unet, CustomUNet2DConditionModel) or isinstance(self.unet, UNet2DConditionModel)
                    if self.customized_head.in_channels == 4:
                        unet_feature = unet_output.sample
                    elif self.customized_head.in_channels == 320:
                        unet_feature = unet_output.sample_320
                    else:
                        raise ValueError
            else:
                raise NotImplementedError

            # 根据预测噪声计算上一步的 latent, x_t -> x_t-1
            step_output = self.scheduler.step(noise_pred, t, depth_latent)
            depth_latent = step_output.prev_sample  # [B, 4, 64, 64]  # TODO delete?
            if mode == 'feature':
                scheduler_z0_list.append(step_output.pred_original_sample)

        # 预测含噪声图像的原始图像
        depth_latent = step_output.pred_original_sample

        depth_latent.to(self.vae.device)
        if mode == 'depth':
            if self.customized_head:
                depth = self.customized_head(unet_feature)
                depth = (depth - depth.min()) / (depth.max() - depth.min())
            else:
                depth = self.decode_depth(depth_latent)
                # clip prediction
                depth = torch.clip(depth, -1.0, 1.0)
                # shift to [0, 1]
                depth = (depth * 0.5) + 0.5
            return depth
        elif mode == 'seg' or 'semseg':
            if self.customized_head:
                raise NotImplementedError

            # 将 latent 解码为分割图
            seg = self.decode_seg(depth_latent)    # [B, 3, H, W]

            # 剪裁到[-1, 1]
            seg = torch.clip(seg, -1.0, 1.0)
            # 将数值映射到 [0, 1]
            seg = (seg + 1.0) / 2.0
            # seg = (seg * 0.5) + 0.5
            # 将数值映射到 [0, 255]
            seg = seg * 255

            return seg
        elif mode == 'sr':
            if self.customized_head:
                raise NotImplementedError
            sr = self.decode_sr(depth_latent)
            # clip prediction
            # sr = sr.mean(dim=0) # NOTE: average ensemble after single_infer.
            sr = torch.clip(sr, -1.0, 1.0)
            # # shift to [0, 1]
            # sr = (sr + 1.0) / 2.0 
            sr = (sr * 0.5) + 0.5
            # # shift to [0, 255]
            sr = sr * 255

            # import pdb
            # pdb.set_trace()
            # output_type = "pil"
            # image = self.image_processor.postprocess(sr, output_type=output_type)

            return sr
        elif mode == 'normal':
            if self.customized_head:
                raise NotImplementedError
            normal = self.decode_normal(depth_latent)
            normal = torch.clip(normal, -1.0, 1.0)
            return normal
        elif mode == 'feature':
            # import pdb
            # pdb.set_trace()
            z0_features = torch.stack(scheduler_z0_list, dim=1)
            return z0_features # [B, steps, 4, H/8, W/8]
        else:
            raise NotImplementedError


    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (torch.Tensor):
                Input RGB image to be encoded.

        Returns:
            torch.Tensor: Image latent
        """
        try:
            # encode
            h_temp = self.vae.encoder(rgb_in)
            moments = self.vae.quant_conv(h_temp)
        except:
            # encode
            h_temp = self.vae.encoder(rgb_in.float())
            moments = self.vae.quant_conv(h_temp.float())
            
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    
    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (torch.Tensor):
                Depth latent to be decoded.

        Returns:
            torch.Tensor: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)

        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean


    def decode_seg(self, seg_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (torch.Tensor):
                Depth latent to be decoded.

        Returns:
            torch.Tensor: Decoded depth map.
        """
        # scale latent
        seg_latent = seg_latent / self.seg_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(seg_latent)
        seg = self.vae.decoder(z)
        seg = seg.clip(-1, 1)

        return seg
    
    def decode_sr(self, sr_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (torch.Tensor):
                Depth latent to be decoded.

        Returns:
            torch.Tensor: Decoded depth map.
        """
        # scale latent
        sr_latent = sr_latent / self.sr_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(sr_latent)
        sr = self.vae.decoder(z)
        sr = sr.clip(-1, 1)

        return sr
    
    def decode_normal(self, normal_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (torch.Tensor):
                Depth latent to be decoded.

        Returns:
            torch.Tensor: Decoded depth map.
        """
        # scale latent
        normal_latent = normal_latent / self.normal_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(normal_latent)
        seg = self.vae.decoder(z)

        return seg
