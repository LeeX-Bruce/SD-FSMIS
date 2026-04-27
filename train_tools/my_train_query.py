#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoModel, AutoImageProcessor
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available

from PIL import Image
import os.path as osp
import torch.nn.functional as F

import sys
sys.path.append('./')

from evaluation_util.data.Medical import TestDataset
from evaluation_util.data.dataset import FSSDataset

from diffews.models.FeatureToConditioningMLP import FeatureToConditioningMLP, get_diffusion_conditioning
from diffews.models.proto import extract_vae_supp_prototype, extract_vae_query_prototype
from diffews.models.unet_2d_condition_v2 import MyUNet2DConditionModel as UNet2DConditionModel
from diffews.my_marigold_pipeline_rgb_latent_noise_v2_query import MarigoldPipelineRGBLatentNoise

from marigold.util.scheduler_customized import DDIMSchedulerCustomized as DDIMScheduler


if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.23.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# 加载用于训练图中可视化的数据
test_datasets = []

def log_validation(vae, scheduler, unet, dinov2, processor, image_proj_model, args, accelerator, weight_dtype, step,
                   test_datasets):
    logger.info("Running validation... ")

    pipeline = MarigoldPipelineRGBLatentNoise.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(unet),
        scheduler=scheduler,
        image_encoder=accelerator.unwrap_model(dinov2),
        image_projector=accelerator.unwrap_model(image_proj_model),
        processor=processor,
        text_encoder=None,
        tokenizer=None,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        text_embeds=None,
        controlnet=None,
        customized_head=None,
    )
    pipeline = accelerator.prepare(pipeline)
    pipeline = pipeline.to(accelerator.device)
    pipeline.test_timestep = 1
    pipeline.vae.to(torch.float32)
    pipeline.image_encoder.to(torch.float32)
    pipeline.image_projector.to(torch.float32)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs_folder = osp.join(args.output_dir, 'image_logs')
    os.makedirs(image_logs_folder, exist_ok=True)

    images = []

    eval_results_all = dict()

    for i, test_data in enumerate(test_datasets):
        validation_prompt = ""
        ref_rgb = test_data['ref_img']
        tag_rgb = test_data['tag_img']
        ref_gt = test_data['ref_cond'].repeat(1, 3, 1, 1) * 2 - 1
        tag_gt = test_data['tag_cond'].repeat(1, 3, 1, 1) * 2 - 1

        validation_image = [
            ref_rgb,  #
            tag_rgb,
            ref_gt
        ]

        if 'semseg' in test_data['style']:
            args.mode = 'seg'
        elif 'depth' in test_data['style']:
            args.mode = 'depth'
        elif 'normal' in test_data['style']:
            args.mode = 'normal'
        else:
            raise NotImplementedError

        with torch.autocast("cuda"):
            image = pipeline(
                validation_image,
                ensemble_size=1,
                processing_res=args.resolution,
                match_input_res=True,
                batch_size=1,
                color_map="Spectral",
                show_progress_bar=True,
                mode=args.mode,
                seed=42,
                denoising_steps=1,
            )

            if args.mode == 'depth':
                image = image.depth_colored
            elif args.mode == 'normal':
                image = image.normal_colored
            elif args.mode == 'seg':
                image = image.seg_colored
            else:
                raise ValueError
        image = transforms.functional.to_tensor(image)  # [3, H, W]

        images.append(image)  # [b, 3, H, W]

        # image_out = Image.fromarray(np.concatenate([np.array(tag_rgb), np.array(image)], axis=1))
        image_np = image.permute(1, 2, 0).cpu().numpy()
        tag_rgb_np = tag_rgb[0].permute(1, 2, 0).cpu().numpy()
        ref_rgb_np = ref_rgb[0].permute(1, 2, 0).cpu().numpy()
        ref_gt_np = ref_gt[0].permute(1, 2, 0).cpu().numpy()
        tag_gt_np = tag_gt[0].permute(1, 2, 0).cpu().numpy()

        image_np_uint8 = (image_np * 255).astype(np.uint8)
        tag_rgb_np_uint8 = ((tag_rgb_np + 1) / 2 * 255).astype(np.uint8)
        ref_rgb_np_uint8 = ((ref_rgb_np + 1) / 2 * 255).astype(np.uint8)
        ref_gt_np_uint8 = ((ref_gt_np + 1) / 2 * 255).astype(np.uint8)
        tag_gt_np_uint8 = ((tag_gt_np + 1) / 2 * 255).astype(np.uint8)

        if i == 0:
            image_out_np = np.concatenate(
                [ref_rgb_np_uint8, ref_gt_np_uint8, tag_rgb_np_uint8, tag_gt_np_uint8, image_np_uint8], axis=1)
        else:
            image_out_i = np.concatenate(
                [ref_rgb_np_uint8, ref_gt_np_uint8, tag_rgb_np_uint8, tag_gt_np_uint8, image_np_uint8], axis=1)
            image_out_np = np.concatenate([image_out_np, image_out_i], axis=0)

    save_path = osp.join(image_logs_folder, f'{args.mode}')
    os.makedirs(save_path, exist_ok=True)
    image_out = Image.fromarray(image_out_np)
    image_out.save(osp.join(save_path, 'step_{}.jpg'.format(step)))

    pipeline.vae.to(weight_dtype)
    pipeline.image_encoder.to(weight_dtype)
    pipeline.image_projector.to(weight_dtype)
    del pipeline
    torch.cuda.empty_cache()

    return eval_results_all


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="coco",
        help="benchmark."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['depth', 'normal', 'seg'],
        default="seg",
        help="inference mode.",
    )
    parser.add_argument(
        "--input_perturbation",
        type=float,
        default=0,
        help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models/stable-diffusion-2-1-ref8inchannels-tag4inchannels",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--pretrained_dinov2_path",
        type=str,
        default="facebook/dinov2-base",
        help="Path to pretrained dinov2 model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--scheduler_load_path",
        type=str,
        default='./scheduler_1.0_1.0',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--gt_data_root",
        type=str,
        default='/test/xugk/data/data_metricdepth',
        required=False,
        help="gt data root",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="./datasets/hypersim_icl/multitaskv1",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_ref_column", type=str, default="img_ref", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--image_tag_column", type=str, default="img_tag", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--conditioning_image_ref_column",
        type=str,
        default="ref_conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--conditioning_image_tag_column",
        type=str,
        default="tag_conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        nargs="+",
        help=(
            "Which dataset to visualize, including 'CHAOST2' and 'SABS'."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs/debug",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='./cache',
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=30000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="polynomial",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_scheduler_power", type=float, default=1.0, help="Lr scheduler power."
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd21_train_dis",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    # add train_time_steps
    parser.add_argument(
        "--train_timestep",
        type=int,
        default=1,
        help="timesteps for training",
    )
    parser.add_argument(
        "--nshot",
        type=int,
        default=1,
        help="number of shots for training",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="number of shots for validation",
    )
    parser.add_argument(
        "--supp_idx",
        type=int,
        default=2,
        help="support index for validation",
    )
    parser.add_argument(
        "--r_threshold",
        type=float,
        default=0.25,
        help="r_threshold."
    )
    parser.add_argument(
        "--setting",
        type=int,
        default=1,
        help="training setting"
    )
    parser.add_argument(
        "--test_label",
        type=int,
        default=1,
        help="Only effective when training with GT labels"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info(f"loading test_datasets")
        data_config = {
            'dataname': args.validation_images[0],
            'datapath': args.train_data_dir,
            'eval_fold': args.fold,
            'supp_idx': args.supp_idx,
            'img_size': args.resolution,
            'use_original_imgsize': False
        }
        test_dataset = TestDataset(**data_config)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        n_slice = 1
        # 0 = 'BG', 1 = 'LIVER', 2 = 'RK', 3 = 'LK', 4 = 'SPLEEN'
        label_val = 1
        test_dataset.label = label_val
        support_batch = test_dataset.getSupport(label=label_val, all_slices=False, N=n_slice)
        query_batch = next(iter(test_loader))
        test_datasets.append({
            'ref_img': support_batch['support_imgs'],
            'tag_img': query_batch['query_img'][:, int(query_batch['query_img'].shape[1] / 2)],
            'ref_cond': support_batch['support_masks'],
            'tag_cond': query_batch['query_mask'][:, int(query_batch['query_img'].shape[1] / 2)],
            'style': 'semseg'
        })
        label_val = 2
        test_dataset.label = label_val
        support_batch = test_dataset.getSupport(label=label_val, all_slices=False, N=n_slice)
        query_batch = next(iter(test_loader))
        test_datasets.append({
            'ref_img': support_batch['support_imgs'],
            'tag_img': query_batch['query_img'][:, int(query_batch['query_img'].shape[1] / 2)],
            'ref_cond': support_batch['support_masks'],
            'tag_cond': query_batch['query_mask'][:, int(query_batch['query_img'].shape[1] / 2)],
            'style': 'semseg'
        })
        label_val = 3
        test_dataset.label = label_val
        support_batch = test_dataset.getSupport(label=label_val, all_slices=False, N=n_slice)
        query_batch = next(iter(test_loader))
        test_datasets.append({
            'ref_img': support_batch['support_imgs'],
            'tag_img': query_batch['query_img'][:, int(query_batch['query_img'].shape[1] / 2)],
            'ref_cond': support_batch['support_masks'],
            'tag_cond': query_batch['query_mask'][:, int(query_batch['query_img'].shape[1] / 2)],
            'style': 'semseg'
        })
        if args.validation_images[0] != "CMR":
            label_val = 4 if args.validation_images[0] == "CHAOST2" else 6
            test_dataset.label = label_val
            support_batch = test_dataset.getSupport(label=label_val, all_slices=False, N=n_slice)
            query_batch = next(iter(test_loader))
            test_datasets.append({
                'ref_img': support_batch['support_imgs'],
                'tag_img': query_batch['query_img'][:, int(query_batch['query_img'].shape[1] / 2)],
                'ref_cond': support_batch['support_masks'],
                'tag_cond': query_batch['query_mask'][:, int(query_batch['query_img'].shape[1] / 2)],
                'style': 'semseg'
            })
        del test_dataset
        del test_loader
        del support_batch
        del query_batch
        logger.info(f"test_datasets loaded")

    if args.scheduler_load_path is not None:
        # noise_scheduler = DDPMScheduler.from_pretrained(args.scheduler_load_path, subfolder="scheduler")
        noise_scheduler_ddim = DDIMScheduler.from_pretrained(args.scheduler_load_path, subfolder="scheduler")
    else:
        # noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        noise_scheduler_ddim = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_ddim.set_timesteps(args.train_timestep)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        dinov2 = AutoModel.from_pretrained(args.pretrained_dinov2_path)
        processor = AutoImageProcessor.from_pretrained(args.pretrained_dinov2_path)
        if args.pretrained_vae_path is not None:
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_vae_path, subfolder="vae", revision=args.revision, variant=args.variant
            )
        else:
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
            )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )
    output_dim = unet.config.cross_attention_dim
    image_proj_model = FeatureToConditioningMLP(
        input_dim=768,
        output_dim=output_dim,
        hidden_dim=768 * 2,
        num_hidden_layers=2
    )


    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    dinov2.requires_grad_(False)
    unet.train()
    image_proj_model.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        output_dim = ema_unet.config.cross_attention_dim
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
        ema_image_proj_model = FeatureToConditioningMLP(
            input_dim=768,
            output_dim=output_dim,
            hidden_dim=768 * 2,
            num_hidden_layers=2
        )
        ema_image_proj_model = EMAModel(
            ema_image_proj_model.parameters(),
            model_cls=FeatureToConditioningMLP,
            model_config=ema_image_proj_model.config
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                    ema_image_proj_model.save_pretrained(os.path.join(output_dir, "image_projector_ema"))

                for i, model in enumerate(models):
                    if isinstance(model, UNet2DConditionModel):
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    elif isinstance(model, FeatureToConditioningMLP):  # 假设是自定义图像编码器
                        model.save_pretrained(os.path.join(output_dir, "image_projector"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

                load_ema_image_proj_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "image_projector_ema"), FeatureToConditioningMLP
                )
                ema_image_proj_model.load_state_dict(load_ema_image_proj_model.state_dict())
                ema_image_proj_model.to(accelerator.device)
                del load_ema_image_proj_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # 根据模型类型加载普通模型
                if isinstance(model, UNet2DConditionModel):
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                elif isinstance(model, FeatureToConditioningMLP):
                    load_model = FeatureToConditioningMLP.from_pretrained(input_dir, subfolder="image_projector")

                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    parameters = [
        {"params": unet.parameters(), "lr": args.learning_rate},
        {"params": image_proj_model.parameters(), "lr": 5 * args.learning_rate},
    ]
    optimizer = optimizer_cls(
        parameters,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 准备数据集和DataLoader
    args.datapath = args.train_data_dir
    args.use_original_imgsize = False
    args.bsz = args.train_batch_size  # 1
    args.nworker = args.dataloader_num_workers

    FSSDataset.initialize(img_size=args.resolution, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_train_nshot = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn',
                                                         args.nshot, args.validation_images[0], args.setting, test_label=args.test_label)

    eval_results_all = {}
    train_dataloader = dataloader_train_nshot
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        power=args.lr_scheduler_power,
    )

    # Prepare everything with our `accelerator`.
    unet, image_proj_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, image_proj_model, optimizer, train_dataloader, lr_scheduler
    )

    # for i in range(len(test_dataloader_list)):
    #     test_dataloader_list[i] = accelerator.prepare(test_dataloader_list[i])

    if args.use_ema:
        ema_unet.to(accelerator.device)
        ema_image_proj_model.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    dinov2.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_images")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Learning Rate = {args.learning_rate}")
    logger.info(f"  Training timesteps = {args.train_timestep}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if os.path.exists(args.resume_from_checkpoint):
                accelerator.load_state(args.resume_from_checkpoint, map_location="cpu")
            else:
                accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # # 加载预先经过text_encoder编码的空文本的token
    # temp_input_ids = torch.load('temp_input_ids.pt').to(accelerator.device)

    # 训练
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch_ in enumerate(dataloader_train_nshot):
            with (accelerator.accumulate([unet, image_proj_model])):
                # 准备数据
                batch = {}
                batch['imgs_ref'] = batch_['support_imgs'][0].to(accelerator.device)  # [shot, 3, H, W]
                batch['imgs_tag'] = batch_['query_img'].to(accelerator.device)  # [bs, 3, H, W]
                batch['conditioning_images_ref'] = batch_['support_masks'][0]  # [shot, H, W]

                # 将mask的值从[0, 1]转换到[-1, -1]
                batch['conditioning_images_ref'] = batch['conditioning_images_ref'] * 2 - 1
                batch['conditioning_images_tag'] = batch_['query_mask']  # [bs, H, W]
                batch['conditioning_images_tag'] = batch['conditioning_images_tag'] * 2 - 1

                # 为mask增加通道维度, 使其与图片的通道数一致
                batch['conditioning_images_ref'] = batch['conditioning_images_ref'].unsqueeze(1).repeat(1, 3, 1, 1).to(
                    accelerator.device)  # [shot, 3, H, W]
                batch['conditioning_images_tag'] = batch['conditioning_images_tag'].unsqueeze(1).repeat(1, 3, 1, 1).to(
                    accelerator.device)  # [bs, 3, H, W]

                # 从[1, nshot]范围随机选择一个值，用于模拟不同的few-shot情形
                max_nshot = args.nshot
                # temp_nshot = random.randint(1, max_nshot)
                temp_nshot = 1
                indices = random.sample(range(0, max_nshot), temp_nshot)
                batch['imgs_ref'] = batch['imgs_ref'][indices]
                batch['conditioning_images_ref'] = batch['conditioning_images_ref'][indices]

                # 将Support编码到latent空间
                latents_ref = vae.encode(batch["imgs_ref"].to(weight_dtype)).latent_dist.sample()
                latents_ref = latents_ref * vae.config.scaling_factor  # [temp_nshot, 4, 64, 64]
                cond_latents_ref = vae.encode(batch["conditioning_images_ref"].to(weight_dtype)).latent_dist.sample()
                cond_latents_ref = cond_latents_ref * vae.config.scaling_factor  # [temp_nshot, 4, 64, 64]

                # 将Query编码到latent空间
                latents_tag = vae.encode( batch["imgs_tag"].to(weight_dtype)).latent_dist.sample()
                latents_tag = latents_tag * vae.config.scaling_factor  # [bs, 4, 64, 64]
                cond_latents_tag = vae.encode(batch["conditioning_images_tag"].to(weight_dtype)).latent_dist.sample()
                cond_latents_tag = cond_latents_tag * vae.config.scaling_factor  # [bs, 4, 64, 64]

                # 构建条件latent, 用于指导 UNet 在扩散过程中的生成
                latents_rgb_cond_ref = torch.cat([latents_ref, cond_latents_ref], dim=1)  # [temp_nshot, 8, 64, 64]

                bsz = latents_tag.shape[0]

                # 为每个图像样本生成一个固定的时间步
                timesteps_nshot = torch.tensor([1 * args.train_timestep]).long().repeat(temp_nshot).to(accelerator.device)  # 此处使用1步
                timesteps = torch.tensor([1 * args.train_timestep]).long().repeat(bsz).to(accelerator.device)  # 此处使用1步

                # 获取Support和Query原型的embedding
                # 转变为[0, 1]范围
                imgs_ref = (batch["imgs_ref"] + 1) / 2  # -> [0, 1], [temp_nshot, 3, H, W]
                # imgs_tag = (batch['imgs_tag'] + 1) / 2  # -> [0, 1], [bs, 3, H, W]
                # 准备dinov2输入
                inputs_ref = processor(images=imgs_ref, return_tensors="pt", do_rescale=False)  # [temp_nshot, 3, 224, 224]
                # inputs_tag = processor(images=imgs_tag, return_tensors="pt", do_rescale=False)  # [bs, 3, 224, 224]
                inputs_ref = inputs_ref.to(device=accelerator.device, dtype=weight_dtype)
                cond_images_ref = batch_['support_masks'][0][indices].to(device=accelerator.device, dtype=weight_dtype)
                # dinov2编码
                imgs_ref_output = dinov2(output_hidden_states=False, **inputs_ref).last_hidden_state

                # 获取support的原型
                support_fg_proto, support_bg_proto = extract_vae_supp_prototype(latents_ref, cond_images_ref)  # [B, 4]

                # 获取query原型
                query_fg_proto, _ = extract_vae_query_prototype(latents_tag, support_fg_proto, support_bg_proto)
                latents_tag = torch.cat(
                    [latents_tag, query_fg_proto.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, latents_tag.shape[2],
                                                                                    latents_tag.shape[3])],
                    dim=1
                )  # [B, 2C, H, W]

                # =================== 前景训练 ===================
                # 希望模型去预测出一个负方向的噪声残差（或生成目标 latent 的反向信息）
                target_fg = -cond_latents_tag

                conditioning_embeds = get_diffusion_conditioning(imgs_ref_output,
                                                                 cond_images_ref,
                                                                 image_proj_model)  # [1, 1, 768]
                encoder_hidden_states_nshot = conditioning_embeds.repeat(temp_nshot, 1, 1)  # [temp_nshot, 1, 768]
                encoder_hidden_states = conditioning_embeds.repeat(bsz, 1, 1)  # [bs, 1, 768]

                # 两次调用Unet，分别用于条件指导和目标预测
                model_pred_cond_ref = unet(latents_rgb_cond_ref, timesteps_nshot, encoder_hidden_states_nshot,
                                           is_target=False).sample
                model_pred_fg = unet(latents_tag, timesteps, encoder_hidden_states, is_target=True).sample
                

                # 清除UNet内部注意力缓存，避免前后batch之间状态互相干扰
                if hasattr(unet, "module"):
                    unet.module.clear_attn_bank()
                else:
                    unet.clear_attn_bank()

                # 计算损失
                loss = F.mse_loss(model_pred_fg.float(), target_fg.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    all_parameters = list(unet.parameters()) + list(image_proj_model.parameters())
                    accelerator.clip_grad_norm_(all_parameters, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                    ema_image_proj_model.step(image_proj_model.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            if eval_results_all != {}:
                logs.update(
                    {str(str(dataset_name) + '_delta1'): eval_results_all[dataset_name]['delta1'] for dataset_name in
                     eval_results_all.keys()})
                with open(args.output_dir + '/eval_results.txt', 'a') as f:
                    f.writelines(str(global_step) + ' , ')
                    for dataset_name in eval_results_all.keys():
                        for key in eval_results_all[dataset_name].keys():
                            f.writelines(
                                dataset_name + '_' + key + ':' + str(eval_results_all[dataset_name][key]) + ' , ')
                    f.writelines('\n')
            progress_bar.set_postfix(**logs)

            # tensorboard to record the loss function
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    tracker.writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], global_step)
                    for dataset_name in eval_results_all.keys():
                        for key in eval_results_all[dataset_name].keys():
                            tracker.writer.add_scalar(str(dataset_name) + '_' + key,
                                                      eval_results_all[dataset_name][key], global_step)
                else:
                    logger.warn(f"image logging not implemented for {tracker.name}")

            if global_step >= args.max_train_steps:
                break

            # if accelerator.is_main_process:
            if args.validation_images is not None and global_step % args.validation_steps == 0:
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                    ema_image_proj_model.store(image_proj_model.parameters())
                    ema_image_proj_model.copy_to(image_proj_model.parameters())
                eval_results_all = log_validation(
                    vae,
                    noise_scheduler_ddim,
                    unet,
                    dinov2,
                    processor,
                    image_proj_model,
                    args,
                    accelerator,
                    weight_dtype,
                    global_step,
                    test_datasets,
                )
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())
                    ema_image_proj_model.restore(image_proj_model.parameters())


    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        image_proj_model = accelerator.unwrap_model(image_proj_model)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
            ema_image_proj_model.copy_to(image_proj_model.parameters())

        pipeline = MarigoldPipelineRGBLatentNoise.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            scheduler=noise_scheduler_ddim,
            unet=unet,
            image_encoder=dinov2,
            processor=processor,
            image_projector=image_proj_model,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            text_embeds=None,
            controlnet=None,
            customized_head=None,
            text_encoder=None,
            tokenizer=None,
            safety_checker=None,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
