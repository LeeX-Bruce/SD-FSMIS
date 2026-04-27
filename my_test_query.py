#!/usr/bin/env python
"""
For evaluation
"""
import os
import shutil
import SimpleITK as sitk
import cv2

from diffusers import AutoencoderKL
from torch import optim, nn
from torchvision.transforms.functional import rotate, adjust_brightness

from diffews.my_marigold_pipeline_rgb_latent_noise_v2_query import MarigoldPipelineRGBLatentNoise as MarigoldPipeline
from diffews.models.unet_2d_condition_v2 import MyUNet2DConditionModel as CustomUNet2DConditionModel
from diffews.models.FeatureToConditioningMLP import FeatureToConditioningMLP
# from diffews.models.proto import Conv_1x1

from transformers import AutoModel, AutoImageProcessor
import torchvision.transforms as transforms

from marigold.util.scheduler_customized import DDIMSchedulerCustomized as DDIMScheduler

from torch.utils.data import DataLoader, TensorDataset

from evaluation_util.common import utils
from evaluation_util.data.Medical import *

import random
import time
import re
from medpy.metric.binary import hd, assd, asd

# 设置随机种子
random.seed(2025)
utils.fix_randseed(2025)


class Scores():
    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.patient_dice = []
        self.patient_iou = []
        self.patient_hd95 = []
        self.patient_assd = []
        self.patient_msd   = []  # 这里用 directed asd (pred -> gt)

    def record(self, preds, label, spacing=None):
        if spacing is None:
            spacing = [1.0, 1.0]
        
        preds = preds.long()
        label = label.long()

        # 确保 preds 和 label 中只包含0和1
        assert torch.all(torch.isin(preds, torch.tensor([0, 1])))
        assert torch.all(torch.isin(label, torch.tensor([0, 1])))

        tp = torch.sum((label == 1) * (preds == 1))
        tn = torch.sum((label == 0) * (preds == 0))
        fp = torch.sum((label == 0) * (preds == 1))
        fn = torch.sum((label == 1) * (preds == 0))

        self.patient_dice.append(2 * tp / (2 * tp + fp + fn))
        self.patient_iou.append(tp / (tp + fp + fn))
        
        preds = preds.cpu().numpy().astype(bool)   # medpy 需要 bool 或 0/1 float
        label = label.cpu().numpy().astype(bool)
        
        B = preds.shape[0]
        for b in range(B):
            p = preds[b]
            g = label[b]

            # medpy metrics (需要至少一个前景点，否则返回 inf/nan)
            if np.any(p) and np.any(g):
                # ASSD = symmetric average surface distance
                assd_val = assd(p, g, voxelspacing=spacing)
                self.patient_assd.append(assd_val)

                # MSD = directed average surface distance (pred -> gt)
                msd_val = asd(p, g, voxelspacing=spacing)
                self.patient_msd.append(msd_val)

                # HD95：medpy 没有内置，需要自己算
                from scipy.ndimage import distance_transform_edt
                # 边界距离图
                gt_dt = distance_transform_edt(~g, sampling=spacing)
                pred_dt = distance_transform_edt(~p, sampling=spacing)

                surf_gt = gt_dt[p]   # pred 表面上的 gt 距离
                surf_pred = pred_dt[g]  # gt 表面上的 pred 距离

                if len(surf_gt) > 0 and len(surf_pred) > 0:
                    all_dists = np.concatenate([surf_gt, surf_pred])
                    hd95_val = np.percentile(all_dists, 95)
                else:
                    hd95_val = float('inf')
                self.patient_hd95.append(hd95_val)
            else:
                # 空预测或空 GT
                self.patient_assd.append(float('inf'))
                self.patient_msd.append(float('inf'))
                self.patient_hd95.append(float('inf'))

        self.TP += tp
        self.TN += tn
        self.FP += fp
        self.FN += fn

    def compute_dice(self):
        # 返回整体的Dice系数
        return 2 * self.TP / (2 * self.TP + self.FP + self.FN)

    def compute_iou(self):
        # 返回整体的IoU
        return self.TP / (self.TP + self.FP + self.FN)
        
    # compute 函数同前，新增
    def compute_hd95(self):
        vals = [v for v in self.patient_hd95 if np.isfinite(v)]
        return np.mean(vals) if vals else float('nan')

    def compute_assd(self):
        vals = [v for v in self.patient_assd if np.isfinite(v)]
        return np.mean(vals) if vals else float('nan')

    def compute_msd(self):
        vals = [v for v in self.patient_msd if np.isfinite(v)]
        return np.mean(vals) if vals else float('nan')
        

def get_connected_components(query_pred_original, query_pred_logits, return_conf=False):
    """
    从二值化预测结果中提取连通区域，并可选择计算每个区域的置信度  1 2 H W
    """
    cca_output = cv2.connectedComponentsWithStats(query_pred_original.astype(np.uint8), connectivity=8)  # 4连通指的是上、下、左、右，8连通指的是上、下、左、右、左上、右上、右下、左下

    # 需要返回置信度
    if return_conf:
        cca_conf = {}  # conf by id
        # 将模型输出的logits通过softmax转换为概率
        query_probs = query_pred_logits.softmax(0).cpu().detach().numpy()  # [1, H, W]
        # 遍历连通区域
        for j in range(cca_output[0]):
            if j == 0:
                cca_conf[0] = 0  # 背景
                continue
            cca_conf[j] = ((query_probs.flatten() * (cca_output[1] == j).flatten()).sum() / (
            (query_pred_original.flatten().sum() + 1e-6)))  # 计算该区域的总概率占整个预测区域的比例，反映其相对重要性
        return cca_output, cca_conf

    return cca_output, None

def cca(query_pred_original, query_pred_logits, return_conf=False, return_cc=False):
    """
    从二值预测图中提取置信度最高的连通区域
    query_pred_original: H x W, 二值化预测结果
    query_pred_logits: 1 x H x W, 模型输出的logits
    """
    # 对二值预测结果进行腐蚀操作
    cca_output, cca_conf = get_connected_components(query_pred_original, query_pred_logits, return_conf=True)

    # 获取连通区域及置信度
    max_conf = cca_conf[0]
    for k, v in cca_conf.items():
        if v > max_conf:
            max_conf = v
            max_key = k

    if max_conf == 0:
        # 若所有区域置信度为0，返回全零掩码
        query_pred = np.zeros_like(query_pred_original)
    else:
        # 修改连通区域输出，仅保留目标区域
        new_cca_output = list(cca_output)
        new_cca_output[0] = 2  # 仅背景+目标区域
        new_cca_output[1] = np.where(cca_output[1] != max_key, 0, 1)  # 二值化，目标区域置1，其他区域置0
        new_cca_output[2] = cca_output[2][[0, max_key]]  # 统计信息（背景+目标）
        new_cca_output[3] = cca_output[3][[0, max_key]]  # 质心（背景+目标）
        cca_output = tuple(new_cca_output)

        # 生成二值掩码
        query_pred = (cca_output[1] == 1).astype(np.uint8)
        query_pred = (query_pred > 0).astype(np.uint8)

    # 返回修改后的连通区域信息
    if return_cc:
        return cca_output

    query_pred_original = query_pred_original * query_pred

    if return_conf:
        return query_pred_original, max_conf

    return query_pred_original



if __name__ == '__main__':
    # 设置要使用的GPU设备
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # 加载配置
    # test_label = [1, 2, 3, 4]  # CHAOST2
    test_label = [1, 2, 3, 6]  # SABS
    # dataname = 'CHAOST2'
    dataname = 'SABS'
    eval_fold = 0  # 0, 1, 2, 3, 4
    supp_idx = 2
    checkpoint_num = 15000
    res = 256
    # checkpoint_name = 'CHAOST2_setting1_res256_lr1e-5_step15000_bs1_gpu0_seed42'  # TODP
    checkpoint_name = 'SABS_setting1_res256_lr1e-5_step15000_bs1_gpu3_seed42'  # TODO
    test_fold_name = 'cv' + str(eval_fold)
    
    # U-Net
    unet_ckpt_path = os.path.join('logs', checkpoint_name, test_fold_name, 'checkpoint-' + str(checkpoint_num), 'unet')
    # VTCT
    image_projector_ckpt_path = os.path.join('logs', checkpoint_name, test_fold_name, 'checkpoint-' + str(checkpoint_num), 'image_projector')
    
    # DINOv2
    image_encoder_ckpt_path = os.path.join('logs', checkpoint_name, test_fold_name, 'image_encoder')
    processor_ckpt_path = os.path.join('logs', checkpoint_name, test_fold_name, 'processor')
    
    
    checkpoint = 'weight/stable-diffusion-1-5-ref8inchannels-tag8inchannels'
    scheduler_load_path = 'scheduler_1.0_1.0'
    result_name = os.path.join(checkpoint_name, test_fold_name)
    test_timestep = 1
    r_threshold = 0.25
    is_cd = False
    use_cca = True

    # 创建目录用于保存预测结果和源文件
    result_path = f'result/{result_name}'
    os.makedirs(result_path, exist_ok=True)

    # 创建模型并加载权重
    unet = CustomUNet2DConditionModel.from_pretrained(
        unet_ckpt_path, subfolder="unet", revision=None
    )
    image_projector = FeatureToConditioningMLP.from_pretrained(
        image_projector_ckpt_path, subfolder="image_projector", revision=None
    )
    vae = AutoencoderKL.from_pretrained(
        checkpoint, subfolder="vae"
    )
    image_encoder = AutoModel.from_pretrained(image_encoder_ckpt_path)
    processor = AutoImageProcessor.from_pretrained(processor_ckpt_path)
    marigold_params_ckpt = dict(
        torch_dtype=torch.float32,
        unet=unet,
        vae=vae,
        image_encoder=image_encoder,
        image_projector=image_projector,
        processor=processor,
        text_encoder=None,
        tokenizer=None,
        safety_checker=None,
        text_embeds=None,
        controlnet=None,
        customized_head=None,
    )
    if scheduler_load_path:
        marigold_params_ckpt['scheduler'] = DDIMScheduler.from_pretrained(scheduler_load_path, subfolder="scheduler")

    # 加载数据
    data_config = {
        'dataname': dataname,
        'datapath': '/data/lmh/dataset',
        'eval_fold': eval_fold,
        'supp_idx': supp_idx,
        'img_size': res,
        'use_original_imgsize': False
    }
    test_dataset = TestDataset(**data_config)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 获取类别名称
    labels = get_label_names(data_config['dataname'])

    # 记录每个类别的Dice和IoU
    class_dice = {}
    class_iou = {}
    class_hd95 = {}
    class_assd = {}
    class_msd = {}
    
    # 加载模型
    pipe = MarigoldPipeline.from_pretrained(checkpoint, **marigold_params_ckpt)
    pipe = pipe.to(device)
    pipe.test_timestep = test_timestep
    
    pipe.unet.enable_xformers_memory_efficient_attention()

    # 初始化log信息
    log_message = ""

    # 开始验证
    print(f'Starting validation...')
    log_message += 'Starting validation...\n'
    for label_val, label_name in labels.items():

        # 跳过背景类和不在test_label中的类
        if label_name == 'BG':
            continue
        elif (not np.intersect1d([label_val], test_label)):
            continue

        print(f'Test Class: {label_name}')
        log_message += f'Test Class: {label_name}\n'

        # 获取当前类别的支持集
        support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=3)

        # 为测试集设置测试标签
        test_dataset.label = label_val

        # Test
        with torch.no_grad():
            # 取出支持样本和掩码
            support_image = support_sample['support_imgs'].float().to(device)  # [n-shot, 3, H, W]
            support_fg_mask = support_sample['support_masks'].float().to(device)  # [n-shot, H, W]

            # 初始化评分器
            scores = Scores()
            
            # 初始化计时统计
            total_time = 0.0
            slice_times = []

            # 遍历query volumes
            for i, sample in enumerate(test_loader):
                # 取出query volumes
                query_image = [sample['query_img'][i].float().cuda() for i in
                               range(sample['query_img'].shape[0])]  # [C x 3 x H x W] query_image is list {(C x 3 x H x W)}
                query_label = sample['query_mask'].float().squeeze(0)  # C x H x W, [0, 1]
                query_id = sample['id'][0].split('image_')[1][:-len('.nii.gz')]

                # 初始化query预测并获取query volume的slice数
                query_pred = torch.zeros(query_label.shape)  # C x H x W
                C_q = sample['query_img'].shape[1]  # slice number of query img

                # 将query volume分成n_part个子块，每个子块与对应的support slice匹配
                idx_ = np.linspace(0, C_q, 3 + 1).astype('int')
                for sub_chunck in range(3):  # n_part = 3
                    query_image_s = query_image[0][idx_[sub_chunck]:idx_[sub_chunck + 1]]  # [C', 3, H, W]
                    
                    support_image_s = support_image  # [n-shot, 3, H, W]

                    # 准备输入
                    input_images_torch = [
                        support_image_s,
                        query_image_s,
                        support_fg_mask_s,
                    ]
                    
                    # 开始计时
                    start_time = time.time()
                    
                    # 预测
                    pipe_out = pipe(
                        input_images_torch,
                        denoising_steps=1,
                        ensemble_size=1,
                        processing_res=512,
                        batch_size=query_image_s.shape[0],
                        show_progress_bar=False,
                        mode='seg',
                        seed=0,
                    )

                    query_pred_s = [transforms.functional.to_tensor(img) for img in pipe_out.seg_colored]
                    query_pred_s = torch.stack(query_pred_s)  # [C', 3, H, W]
                    query_pred_s_logits = query_pred_s.mean(dim=1)  # [C', H, W]

                    # 转为0-1标签
                    if r_threshold > 0:
                        dynamic_thres = query_pred_s.max() * r_threshold
                        query_pred_s = (query_pred_s.mean(dim=1) > dynamic_thres).to(query_image_s)  # [C', H, W]

                    if use_cca:
                        # 对每个查询切片的预测结果应用CCA
                        query_pred_s_cca = []
                        for i in range(query_pred_s.shape[0]):
                            # query_pred_s[i]: [H, W], query_pred_s_logits[i].unsqueeze(0): [1, H, W]
                            query_pred_cca = cca(query_pred_s[i].cpu().numpy(), query_pred_s_logits[i].unsqueeze(0))
                            query_pred_s_cca.append(torch.from_numpy(query_pred_cca))
                        query_pred_s_cca = torch.stack(query_pred_s_cca, dim=0)  # 将CCA处理后的结果重新拼接成一个张量

                        query_pred[idx_[sub_chunck]:idx_[sub_chunck + 1]] = query_pred_s_cca
                    else:
                        query_pred[idx_[sub_chunck]:idx_[sub_chunck + 1]] = query_pred_s
                        
                    # 结束计时并记录
                    end_time = time.time()
                    batch_time = end_time - start_time
                    per_slice_time = batch_time / query_image_s.shape[0]
                    total_time += batch_time
                    slice_times.extend([per_slice_time] * query_image_s.shape[0])    

                # 计算指标
                scores.record(query_pred, query_label)

                # Log.
                print(
                    f'Tested query volume: {query_id}')
                print(f'Dice score: {scores.patient_dice[-1].item()}')
                log_message += (
                    f'Tested query volume: {query_id}\n'
                    f'Dice score: {scores.patient_dice[-1].item()}\n'
                )

                # 保存预测结果
                file_name = os.path.join(f"{result_path}/prediction_{query_id}_{label_name}.nii.gz")
                itk_pred = sitk.GetImageFromArray(query_pred)
                sitk.WriteImage(itk_pred, file_name, True)
                print(f'{query_id} has been saved. ')
                log_message += (f'{query_id} has been saved.\n')


            # 记录每个类别的平均Dice和IoU
            class_dice[label_name] = scores.compute_dice()
            class_iou[label_name] = scores.compute_iou()
            class_hd95[label_name] = scores.compute_hd95()
            class_assd[label_name] = scores.compute_assd()
            class_msd[label_name] = scores.compute_msd()
            print(f'Test Class: {label_name}')
            print(f'Mean class IoU: {class_iou[label_name]}\n')
            print(f'Mean class Dice: {class_dice[label_name]}\n')
            log_message += (
                f'Test Class: {label_name}\n'
                f'Mean class IoU: {class_iou[label_name]}\n'
                f'Mean class Dice: {class_dice[label_name]}\n\n'
            )

        # pipe.hook_handle.remove()
        #del pipe

    # 记录每个类整体平均Dice和IoU
    print(f'====================================')
    print(f'Final results')
    print(f'Mean IoU: {class_iou}')
    print(f'Mean Dice: {class_dice}')
    print(f'Mean hd95: {class_hd95}')
    print(f'Mean assd: {class_assd}')
    print(f'Mean msd: {class_msd}')
    log_message += (
        f'====================================\n'
        f'Final results...\n'
        f'Mean IoU: {class_iou}\n'
        f'Mean Dice: {class_dice}\n'
        f'Mean hd95: {class_hd95}\n'
        f'Mean assd: {class_assd}\n'
        f'Mean msd: {class_msd}\n'
    )
    
    # 打印计时结果
    print(f"总推理时间: {total_time:.2f}秒")
    print(f"处理切片总数: {len(slice_times)}")
    print(f"平均每切片时间: {np.mean(slice_times):.4f}秒")
    print(f"最快切片时间: {np.min(slice_times):.4f}秒")
    print(f"最慢切片时间: {np.max(slice_times):.4f}秒")
    print(f"时间标准差: {np.std(slice_times):.4f}秒")
    log_message += (
        f"总推理时间: {total_time:.2f}秒\n"
        f"处理切片总数: {len(slice_times)}\n"
        f"平均每切片时间: {np.mean(slice_times):.4f}秒\n"
        f"最快切片时间: {np.min(slice_times):.4f}秒\n"
        f"最慢切片时间: {np.max(slice_times):.4f}秒\n"
        f"时间标准差: {np.std(slice_times):.4f}秒\n"
    )

    def dict_Avg(Dict):
        L = len(Dict)  # 取字典中键值对的个数
        S = sum(Dict.values())  # 取字典中键对应值的总和
        A = S / L
        return A

    print(f'Whole mean Dice: {dict_Avg(class_dice)}')
    print(f'End of validation.')
    log_message += (
        f'Whole mean Dice: {dict_Avg(class_dice)}\n'
        f'Whole mean hd95: {dict_Avg(class_hd95)}\n'
        f'Whole mean assd: {dict_Avg(class_assd)}\n'
        f'End of validation.\n'
    )

    if is_cd:
        file_name = "log_cd_" + str(checkpoint_num)
    else:
        file_name = "log_" + str(checkpoint_num)
    if use_cca:
        file_name = file_name + "_CCA"

    file_name = file_name + ".txt"
    # with open(f'result_time_5-shot/{result_name}/{file_name}', 'w') as file:
    with open(f'{result_path}/{result_name}/{file_name}', 'w') as file:
        file.write(log_message)

