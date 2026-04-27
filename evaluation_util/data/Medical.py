"""
Dataset for Training and Test
Extended from ADNet code by Hansen et al.
"""
import torch
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import glob
import os
import SimpleITK as sitk
import random
import numpy as np


def get_label_names(dataset):
    label_names = {}
    if dataset == 'CMR':
        label_names[0] = 'BG'
        label_names[1] = 'LV-MYO'
        label_names[2] = 'LV-BP'
        label_names[3] = 'RV'

    elif dataset == 'CHAOST2':
        label_names[0] = 'BG'
        label_names[1] = 'LIVER'
        label_names[2] = 'RK'
        label_names[3] = 'LK'
        label_names[4] = 'SPLEEN'
    elif dataset == 'SABS':
        label_names[0] = 'BG'
        label_names[1] = 'SPLEEN'
        label_names[2] = 'RK'
        label_names[3] = 'LK'
        label_names[4] = 'GALLBLADDER'
        label_names[5] = 'ESOPHAGUS'
        label_names[6] = 'LIVER'
        label_names[7] = 'STOMACH'
        label_names[8] = 'AORTA'
        label_names[9] = 'IVC'  # Inferior vena cava
        label_names[10] = 'PS_VEIN'  # portal vein and splenic vein
        label_names[11] = 'PANCREAS'
        label_names[12] = 'AG_R'  # right adrenal gland
        label_names[13] = 'AG_L'  # left adrenal gland

    return label_names


def get_folds(dataset):
    FOLD = {}
    if dataset == 'CMR':  # 35个病人
        FOLD[0] = set(range(0, 8))
        FOLD[1] = set(range(7, 15))
        FOLD[2] = set(range(14, 22))
        FOLD[3] = set(range(21, 29))
        FOLD[4] = set(range(28, 35))
        FOLD[4].update([0])
        return FOLD

    elif dataset == 'CHAOST2':  # 20个病人
        FOLD[0] = set(range(0, 5))
        FOLD[1] = set(range(4, 9))
        FOLD[2] = set(range(8, 13))
        FOLD[3] = set(range(12, 17))
        FOLD[4] = set(range(16, 20))
        FOLD[4].update([0])
        return FOLD
    elif dataset == 'SABS':  # 30个病人
        FOLD[0] = set(range(0, 7))
        FOLD[1] = set(range(6, 13))
        FOLD[2] = set(range(12, 19))
        FOLD[3] = set(range(18, 25))
        FOLD[4] = set(range(24, 30))
        FOLD[4].update([0])
        return FOLD
    else:
        raise ValueError(f'Dataset: {dataset} not found')

def elastic_transform(image, alpha, sigma, random_state=None, order=1, mode='reflect'):
    """
    对单张图像（多通道）或单通道图像进行弹性变换。

    参数：
        image: numpy 数组，形状为 (C, H, W) 或 (H, W)
        alpha: 形变幅度
        sigma: 高斯核标准差，用于平滑随机位移场
        random_state: 随机数生成器
        order: 插值阶数；图像用 order=1（双线性），标签用 order=0（最近邻）
        mode: 边界处理模式
    返回：
        变换后的图像，形状与输入相同
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    # 如果是多通道图像，则获取 (C, H, W)
    if image.ndim == 3:
        C, H, W = image.shape
    else:
        H, W = image.shape

    # 随机位移场（注意这里生成的是二维场，对所有通道共用同一变换）
    dx = gaussian_filter((random_state.rand(H, W) * 2 - 1), sigma, mode="reflect") * alpha
    dy = gaussian_filter((random_state.rand(H, W) * 2 - 1), sigma, mode="reflect") * alpha

    # 构造坐标网格
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    # 计算变换后的坐标（拉平成一维后再 reshape）
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # 对每个通道进行坐标映射插值
    if image.ndim == 3:
        deformed = np.empty_like(image)
        for c in range(C):
            deformed[c] = map_coordinates(image[c], indices, order=order, mode=mode).reshape(H, W)
    else:
        deformed = map_coordinates(image, indices, order=order, mode=mode).reshape(H, W)

    return deformed

def elastic_transform_pair(image, label, alpha, sigma, random_state=None):
    """
    对图像与标签同时应用同一个随机弹性变换。

    参数：
        image: numpy 数组，形状 (3, H, W) —— 图像（3通道）
        label: numpy 数组，形状 (H, W) —— 标签（单通道）
        alpha, sigma: 弹性变换参数
        random_state: 随机数生成器
    返回：
        变换后的 (image, label)
    """
    # 图像采用双线性插值（order=1），标签采用最近邻（order=0）
    image_trans = elastic_transform(image, alpha, sigma, random_state=random_state, order=1, mode='reflect')
    label_trans = elastic_transform(label, alpha, sigma, random_state=random_state, order=0, mode='reflect')
    return image_trans, label_trans


class TestDataset(Dataset):
    def __init__(self, dataname, datapath, eval_fold, supp_idx, img_size, use_original_imgsize):
        # 设置数据集路径
        if dataname == 'CMR':
            self.image_dirs = glob.glob(os.path.join(datapath, dataname, 'cmr_MR_normalized/image*'))
        elif dataname == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(datapath, dataname, 'chaos_MR_T2_normalized/image*'))
        elif dataname == 'SABS':
            self.image_dirs = glob.glob(os.path.join(datapath, dataname, 'sabs_CT_normalized/image*'))
        else:
            raise ValueError("Unsupported dataset")
        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))

        # 获取测试的fold作为测试集
        self.FOLD = get_folds(dataname)
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx in self.FOLD[eval_fold]]

        # 从测试集中选择一个作为支持集
        idx = np.arange(len(self.image_dirs))
        try:
            self.support_dir = self.image_dirs[idx[supp_idx]]
        except:
            raise ValueError(f'Invalid support index: {supp_idx}, available indices: {idx}')

        # 从测试集中移除支持集
        self.image_dirs.pop(idx[supp_idx])

        self.label = None
        self.img_size = img_size
        self.use_original_imgsize = use_original_imgsize

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        # 读取测试集图像
        img_path = self.image_dirs[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))  # (slices, h, w)

        if not self.use_original_imgsize:
            img = F.interpolate(torch.tensor(img).unsqueeze(1).float(), size=self.img_size,
                                mode='nearest').squeeze().numpy()

        # 将图像的值变到[-1, 1]
        img = (img - img.min()) / (img.max() - img.min())  # -> [0, 1]
        img = (img * 2.0) - 1.0  # [0, 1] -> [-1, 1]
        img = np.stack(3 * [img], axis=1)

        # 读取对应的标签
        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1])).astype('int')  # (slices, h, w)
        if not self.use_original_imgsize:
            lbl = F.interpolate(torch.tensor(lbl).unsqueeze(1).float(), size=self.img_size,
                                mode='nearest').squeeze().numpy()
        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        lbl = 1 * (lbl == self.label)

        # 选取有标签的切片
        idx = lbl.sum(axis=(1, 2)) > 0
        label = lbl[idx]

        batch = {
            'id': img_path,
            'query_img': torch.from_numpy(img[idx]).float(),  # [slices, 3, H, W]
            'query_mask': torch.from_numpy(label).float()  # [slices, H, W]
        }

        # img: [-1, 1], mask: [0, 1]
        assert (-1 <= batch['query_img'].all() <= 1), (batch['query_img'].min(), batch['query_img'].max())
        assert (0 <= batch['query_mask'].all() <= 1), (batch['query_mask'].min(), batch['query_mask'].max())

        return batch

    def get_support_index(self, n_shot, C):
        """
        Selecting intervals according to Ouyang et al.
        """
        if n_shot == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (n_shot * 2)
            part_interval = (1.0 - 1.0 / n_shot) / (n_shot - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_shot)]

        return (np.array(pcts) * C).astype('int')

    def getSupport(self, label, all_slices=True, N=None):
        if label is None:
            raise ValueError('Need to specify label!')

        # 读取支持集图像
        img_path = self.support_dir
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))  # (slices, h, w)

        # 处理图像
        if not self.use_original_imgsize:
            img = F.interpolate(torch.tensor(img).unsqueeze(1).float(), size=self.img_size,
                                mode='nearest').squeeze().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # -> [0, 1]
        img = (img * 2.0) - 1.0  # [0, 1] -> [-1, 1]
        img = np.stack(3 * [img], axis=1)

        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))  # (slices, h, w)
        if not self.use_original_imgsize:
            lbl = F.interpolate(torch.tensor(lbl).unsqueeze(1).float(), size=self.img_size,
                                mode='nearest').squeeze().numpy()
        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        lbl = 1 * (lbl == label)

        batch = {}
        if all_slices:  # 选取所有支持集切片
            batch['support_imgs'] = torch.from_numpy(img)
            batch['support_masks'] = torch.from_numpy(lbl)
        else:  # 选取N个支持集切片
            if N is None:
                raise ValueError('Need to specify number of labeled slices!')
            idx = lbl.sum(axis=(1, 2)) > 0
            idx_ = self.get_support_index(N, idx.sum())
            batch['support_imgs'] = torch.from_numpy(img[idx][idx_]).float()
            batch['support_masks'] = torch.from_numpy(lbl[idx][idx_]).float()

        # img: [-1, 1], mask: [0, 1]
        assert (-1 <= batch['support_imgs'].all() <= 1), (batch['support_imgs'].min(), batch['support_imgs'].max())
        assert (0 <= batch['support_masks'].all() <= 1), (batch['support_masks'].min(), batch['support_masks'].max())

        return batch


class TrainDataset(Dataset):
    def __init__(self, dataname, datapath, eval_fold, img_size, use_original_imgsize, shot, setting,
                 use_gt=False, gt_rate=1.0, test_label=None, max_iter=5000):
        self.n_sv = '1000' if dataname == 'CMR' else '5000'
        self.use_gt = use_gt
        self.gt_rate = gt_rate
        self.eval_fold = eval_fold
        self.max_iter = max_iter
        self.img_size = img_size
        self.use_original_imgsize = use_original_imgsize
        self.n_shot = shot
        if setting == 1:
            self.exclude_label = None
        elif setting == 2:
            if dataname == 'CHAOST2':
                self.exclude_label = [1, 2, 3, 4]
            elif dataname == 'SABS':
                self.exclude_label = [1, 2, 3, 6]
            elif dataname == 'CMR':
                self.exclude_label = None
        else:
            raise ValueError('Invalid setting!')


        self.min_size = 200
        # self.min_size = 400
        if test_label is None:
            self.test_label = None
        else:
            if test_label == 1:
                self.test_label = [1, 4]
                if dataname == 'SABS':
                    self.test_label = [1, 6]
            elif test_label == 2:
                self.test_label = [2, 3]
            else:
                raise ValueError('Invalid test_label!')
        # self.test_label = test_label

        # 读取数据集路径
        if dataname == 'CMR':
            self.image_dirs = glob.glob(os.path.join(datapath, dataname, 'cmr_MR_normalized/image*'))
            self.label_dirs = glob.glob(os.path.join(datapath, dataname, 'cmr_MR_normalized/label*'))
        elif dataname == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(datapath, dataname, 'chaos_MR_T2_normalized/image*'))
            self.label_dirs = glob.glob(os.path.join(datapath, dataname, 'chaos_MR_T2_normalized/label*'))
        elif dataname == 'SABS':
            self.image_dirs = glob.glob(os.path.join(datapath, dataname, 'sabs_CT_normalized/image*'))
            self.label_dirs = glob.glob(os.path.join(datapath, dataname, 'sabs_CT_normalized/label*'))
        else:
            raise ValueError("Unsupported dataset")
        self.sprvxl_dirs = glob.glob(os.path.join(datapath, dataname, 'supervoxels_' + self.n_sv, 'super*'))

        # 排序数据集
        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
        self.label_dirs = sorted(self.label_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
        self.sprvxl_dirs = sorted(self.sprvxl_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))

        # 从数据集中移除测试集
        self.FOLD = get_folds(dataname)
        if self.eval_fold is not None:
            try:
                self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if
                                   idx not in self.FOLD[eval_fold]]
                self.label_dirs = [elem for idx, elem in enumerate(self.label_dirs) if
                                   idx not in self.FOLD[eval_fold]]
                self.sprvxl_dirs = [elem for idx, elem in enumerate(self.sprvxl_dirs) if
                                    idx not in self.FOLD[eval_fold]]
            except:
                raise ValueError(f'Invalid fold: {eval_fold}, available folds: {list(self.FOLD.keys())}')
        else:
            raise ValueError('Need to specify evaluation fold!')

        # 读取图像和标签
        self.images = {}
        self.labels = {}
        self.sprvxls = {}
        for image_dir, label_dir, sprvxl_dir in zip(self.image_dirs, self.label_dirs, self.sprvxl_dirs):
            self.images[image_dir] = sitk.GetArrayFromImage(sitk.ReadImage(image_dir))
            self.labels[label_dir] = sitk.GetArrayFromImage(sitk.ReadImage(label_dir))
            self.sprvxls[sprvxl_dir] = sitk.GetArrayFromImage(sitk.ReadImage(sprvxl_dir))

    def __len__(self):
        return self.max_iter

    def __getitem__(self, idx):
        # 随机选择一个病人
        pat_idx = random.choice(range(len(self.image_dirs)))

        # 获取图像和标签
        img = self.images[self.image_dirs[pat_idx]]   # numpy.ndarray: (slices, h, w)
        if self.use_gt and random.random() < self.gt_rate:
            mask = self.labels[self.label_dirs[pat_idx]]  # numpy.ndarray: (slices, h, w)
        else:
            mask = self.sprvxls[self.sprvxl_dirs[pat_idx]]  # numpy.ndarray: (slices, h, w)

        # # 移除存在有 exclude_label 中任意label的切片 (Fixed)
        # if self.exclude_label is not None:
        #     idx = np.arange(mask.shape[0])
        #     exclude_idx = np.full(mask.shape[0], False, dtype=bool)
        #     for i in range(len(self.exclude_label)):
        #         exclude_idx = exclude_idx | (np.sum(mask == self.exclude_label[i], axis=(1, 2)) > 0)
        #     exclude_idx = idx[exclude_idx]
        # else:
        #     exclude_idx = []

        # 前人工作使用的处理方法
        if self.exclude_label is not None:  # identify the slices containing test labels
            idx = np.arange(mask.shape[0])
            exclude_idx = np.full(mask.shape[0], True, dtype=bool)
            for i in range(len(self.exclude_label)):
                exclude_idx = exclude_idx & (np.sum(mask == self.exclude_label[i], axis=(1, 2)) > 0)
            exclude_idx = idx[exclude_idx]
        else:
            exclude_idx = []

        # 处理图像
        org_query_imsize = img.shape[1:]  # (h, w)
        if not self.use_original_imgsize:
            img = F.interpolate(torch.tensor(img).unsqueeze(1).float(), size=self.img_size,
                                mode='nearest').squeeze().numpy()  # (slices, H, W)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)  # -> [0, 1]

        # 训练标签
        lbl = mask.copy()

        # 获取所有类别
        unique = list(np.unique(lbl).astype('int'))
        unique.remove(0)  # 移除背景类
        # 如果使用gt，则移除测试集标签
        if self.use_gt and self.test_label is not None:
            unique = list(set(unique) - set(self.test_label))

        size = 0
        while size < self.min_size:
            n_slices = self.n_shot
            while n_slices < (self.n_shot + 1):
                cls_idx = random.choice(unique)  # cls_idx is sampled class id

                # extract slices containing the sampled class
                sli_idx = np.sum(lbl == cls_idx, axis=(1, 2)) > 0
                idx = np.arange(lbl.shape[0])
                sli_idx = idx[sli_idx]
                sli_idx = list(
                    set(sli_idx) - set(np.intersect1d(sli_idx, exclude_idx)))  # remove slices containing test labels
                n_slices = len(sli_idx)

            # generate possible subsets with successive slices (size = self.n_shot + 1)
            subsets = []
            for i in range(len(sli_idx)):
                if not subsets:
                    subsets.append([sli_idx[i]])
                elif sli_idx[i - 1] + 1 == sli_idx[i]:
                    subsets[-1].append(sli_idx[i])
                else:
                    subsets.append([sli_idx[i]])
            i = 0
            while i < len(subsets):
                if len(subsets[i]) < (self.n_shot + 1):
                    del subsets[i]
                else:
                    i += 1
            if not len(subsets):
                return self.__getitem__(idx + np.random.randint(low=0, high=self.max_iter - 1, size=(1,)))

            # sample support and query slices
            i = random.choice(np.arange(len(subsets)))  # subset index
            i = random.choice(subsets[i][:-self.n_shot])
            sample = np.arange(i, i + (self.n_shot) + 1)

            lbl_cls = 1 * (lbl == cls_idx)

            size = max(np.sum(lbl_cls[sample[0]]), np.sum(lbl_cls[sample[1]]))

        # invert order
        if np.random.random(1) > 0.5:
            sample = sample[::-1]  # successive slices (inverted)

        # 处理图像
        sup_img = img[sample[:self.n_shot]][None,]
        sup_img = np.stack((sup_img, sup_img, sup_img), axis=2)  # (1, n_shot, 3, H, W)
        qry_img = img[sample[self.n_shot:]]
        qry_img = np.stack((qry_img, qry_img, qry_img), axis=1)  # (1, 3, H, W)
        # [0, 1] -> [-1, 1]
        sup_img = (sup_img * 2.0) - 1.0
        qry_img = (qry_img * 2.0) - 1.0
        # to Tensor
        sup_img = torch.tensor(sup_img).float().squeeze(0)  # [n_shot, 3, 512, 512]
        qry_img = torch.tensor(qry_img).float().squeeze(0)  # [3, 512, 512]

        # 处理标签，将尺寸调整到(H, W)
        sup_lbl = lbl_cls[sample[:self.n_shot]][None,]  # (1, n_shot, h, w)
        qry_lbl = lbl_cls[sample[self.n_shot:]]  # (1, h, w)
        if not self.use_original_imgsize:
            sup_lbl = F.interpolate(torch.tensor(sup_lbl).float(), size=self.img_size, mode='nearest').squeeze(0)  # [n_shot, 512, 512]
            qry_lbl = F.interpolate(torch.tensor(qry_lbl).unsqueeze(1).float(), size=self.img_size,
                                    mode='nearest').squeeze(1).squeeze(0)  # [512, 512]


        # ----------------------- 应用数据增强 -----------------------
        # 定义随机参数
        angle = random.uniform(-5, 5)  # 旋转角度（度数）
        translate = (random.uniform(-5, 5), random.uniform(-5, 5))  # 水平、垂直平移（像素）
        scale = random.uniform(0.9, 1.2)  # 缩放因子
        shear = random.uniform(-5, 5)  # 剪切角度
        # 弹性变换参数（可根据实际情况调整）
        alpha = 10
        sigma = 5
        # 随机仿射变换（Geom Transform）和随机弹性变换（Elastic Deformation）
        if random.random() >= 0.5:
            # 对支持集进行随机仿射变换
            sup_img_list = []
            for i in range(sup_img.shape[0]):
                img_i = TF.affine(sup_img[i], angle=angle, translate=translate, scale=scale, shear=shear,
                                  interpolation=TF.InterpolationMode.BILINEAR)
                sup_img_list.append(img_i)
            sup_img = torch.stack(sup_img_list, dim=0)
            sup_lbl_list = []
            for i in range(sup_lbl.shape[0]):
                lbl_i = TF.affine(sup_lbl[i].unsqueeze(0).float(), angle=angle, translate=translate, scale=scale,
                                  shear=shear, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
                sup_lbl_list.append(lbl_i)
            sup_lbl = torch.stack(sup_lbl_list, dim=0)

            # 对支持集进行随机弹性变换
            aug_sup_imgs = []
            aug_sup_lbls = []
            for img, lbl in zip(sup_img, sup_lbl):
                # 将 torch.Tensor 转为 numpy 数组
                img_np = img.cpu().numpy()
                lbl_np = lbl.cpu().numpy()
                # 使用同一个随机数生成器保证图像与标签变换一致
                random_state = np.random.RandomState(2025)
                img_aug, lbl_aug = elastic_transform_pair(img_np, lbl_np, alpha, sigma, random_state=random_state)
                # 转回 torch.Tensor
                aug_sup_imgs.append(torch.tensor(img_aug, dtype=torch.float32))
                aug_sup_lbls.append(torch.tensor(lbl_aug, dtype=torch.float32))
            # 堆叠为 tensor
            sup_img = torch.stack(aug_sup_imgs)  # 形状: [n_shot, 3, H, W]
            sup_lbl = torch.stack(aug_sup_lbls)  # 形状: [n_shot, H, W]
        else:
            # 对查询集进行仿射变换
            qry_img = TF.affine(qry_img, angle=angle, translate=translate, scale=scale, shear=shear,
                                interpolation=TF.InterpolationMode.BILINEAR)
            qry_lbl = TF.affine(qry_lbl.unsqueeze(0).float(), angle=angle, translate=translate, scale=scale,
                                shear=shear, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

            # 对查询集进行随机弹性变换
            qry_img_np = qry_img.cpu().numpy()
            qry_lbl_np = qry_lbl.cpu().numpy()
            random_state = np.random.RandomState(2025)
            qry_img_np, qry_lbl_np = elastic_transform_pair(qry_img_np, qry_lbl_np, alpha, sigma,
                                                            random_state=random_state)
            # 转回 torch.Tensor
            qry_img = torch.tensor(qry_img_np, dtype=torch.float32)  # 形状: [3, H, W]
            qry_lbl = torch.tensor(qry_lbl_np, dtype=torch.float32)  # 形状: [H, W]


        # Gamma参数
        gamma_value = random.uniform(0.5, 1.5)
        # 随机Gamma变换（Gamma Transform）
        if random.random() >= 0.5:
            # 支持图像进行随机Gamma变换
            sup_img_list = []
            for i in range(sup_img.shape[0]):
                img_norm = (sup_img[i] + 1.0) / 2.0
                img_norm = TF.adjust_gamma(img_norm, gamma=gamma_value)
                sup_img_list.append(img_norm * 2.0 - 1.0)  # [0, 1] -> [-1, 1]
            sup_img = torch.stack(sup_img_list, dim=0)
        else:
            # 查询图像进行随机Gamma变换
            qry_img_norm = (qry_img + 1.0) / 2.0
            qry_img_norm = TF.adjust_gamma(qry_img_norm, gamma=gamma_value)
            qry_img = qry_img_norm * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        # --------------------- 数据增强结束 ---------------------


        assert (-1 <= sup_img.all() <= 1), (sup_img.min(), sup_img.max())
        assert (-1 <= qry_img.all() <= 1), (qry_img.min(), qry_img.max())
        assert (0 <= sup_lbl.all() <= 1), (sup_lbl.min(), sup_lbl.max())
        assert (0 <= qry_lbl.all() <= 1), (qry_lbl.min(), qry_lbl.max())
        
        # print(f"{qry_img.dtype} {qry_lbl.dtype} {org_query_imsize[0]} {org_query_imsize[1]} {sup_img.dtype} {sup_lbl.dtype} {cls_idx}")

        batch = {
            'query_img': qry_img,   # [3, H, W], [-1, 1]
            'query_mask': qry_lbl,  # [H, W], [0, 1]

            'org_query_imsize': org_query_imsize,

            'support_imgs': sup_img,   # [n_shot, 3, H, W], [-1, 1]
            'support_masks': sup_lbl,  # [n_shot, H, W], [0, 1]

            'class_id': cls_idx
        }

        return batch
