# SD-FSMIS

**Paper**: [SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](https://arxiv.org/abs/2604.03134)

This repository contains the official implementation of SD-FSMIS, a frequency-aware medical image segmentation method leveraging Stable Diffusion priors.

## 项目结构

```
DiffewS/
├── data/                          # 医学影像数据集
│   ├── CHAOST2/                   # CHAOST2数据集
│   ├── SABS/                      # SABS数据集
│   └── supervoxels/               # 超体素数据生成代码，源于RPT(https://github.com/YazhouZhu19/RPT)
├── diffews/                       # DiffewS
│   ├── my_marigold_pipeline_rgb_latent_noise_v2_query.py  # 自定义pipeline
│   └── models/
│       ├── attention_processor_v2.py      # U-Net注意力
│       ├── unet_2d_condition_v2.py        # U-Net
│       ├── proto.py                       # Query Enhancement
│       └── FeatureToConditioningMLP.py    # Visual-to-Textual Condition Translator
├── evaluation_util/               # 评估工具
│   ├── common/                    # 通用工具函数
│   └── data/                      # 数据集加载器
│       ├── coco.py
│       ├── pascal.py
│       ├── pascal_part.py
│       ├── paco_part.py
│       ├── Medical.py             # 使用的加载器
│       ├── lvis.py
│       └── fss.py
├── marigold/                      # Marigold
├── scripts/                       # 脚本文件
│   ├── model_weight_preprocess.py # SD模型权重预处理
│   └── my_train_query.sh          # 训练启动脚本
├── scheduler_1.0_1.0/             # 调度器配置
│   └── scheduler_config.json
├── train_tools/                   # 训练工具
│   └── my_train_query.py          # 训练脚本
├── weight/                        # SD模型权重目录
├── my_test_query.py               # 测试脚本
└── requirements.txt               # Python依赖
```

## 环境安装 | Installation

### 1. 创建Conda环境

```bash
conda create -n sdfsmis python=3.10 -y
conda activate sdfsmis
```

### 2. 安装PyTorch依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 处理SD权重

```python
python scripts/model_weight_preprocess.py
```

### 训练

使用 `scripts/my_train_query.sh` 脚本启动训练：

```bash
bash scripts/my_train_query.sh
```

### 测试

使用提供的测试脚本进行推理：

```bash
python my_test_query.py
```

## 引用

如果您发现这个项目对研究有用，请考虑引用：

```bibtex
@misc{li2026sdfsmisadaptingstablediffusion,
      title={SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation}, 
      author={Meihua Li and Yang Zhang and Weizhao He and Hu Qu and Yisong Li},
      year={2026},
      eprint={2604.03134},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.03134}, 
}
```

## 致谢

本代码基于以下开源项目修改而来：
- [**DiffewS**](https://github.com/aim-uofa/DiffewS)
- [**Marigold**](https://github.com/prs-eth/Marigold)

感谢这些优秀的工作为本次研究提供了基础。

## 许可证

本项目遵循相应的开源许可证。
