## Introduction

This repo contains the official PyTorch implementation of our paper **Latent Feature Self-distillation with Task Dual Decoupling for Few-Shot Object Detection and Instance Segmentation**.

LFSD introduces a novel approach to few-shot object detection by combining:
- **SAIA (Self-Adaptive Instance Attention)**: A deformable transformer-based feature enhancement module
- **Dual-Stream Decoupling**: Orthogonal feature learning with prototype memory bank
- **Self-Distillation**: Knowledge transfer from base classes to novel classes

## Quick Start

**1. Check Requirements**
* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.6 & [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch version
* CUDA 10.1, 10.2, or 11.0+
* GCC >= 4.9

**2. Build LFSD**

* Clone Code
  ```bash
  git clone https://github.com/qiwang-GZU/LFSD.git
  cd LFSD
  ```

* Create a virtual environment (recommended)
  ```bash
  conda create -n lfsd python=3.8 -y
  conda activate lfsd
  ```

* Install PyTorch 1.9.0 with CUDA 11.1 (or your preferred version)
  ```bash
  pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
  ```
  
  For other PyTorch/CUDA versions, check [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/).

* Install Detectron2
  ```bash
  python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
  ```
  
  For other versions, check [Detectron2 Installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

* Install mmcv-full (required for SAIA module)
  ```bash
  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
  ```
  
  For other versions, check [MMCV Installation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

* Install mmdetection
  ```bash
  pip install mmdet==2.22.0
  ```

* Install other requirements
  ```bash
  pip install -r requirements.txt
  ```

**3. Prepare Data and Weights**

* Data Preparation

  We evaluate our models on two datasets for both FSOD and G-FSOD settings:

  | Dataset | Size | GoogleDrive | BaiduYun | Note |
  |:---:|:---:|:---:|:---:|:---:|
  |VOC2007| 0.8G |[download](https://drive.google.com/file/d/1BcuJ9j9Mtymp56qGSOfYxlXN4uEVyxFm/view?usp=sharing)|[download](https://pan.baidu.com/s/1kjAmHY5JKDoG0L65T3dK9g)| - |
  |VOC2012| 3.5G |[download](https://drive.google.com/file/d/1NjztPltqm-Z-pG94a6PiPVP4BgD8Sz1H/view?usp=sharing)|[download](https://pan.baidu.com/s/1DUJT85AG_fqP9NRPhnwU2Q)| - |
  |vocsplit| <1M |[download](https://drive.google.com/file/d/1BpDDqJ0p-fQAFN_pthn2gqiK5nWGJ-1a/view?usp=sharing)|[download](https://pan.baidu.com/s/1518_egXZoJNhqH4KRDQvfw)| refer from [TFA](https://github.com/ucbdrive/few-shot-object-detection#models) |
  |COCO| ~19G | - | - | download from [official](https://cocodataset.org/#download)|
  |cocosplit| 174M |[download](https://drive.google.com/file/d/1T_cYLxNqYlbnFNJt8IVvT7ZkWb5c0esj/view?usp=sharing)|[download](https://pan.baidu.com/s/1NELvshrbkpRS8BiuBIr5gA)| refer from [TFA](https://github.com/ucbdrive/few-shot-object-detection#models) |

  Unzip the downloaded data-source to `datasets` and put it into your project directory:
  ```
  LFSD/
  ├── datasets/
  │   ├── coco/
  │   │   ├── trainval2014/
  │   │   ├── val2014/
  │   │   └── annotations/
  │   ├── cocosplit/
  │   ├── VOC2007/
  │   ├── VOC2012/
  │   └── vocsplit/
  ├── lfsd/
  ├── tools/
  └── ...
  ```

* Weights Preparation

  We use the ImageNet pretrain weights to initialize our model:
  
  | Model | GoogleDrive | BaiduYun |
  |:---:|:---:|:---:|
  | ResNet-101 | [download](https://drive.google.com/file/d/1rsE20_fSkYeIhFaNU04rBfEDkMENLibj/view?usp=sharing) | [download](https://pan.baidu.com/s/1IfxFq15LVUI3iIMGFT8slw) |
  
  The extract code for all BaiduYun links is **0000**

**4. Training and Evaluation**

For ease of training and evaluation, we provide integrated scripts that handle the full pipeline including base pre-training and novel fine-tuning.

* **VOC Dataset**

  To reproduce results on VOC, where `SPLIT_ID` must be `1`, `2`, or `3`:
  
  ```bash
  # Full pipeline: base training + novel fine-tuning
  bash run_voc.sh EXP_NAME SPLIT_ID
  
  # Example: Train on split 1
  bash run_voc.sh lfsd_voc 1
  ```

* **COCO Dataset**

  To reproduce results on COCO:
  
  ```bash
  # Full pipeline: base training + novel fine-tuning
  bash run_coco.sh EXP_NAME
  
  # Example
  bash run_coco.sh lfsd_coco
  ```

* **Step-by-Step Training**

  If you prefer manual control, follow these steps:

  1. **Base Pre-training** (train on base classes only):
     ```bash
     python train_net.py --num-gpus 2 \
         --config-file configs/voc/base1.yaml \
         MODEL.WEIGHTS /path/to/ImageNet/R-101.pkl \
         OUTPUT_DIR checkpoints/voc/lfsd/base1
     ```

  2. **Model Surgery** (prepare weights for fine-tuning):
     ```bash
     python tools/model_surgery.py --dataset voc --method randinit \
         --src-path checkpoints/voc/lfsd/base1/model_final.pth \
         --save-dir checkpoints/voc/lfsd/base1
     ```

  3. **Novel Fine-tuning** (with few-shot data):
     ```bash
     # Generate config for specific shot setting
     python tools/create_config.py --dataset voc --config_root configs/voc \
         --shot 10 --seed 0 --setting gfsod --split 1
     
     # Fine-tune
     python train_net.py --num-gpus 2 \
         --config-file configs/voc/lfsd_gfsod_novel1_10shot_seed0.yaml \
         MODEL.WEIGHTS checkpoints/voc/lfsd/base1/model_reset_surgery.pth \
         OUTPUT_DIR checkpoints/voc/lfsd/novel1/10shot_seed0 \
         TEST.PCB_MODELPATH /path/to/resnet101-5d3b4d8f.pth
     ```

  4. **Evaluation Only**:
     ```bash
     python train_net.py --num-gpus 2 --eval-only \
         --config-file configs/voc/lfsd_gfsod_novel1_10shot_seed0.yaml \
         MODEL.WEIGHTS checkpoints/voc/lfsd/novel1/10shot_seed0/model_final.pth
     ```

* **Key Configuration Options**

  | Option | Description | Default |
  |--------|-------------|---------|
  | `MODEL.BACKBONE.WITHSAIA` | Enable SAIA attention module | `False` |
  | `MODEL.BACKBONE.SAIA_ALPHA` | SAIA feature fusion weight | `0.3` |
  | `MODEL.ROI_HEADS.MEMORY` | Enable prototype memory | `False` |
  | `MODEL.ROI_HEADS.SEMANTIC` | Enable semantic distillation | `False` |
  | `DATASETS.TWO_STREAM` | Enable dual-stream training | `False` |
  | `TEST.PCB_ENABLE` | Enable Prototype Calibration Block | `True` |

**5. Extract Results**

After running multiple seeds, extract averaged results:

```bash
python tools/extract_results.py \
    --res-dir checkpoints/voc/lfsd/novel1/tfa-like \
    --times 1 2 3 4 5 6 7 8 9 10 \
    --shot-list 1 2 3 5 10
```

## Acknowledgement

This repo is developed based on [DeFRCN](https://github.com/er-muyue/DeFRCN), [TFA](https://github.com/ucbdrive/few-shot-object-detection), and [Detectron2](https://github.com/facebookresearch/detectron2). We thank the authors for their excellent work.


## License

This project is released under the [Apache 2.0 License](LICENSE).
