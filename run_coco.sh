#!/usr/bin/env bash

#SBATCH -J LFSD-COCO
#SBATCH -o log-out-%j-%x.txt
#SBATCH -e log-err-%j-%x.txt
#SBATCH --partition=gpu-a100-2
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1

EXPNAME=$1
SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN=/home/dongxy/code/DeFRCN/pretrain/weight/R-101.pkl                          
IMAGENET_PRETRAIN_TORCH=/home/dongxy/code/DeFRCN/pretrain/weight/resnet101-5d3b4d8f.pth  
DIST_URL=$2

python3 train_net.py --num-gpus 2 --config-file configs/coco/base.yaml  --dist-url ${DIST_URL}   \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                         \
           OUTPUT_DIR ${SAVEDIR}/base


python3 tools/model_surgery.py --dataset coco --method randinit                        \
    --src-path ${SAVEDIR}/base/model_final.pth                               \
    --save-dir ${SAVEDIR}/base

BASE_WEIGHT=${SAVEDIR}/base/model_reset_surgery.pth


for times in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    for seed in 0 
    do
        for shot in 1 2 3 5 10 30
        do
            python3 tools/create_config.py --dataset coco14 --config_root configs/coco     \
                --shot ${shot} --seed ${seed} --setting 'gfsod'
            CONFIG_PATH=configs/coco/lfsd_gfsod_novel_${shot}shot_seed${seed}.yaml
            OUTPUT_DIR=${SAVEDIR}/lfsd_gfsod_novel/tfa-like/${times}/${shot}shot_seed${seed}
            python3 train_net.py --num-gpus 2 --config-file ${CONFIG_PATH} --dist-url ${DIST_URL}   \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}               \
                    TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
            rm ${CONFIG_PATH}
            rm ${OUTPUT_DIR}/model_final.pth
        done
    done
done
python3 tools/extract_results.py --res-dir ${SAVEDIR}/lfsd_gfsod_novel/tfa-like\
--times 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --shot-list 1 2 3 5 10 30 