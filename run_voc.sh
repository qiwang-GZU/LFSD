#!/usr/bin/env bash

#SBATCH -J LFSD-VOC-a800
#SBATCH -o ./logs/log-out-%j-%x.txt
#SBATCH -e ./logs/log-err-%j-%x.txt
#SBATCH --partition=gpu-l20
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1

EXP_NAME=$1
SAVE_DIR=./checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN=/home/dongxy/code/DeFRCN/pretrain/weight/R-101.pkl                          
IMAGENET_PRETRAIN_TORCH=/home/dongxy/code/DeFRCN/pretrain/weight/resnet101-5d3b4d8f.pth  
SPLIT_ID=$2
DIST_URL=$3


# python3 train_net.py --num-gpus 2 --config-file configs/voc/base${SPLIT_ID}.yaml --dist-url ${DIST_URL}   \
#     --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}                                                   \
#            OUTPUT_DIR ${SAVE_DIR}/base${SPLIT_ID}

# python3 tools/model_surgery.py --dataset voc --method randinit                                \
#     --src-path ${SAVE_DIR}/base${SPLIT_ID}/model_final.pth                    \
#     --save-dir ${SAVE_DIR}/base${SPLIT_ID} 

# BASE_WEIGHT=${SAVE_DIR}/base${SPLIT_ID}/model_reset_surgery.pth
BASE_WEIGHT=/home/dongxy/code/LFSD-DCFS/checkpoints/voc/A800-8/base1/model_reset_surgery.pth


for times in 1 2 3 4 5
do
    for seed in 0 
    do
        for shot in 1   
        do
            python3 tools/create_config.py --dataset voc --config_root configs/voc               \
                --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
            CONFIG_PATH=configs/voc/lfsd_gfsod_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
            OUTPUT_DIR=${SAVE_DIR}/lfsd_gfsod_novel${SPLIT_ID}/tfa-like/${times}/${shot}shot_seed${seed}
            python3 train_net.py --num-gpus 2 --config-file ${CONFIG_PATH}   --dist-url ${DIST_URL}  \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
                TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
            rm ${CONFIG_PATH}
            # rm ${OUTPUT_DIR}/model_final.pth
        done
    done
done
# python3 tools/extract_results.py --res-dir ${SAVE_DIR}/lfsd_gfsod_novel${SPLIT_ID}/tfa-like \
# --times 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 --shot-list 1 2 3 5 10 