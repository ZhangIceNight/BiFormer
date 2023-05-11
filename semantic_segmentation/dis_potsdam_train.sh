MODEL=upernet.biformer_small

OUTPUT_DIR=/home/wjzhang/workspace/results/biformer/seg

CONFIG_DIR=configs/potsdam
CONFIG=${CONFIG_DIR}/${MODEL}.py

DATASET=potsdam

NOW=$(date '+%m-%d-%H:%M:%S')
WORK_DIR=${OUTPUT_DIR}/${MODEL}/${DATASET}/${NOW}
CKPT=/home/wjzhang/workspace/codes/BiFormer/biformer_small_best.pth

python -m torch.distributed.launch --nproc_per_node=2 --master_port=25646 train.py ${CONFIG} \
            --launcher="pytorch" \
            --work-dir=${WORK_DIR} \
            --options model.pretrained=${CKPT} \
            
# python -u train.py ${CONFIG} \
#             --launcher="none" \
#             --work-dir=${WORK_DIR} \
            
