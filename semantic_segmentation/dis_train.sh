MODEL=upernet.biformer_small
OUTPUT_DIR=/home/h3c/workspace/results/biformer/seg

CONFIG_DIR=configs/ade20k
CONFIG=${CONFIG_DIR}/${MODEL}.py

NOW=$(date '+%m-%d-%H:%M:%S')
WORK_DIR=${OUTPUT_DIR}/${MODEL}/${NOW}
CKPT=/home/h3c/workspace/codes/BiFormer/biformer_small_best.pth

python -m torch.distributed.launch --nproc_per_node=2 --master_port=25642 train.py ${CONFIG} \
            --launcher="pytorch" \
            --work-dir=${WORK_DIR} \
            --options model.pretrained=${CKPT} \
            
# python -u train.py ${CONFIG} \
#             --launcher="none" \
#             --work-dir=${WORK_DIR} \
            