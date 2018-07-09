DATASET_NAME="VOC"
BACKBONE_MODEL="vgg16"
# BACKBONE_MODEL_NAME="vgg16_reducedfc.pth"
# BACKBONE_MODEL_NAME="vgg16-397923af.pth"
BACKBONE_MODEL_NAME="vgg16_batch448_lrIni0.01.pth.tar"
BATCH_SIZE=35
LR=0.0001
JOB_DIR="weights/${DATASET_NAME}_${BACKBONE_MODEL}_batch${BATCH_SIZE}_myvgg_lr${LR}"
LOG_NAME="${JOB_DIR}/${BACKBONE_MODEL}_batch${BATCH_SIZE}"
# RESUME_MODEL="${JOB_DIR}/.pth"

if [ ! -d $JOB_DIR ]; then
  mkdir -p $JOB_DIR
fi

# copy this shell script to the save_dir
cp -f $BASH_SOURCE $JOB_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python train.py \
--dataset "VOC" \
--dataset_root "/home/yanleizhang/data/VOCdevkit" \
--backbone "$BACKBONE_MODEL" \
--basenet "$BACKBONE_MODEL_NAME" \
--batch_size $BATCH_SIZE \
--lr $LR \
--num_workers 30 \
--save_folder $JOB_DIR \
--visdom False \
2>&1 | tee "${LOG_NAME}.log"
