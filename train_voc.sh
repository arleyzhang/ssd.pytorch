DATASET_NAME="VOC"
BACKBONE_MODEL="drn_d_39"
# BACKBONE_MODEL_NAME="vgg16_reducedfc.pth"
# BACKBONE_MODEL_NAME="vgg16-397923af.pth"
# BACKBONE_MODEL_NAME="drn_d_22-4bd2f8ea.pth"
BACKBONE_MODEL_NAME="drn_d_38-eebb45f0.pth"
BATCH_SIZE=32
LR=0.001
IMAGE_SIZE=300
JOB_DIR="weights/${DATASET_NAME}_${BACKBONE_MODEL}_batch${BATCH_SIZE}_lr${LR}_ori"
LOG_NAME="${JOB_DIR}/${BACKBONE_MODEL}_batch${BATCH_SIZE}_resume12w"
RESUME_MODEL="${JOB_DIR}/VOC.pth"

if [ ! -d $JOB_DIR ]; then
  mkdir -p $JOB_DIR
fi

# copy this shell script to the save_dir
cp -f $BASH_SOURCE $JOB_DIR


CUDA_VISIBLE_DEVICES=3,4,5,6 python train.py \
--dataset "VOC" \
--dataset_root "/home/yanleizhang/data/VOCdevkit" \
--backbone "$BACKBONE_MODEL" \
--basenet "$BACKBONE_MODEL_NAME" \
--batch_size $BATCH_SIZE \
--lr $LR \
--num_workers 48 \
--save_folder $JOB_DIR \
--image_size $IMAGE_SIZE \
--visdom True \
--resume $RESUME_MODEL \
--start_iter 120000 \
2>&1 | tee "${LOG_NAME}.log"
