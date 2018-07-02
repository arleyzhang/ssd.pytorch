BACKBONE_MODEL="vgg16"
BATCH_SIZE=32
JOB_DIR="weights/${BACKBONE_MODEL}_batch${BATCH_SIZE}"
LOG_NAME="${JOB_DIR}/${BACKBONE_MODEL}_batch${BATCH_SIZE}"
# RESUME_MODEL="${JOB_DIR}/.pth"

if [ ! -d $JOB_DIR ]; then
  mkdir -p $JOB_DIR
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python train.py \
--dataset 'VOC' \
--dataset_root '/home/yanleizhang/data/VOCdevkit' \
--batch_size $BATCH_SIZE \
--num_workers 10 \
2>&1 | tee "${LOG_NAME}.log"
