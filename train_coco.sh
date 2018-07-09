BACKBONE_MODEL="vgg16"
DATASET="COCO"
DATASET_ROOT="/home/yanleizhang/data/coco"
BATCH_SIZE=32
JOB_DIR="weights/${DATASET}_${BACKBONE_MODEL}_batch${BATCH_SIZE}"
LOG_NAME="${JOB_DIR}/${DATASET}_${BACKBONE_MODEL}_batch${BATCH_SIZE}"
# RESUME_MODEL="${JOB_DIR}/.pth"

if [ ! -d $JOB_DIR ]; then
  mkdir -p $JOB_DIR
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python train.py \
--dataset "$DATASET" \
--dataset_root "$DATASET_ROOT" \
--batch_size $BATCH_SIZE \
--num_workers 30 \
--save_folder $JOB_DIR \
2>&1 | tee "${LOG_NAME}.log"
