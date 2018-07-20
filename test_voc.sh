CUDA_VISIBLE_DEVICES=6 python eval.py \
--dataset 'VOC' \
--backbone 'drn_d_39' \
--image_size 300 \
--trained_model 'weights/VOC_drn_d_39_batch32_lr0.001_ori/VOC.pth'
