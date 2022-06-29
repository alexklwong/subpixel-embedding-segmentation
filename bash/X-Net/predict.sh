export CUDA_VISIBLE_DEVICES=0

python -u external_src/X-Net/eval.py \
--data_file_path data/train.h5 \
--pretrained_weight_file pretrained_models/x-net/xnet_atlas_traintest.h5 \
--input_shape 224 192 1 \
--save_path external_src/X-Net/outputs \
--data_order_path data/data_order.txt