export CUDA_VISIBLE_DEVICES=0

python -u external_src/CLCI-Net/eval.py \
--data_file_path data/train.h5 \
--pretrained_weight_file pretrained_models/clci-net/clcinet_atlas_traintest.h5 \
--input_shape 224 176 1 \
--save_path external_src/CLCI-Net/outputs \
--data_order_path data/data_order.txt