export CUDA_VISIBLE_DEVICES=0

python -u external_src/KiU-Net/train.py \
--train_dataset "external_src/KiU-Net/data/train" \
--val_dataset "external_src/KiU-Net/data/validation" \
--direc "external_src/KiU-Net/trained_model/trial2" \
--batch_size 4 \
--epoch 20 \
--save_freq 1 \
--modelname "kiunet" \
--learning_rate 0.0001