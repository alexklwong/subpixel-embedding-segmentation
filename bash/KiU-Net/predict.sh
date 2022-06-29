export CUDA_VISIBLE_DEVICES=0

python -u external_src/KiU-Net/test.py \
--loaddirec "pretrained_models/kiu-net/kiunet_atlas_traintest.pth" \
--val_dataset "data/kiunet_validation" \
--direc 'external_src/KiU-Net/outputs' \
--batch_size 1 \
--modelname "kiunet" \
--visual_paths "external_src/KiU-Net/visual_sl" \
--small_lesion_only \
--small_lesion_idx "testing/atlas/traintest/atlas_test_small_lesion_map_indices.txt" \
--n_samples 27 \
--n_classes 2 \
--log_path "external_src/KiU-Net/val_result.txt" \
--image_size 128 128
