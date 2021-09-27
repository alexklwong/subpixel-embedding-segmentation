export CUDA_VISIBLE_DEVICES=1

python -u external_src/KiU-Net/test.py \
--loaddirec "external_src/KiU-Net/trained_model/kiunet.pth" \
--val_dataset "external_src/KiU-Net/data/validation" \
--direc 'external_src/KiU-Net/trained_model/outputs' \
--batch_size 1 \
--modelname "kiunet" \
--visual_paths "external_src/KiU-Net/trained_model/visual/small" \
--small_lesion_only \
--small_lesion_idx "data/atlas_lesion_segmentation_small_lesions/validation_small_lesion_idxs.txt" \
--n_samples 27 \
--n_classes 2 \
--log_path "external_src/KiU-Net/trained_model/val_result.txt" \
--image_size 128 128
