export CUDA_VISIBLE_DEVICES=0,1

python -u external_src/DUnet/eval.py \
--val_image_path data_split/atlas/traintest/testing/test_scans.txt \
--load_weight pretrained_models/dunet/dunet_atlas_traintest.hdf5 \
--log_path external_src/DUnet/val_result.txt \
--visual_path external_src/DUnet/visual_sl \
--small_lesion_only \
--save_path external_src/DUnet/outputs