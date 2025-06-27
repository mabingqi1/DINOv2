export PYTHONPATH=.
# export CUDA_VISIBLE_DEVICES=1,0,3
torchrun --nproc_per_node=4 --master_port=29505 dinov2/train/train.py  \
    --config-file dinov2/configs/train/yh_swinl16.yaml \
    --output-dir "./output_dir/swinl16_512" \