# torchrun --nproc_per_node=1 dinov2/run/eval/linear.py \
#     --config-file /home/coder/dinov2/dinov2/configs/eval/vitl14_pretrain.yaml \
#     --pretrained-weights /home/coder/dinov2/dinov2_vitl14_pretrain.pth \
#     --output-dir /home/coder/dinov2/output_dir/eval/training_19999/linear \
#     --train-dataset ImageNet:split=TRAIN:root="/yinghepool/yinghe/Public_data/ImageNet1K":extra="/yinghepool/yinghe/Public_data/ImageNet1K/metadata" \
#     --val-dataset ImageNet:split=VAL:root="/yinghepool/yinghe/Public_data/ImageNet1K":extra="/yinghepool/yinghe/Public_data/ImageNet1K/metadata" \
#     --batch-size 128 \
#     --epochs 10
export PYTHONPATH=.
torchrun --nproc_per_node=1 dinov2/run/eval/linear.py \
    --config-file /home/coder/dinov2/dinov2/configs/eval/vitl16_yh.yaml \
    --pretrained-weights /home/coder/dinov2/dinov2_vitl14_pretrain.pth \
    --output-dir /home/coder/dinov2/output_dir/eval/training_19999/linear \
    --train-dataset  \
    --val-dataset  \
    --batch-size 128 \
    --epochs 10