

export PYTHONPATH=.
torchrun --nproc_per_node=1 dinov2/run/eval/knn.py \
    --config-file /home/coder/dinov2/dinov2/configs/eval/vitl14_pretrain.yaml \
    --pretrained-weights /home/coder/dinov2/output_dir/eval/training_19999/teacher_checkpoint.pth \
    --output-dir /home/coder/dinov2/output_dir/eval/training_19999/knn \
    --train-dataset ImageNet:split=TRAIN:root="/yinghepool/yinghe/Public_data/ImageNet1K":extra="/yinghepool/yinghe/Public_data/ImageNet1K/metadata" \
    --val-dataset ImageNet:split=VAL:root="/yinghepool/yinghe/Public_data/ImageNet1K":extra="/yinghepool/yinghe/Public_data/ImageNet1K/metadata" \
    --batch-size 128 \
    # --epochs 10