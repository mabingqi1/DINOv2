# python dinov2/run/train/train.py \
#     --nodes 1 \
#     --ngpus 2 \
#     --config-file dinov2/configs/train/vitl16_short.yaml \
#     --output-dir "./output_dir" \
#     train.dataset_path=ImageNet:split=TRAIN:root="/yinghepool/yinghe/Public_data/imagenet1k/OpenDataLab___ImageNet-1K/raw/ImageNet-1K":extra="/yinghepool/yinghe/Public_data/imagenet1k/OpenDataLab___ImageNet-1K/raw/ImageNet-1K"
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1,0,3
torchrun --nproc_per_node=3 dinov2/train/train.py  \
    --config-file dinov2/configs/train/vitl16_yh.yaml \
    --output-dir "./output_dir/vitl16-1152_yh_512" \
    # train.dataset_path=ImageNet:split=TRAIN:root="/yinghepool/yinghe/Public_data/ImageNet1K":extra="/yinghepool/yinghe/Public_data/ImageNet1K/metadata"
    # --config-file dinov2/configs/train/vitl16_yh.yaml \
