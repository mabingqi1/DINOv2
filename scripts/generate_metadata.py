from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, 
                       root="/yinghepool/yinghe/Public_data/ImageNet1K", 
                       extra="/yinghepool/yinghe/Public_data/ImageNet1K/metadata"
                       )
    dataset.dump_extra()