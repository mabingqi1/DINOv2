# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms
from monai import transforms as mntransforms

from .transforms import (
    GaussianBlur,
    ZscoreNormWithOptionClip,
    make_normalize_transform,
    RandResizedCrop
)


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size,
        local_crops_size,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # # random resized crop and flip
        # self.geometric_augmentation_global = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(
        #             global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
        #         ),
        #         transforms.RandomHorizontalFlip(p=0.5),
        #     ]
        # )

        # self.geometric_augmentation_local = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(
        #             local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
        #         ),
        #         transforms.RandomHorizontalFlip(p=0.5),
        #     ]
        # )

        # # color distorsions / blurring
        # color_jittering = transforms.Compose(
        #     [
        #         transforms.RandomApply(
        #             [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        #             p=0.8,
        #         ),
        #         transforms.RandomGrayscale(p=0.2),
        #     ]
        # )

        # global_transfo1_extra = GaussianBlur(p=1.0)

        # global_transfo2_extra = transforms.Compose(
        #     [
        #         GaussianBlur(p=0.1),
        #         transforms.RandomSolarize(threshold=128, p=0.2),
        #     ]
        # )

        # local_transfo_extra = GaussianBlur(p=0.5)

        ## MONAI geometric augmentation for global crops
        self.geometric_augmentation_global = transforms.Compose(
            [
                RandResizedCrop(size=global_crops_size, 
                                scale=self.global_crops_scale, 
                                interpolation="bicubic"
                                ),
                mntransforms.RandFlip(
                    spatial_axis=1,  # Horizontal flip
                    prob=0.5,
                ),
            ]
        )

        # MONAI geometric augmentation for local crops
        self.geometric_augmentation_local = mntransforms.Compose(
            [
                RandResizedCrop(size=local_crops_size, 
                                scale=self.local_crops_scale, 
                                interpolation="bicubic"
                                ),
                mntransforms.RandFlip(
                    spatial_axis=1,  # Horizontal flip
                    prob=0.5,
                ),
            ]
        )

        # MONAI color distortions / grayscale
        color_jittering = transforms.Compose(
            [
                mntransforms.RandAdjustContrast(
                    prob=0.8,
                    gamma=(0.6, 1.4),  # Simulate brightness/contrast (0.4 range)
                ),
                mntransforms.RandHistogramShift(
                    prob=0.8,
                    num_control_points=10,  # Simulate hue/saturation
                ),
                mntransforms.Rand2DElastic(
                    prob=0.2,  # Simulate grayscale effect indirectly
                    spacing=(30, 30),
                    magnitude_range=(0, 0.1),
                ),
            ]
        )

        # MONAI Gaussian blur for global crops (transfo1)
        global_transfo1_extra = mntransforms.RandGaussianSmooth(
            prob=1.0,  # Match original p=1.0
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        )

        # MONAI Gaussian blur and histogram shift for global crops (transfo2)
        global_transfo2_extra = transforms.Compose(
            [
                mntransforms.RandGaussianSmooth(
                    prob=0.1,  # Match original p=0.1
                    sigma_x=(0.5, 1.0),
                    sigma_y=(0.5, 1.0),
                    sigma_z=(0.5, 1.0),
                ),
                mntransforms.RandHistogramShift(
                    prob=0.2,  # Simulate solarize effect
                    num_control_points=10,
                ),
            ]
        )

        # MONAI Gaussian blur for local crops
        local_transfo_extra = mntransforms.RandGaussianSmooth(
            prob=0.5,  # Match original p=0.5
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        )

        # normalization
        self.normalize = transforms.Compose(
            [
                ZscoreNormWithOptionClip(clip=True, 
                                        clip_min_value=-1024, 
                                        clip_max_value=2048),
                # transforms.ToTensor(),
                # make_normalize_transform(),

            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, 
                                                   global_transfo1_extra, 
                                                   self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, 
                                                   global_transfo2_extra, 
                                                   self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, 
                                                 local_transfo_extra, 
                                                 self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
