import sys

sys.path.append('.')
import torch as t

from monai.networks.nets import BasicUNet
from monai.transforms import (Compose, LoadImageD, AddChannelD, AddCoordinateChannelsD, Rand3DElasticD, SplitChannelD,
                              DeleteItemsD, ScaleIntensityRangeD, ConcatItemsD, RandSpatialCropD, ToTensorD,
                              CastToTypeD)
from monai.data import Dataset, DataLoader
from monai.metrics import compute_meandice

from torch import nn, sigmoid, cuda
from flnode.pipeline.monaiopener import MonaiOpener
from flnode.pipeline.monaialgo import MonaiAlgo

import numpy as np
from pathlib import Path
import logging

logging.basicConfig(format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.NOTSET)

if cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"


def instantiateMonaiAlgo(frac_val=0.1, frac_test=0.1, frac_initial_dataset=1, dataset_name='CROMIS4AD_READY'):
    cwd = Path.cwd()
    data_path = cwd.parent / 'data_provider'/ 'FLIP'
    data_dir = data_path / dataset_name
    if Path(data_dir).exists():
        logger.info(f'dataset available in path {data_dir}')
    else:
        raise FileNotFoundError(f"dataset not reachable")

    mo = MonaiOpener(data_dir)
    mo.data_summary(logger)

    train, val, test = mo.get_x_y(frac_val, frac_test, frac_initial_dataset)
    logger.info(f"Training count: {len(train)}, Validation count: {len(val)}, Test count: {len(test)}")

    train_transforms = Compose(
        [LoadImageD(keys=['img', 'seg'], reader='NiBabelReader', as_closest_canonical=False),
     AddChannelD(keys=['img', 'seg']),
     AddCoordinateChannelsD(keys=['img'], spatial_channels=(1, 2, 3)),
     Rand3DElasticD(keys=['img', 'seg'], sigma_range=(1, 3), magnitude_range=(-10, 10), prob=0.5,
                    mode=('bilinear', 'nearest'),
                    rotate_range=(-0.34, 0.34),
                    scale_range=(-0.1, 0.1), spatial_size=None),
     SplitChannelD(keys=['img']),
     ScaleIntensityRangeD(keys=['img_0'], a_min=-15, a_max=100, b_min=-1, b_max=1, clip=True),
     ConcatItemsD(keys=['img_0', 'img_1', 'img_2', 'img_3'], name='img'),
     DeleteItemsD(keys=['img_0', 'img_1', 'img_2', 'img_3']),
     RandSpatialCropD(keys=['img', 'seg'], roi_size=(128, 128, 128), random_center=True, random_size=False),
     ToTensorD(keys=['img', 'seg'])
     ])

    val_transforms = Compose(
    [LoadImageD(keys=['img', 'seg'], reader='NiBabelReader', as_closest_canonical=False),
     AddChannelD(keys=['img', 'seg']),
     AddCoordinateChannelsD(keys=['img'], spatial_channels=(1, 2, 3)),
     SplitChannelD(keys=['img']),
     ScaleIntensityRangeD(keys=['img_0'], a_min=-15, a_max=100, b_min=-1, b_max=1, clip=True),
     ConcatItemsD(keys=['img_0', 'img_1', 'img_2', 'img_3'], name='img'),
     DeleteItemsD(keys=['img_0', 'img_1', 'img_2', 'img_3'], ),
     CastToTypeD(keys=['img'], dtype=np.float32),
     ToTensorD(keys=['img', 'seg'])
     ])

    # monai algorithm object
    ma = MonaiAlgo()

    train_dataset = Dataset(train, transform=train_transforms)
    val_dataset = Dataset(val, transform=val_transforms)
    test_dataset = Dataset(test, transform=val_transforms)

    ma.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    ma.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    ma.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # model initiliatization


    class BasicUnet(nn.Module):
        def __init__(self, num_classes=1):
            super().__init__()
            self.net = BasicUNet(dimensions=3,
                                 features=(32, 32, 64, 128, 256, 32),
                                 in_channels=4,
                                 out_channels=num_classes
                                 )

        def forward(self, x, do_sigmoid=True):
            logits = self.net(x)
            if do_sigmoid:
                return sigmoid(logits)
            else:
                return logits


    ma.model = BasicUnet().to(DEVICE)


    # model loss function
    def mean_dice(output, target, average=True):
        # empty labels return a dice score of NaN - replace with 0
        dice_per_batch = compute_meandice(output, target, include_background=False)
        dice_per_batch[dice_per_batch.isnan()] = 0
        if average:
            return dice_per_batch.mean().cpu()
        else:
            return dice_per_batch.cpu()


    ma.loss = mean_dice

    # model metric
    ma.metric = compute_meandice

    # model optimizer
    ma.optimizer = t.optim.Adam(ma.model.parameters(), lr=1e-5, weight_decay=0, amsgrad=True)

    # number of epochs
    ma.epochs = 1

    return ma

if __name__ == '__main__':
    ma = instantiateMonaiAlgo()


