import sys

sys.path.append('.')

import torch
from flnode.pipeline.algo import Algo
from common.utils import Mapping
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
import numpy as np

from monai.networks.nets import BasicUNet
from monai.transforms import (Compose, LoadImageD, AddChannelD, AddCoordinateChannelsD, Rand3DElasticD, SplitChannelD,
                              DeleteItemsD, ScaleIntensityRangeD, ConcatItemsD, RandSpatialCropD, ToTensorD,
                              CastToTypeD)
from monai.data import Dataset, DataLoader
from monai.metrics import compute_meandice

from torch import nn, sigmoid
from flnode.pipeline.monaiopener import MonaiOpener

import numpy as np
from pathlib import Path

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"


class MonaiAlgo(Algo):
    def __init__(self, logger):
        self.model = None
        self.loss = None
        self.optimizer = None
        self.epochs = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.metric = None

        self.logger = logger
        self._initialize_attrs()

    def train(self):
        # Set deterministic training for reproducibility
        # set_determinism(seed=0)
        device = torch.device(DEVICE)
        val_interval = 1
        epoch_loss_values = list()
        val_mean_dice_scores = list()

        self.model.to(device)
        for epoch in range(self.epochs):
            self.logger.info(f"epoch {epoch + 1}/{self.epochs}")
            self.model.train()
            epoch_loss = 0
            for batch_idx, (data_batch) in enumerate(self.train_loader):
                data, target = data_batch['img'].to(DEVICE), data_batch['seg'].to(DEVICE)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= (batch_idx + 1) 
            epoch_loss_values.append(epoch_loss) 
            self.logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                self.logger.info('Validating...')
                self.model.eval()
                with torch.no_grad():
                    val_metrics = []
                    for batch_idx, (data_batch) in enumerate(self.val_loader):
                        data, target = data_batch['img'].to(DEVICE), data_batch['seg'].to(DEVICE)

                        output_logits = sliding_window_inference(data,
                                                                 sw_batch_size=2,
                                                                 roi_size=(128, 128, 128),
                                                                 predictor=self.model,
                                                                 overlap=0.25,
                                                                 do_sigmoid=False)
                        output = torch.sigmoid(output_logits)

                        # loss = self.criterion(output, target)
                        val_metrics.append(self.metric(output, target, include_background=False).cpu().numpy())
                mean_val_metric = float(np.mean(val_metrics))
                self.logger.info(f"epoch {epoch + 1} average validation dice score: {mean_val_metric:.4f}")
                val_mean_dice_scores.append(mean_val_metric)

        checkpoint = Mapping()
        checkpoint.update(epoch=epoch+1, weights=self.model.state_dict(), val_mean_dice_scores=val_mean_dice_scores, train_loss_values=epoch_loss_values)
        return checkpoint

    def load_model(self, modelFile):
        path = modelFile
        self.model.load_state_dict(torch.load(path))

    def save_model(self, model, path):
        pass
        # json.dump(model, path)

    def predict(self, headModelFile):
        set_determinism(seed=0)
        device = torch.device(DEVICE)
        self.load_model(headModelFile)
        self.model.to(device)
        self.model.eval()

        dice_scores = []
        with torch.no_grad():
            for data_batch in self.test_loader:
                data, target = data_batch['img'].to(DEVICE), data_batch['seg'].to(DEVICE)

                output_logits = sliding_window_inference(data,
                                                         sw_batch_size=2,
                                                         roi_size=(128, 128, 128),
                                                         predictor=self.model,
                                                         overlap=0.25,
                                                         do_sigmoid=False)
                output = torch.sigmoid(output_logits)

                # loss = self.criterion(output, target)
                dice_scores.append(self.metric(output, target, include_background=False).cpu().numpy().tolist()[0][0])
        test_report = Mapping()
        test_report.update(test_dice_scores=dice_scores, target_names='dice', digits=4)

        return test_report

    def _initialize_attrs(self, frac_val=0.2, frac_test=0.2, dataset_name='CROMIS4AD_READY'):
        cwd = Path.cwd()
        data_path = cwd.parent / 'data_provider'/ 'FLIP'
        data_dir = data_path / dataset_name
        if Path(data_dir).exists():
            self.logger.info(f'dataset available in path {data_dir}')
        else:
            raise FileNotFoundError(f"dataset not reachable")

        mo = MonaiOpener(data_dir)
        mo.data_summary(self.logger)

        train, val, test = mo.get_x_y(frac_val, frac_test)
        self.logger.info(f"Training count: {len(train)}, Validation count: {len(val)}, Test count: {len(test)}")

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

        train_dataset = Dataset(train, transform=train_transforms)
        val_dataset = Dataset(val, transform=val_transforms)
        test_dataset = Dataset(test, transform=val_transforms)

        ######################################## MARK FIX FOR FUTURE EVENTUAL MULTIPROCESSING ERRORS ###############################
        # sharing_strategy = "file_system"
        # torch.multiprocessing.set_sharing_strategy(sharing_strategy)

        # def set_worker_sharing_strategy(worker_id: int) -> None:
        #     torch.multiprocessing.set_sharing_strategy(sharing_strategy)

        # self.train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, worker_init_fn=set_worker_sharing_strategy)
        # self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, worker_init_fn=set_worker_sharing_strategy)
        # self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, worker_init_fn=set_worker_sharing_strategy)
        #############################################################################################################################

        self.train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        # model initiliatization


        class BasicUnetSigmoid(nn.Module):
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


        self.model = BasicUnetSigmoid().to(DEVICE)

        # model loss function
        self.loss = DiceLoss(include_background=False)

        # model metric
        self.metric = compute_meandice

        # model optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0, amsgrad=True)

        # number of epochs
        self.epochs = 1