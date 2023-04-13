import pytorch_lightning as pl
import torch
import numpy as np
import torch.utils.data as td
from config import BATCH_SIZE, NUM_WORKERS


class ContextualRankerData(pl.LightningDataModule):
    def __init__(self, train_data,
                 val_data,
                 test_data,
                 features,
                 batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.features = features
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.test_data = self.test_data.assign(rdm=np.random.random(len(self.test_data))).assign(
            avg=self.train_data["time"].mean())

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = td.TensorDataset(
                torch.from_numpy(self.train_data[self.features].values),
                torch.from_numpy(self.train_data["time"].values)
            )

            self.val_dataset = td.TensorDataset(
                torch.from_numpy(self.val_data[self.features].values),
                torch.from_numpy(self.val_data["time"].values)
            )

        if stage == "test" or stage is None:
            self.test_dataset = td.TensorDataset(
                torch.from_numpy(self.test_data[self.features].values),
                torch.from_numpy(self.test_data[["time", "avg", "rdm"]].values)
            )

    def train_dataloader(self):
        return td.DataLoader(self.train_dataset,
                             batch_size=self.batch_size,
                             shuffle=True,
                             num_workers=self.num_workers)

    def val_dataloader(self):
        return td.DataLoader(self.val_dataset,
                             batch_size=self.batch_size,
                             num_workers=self.num_workers)

    def test_dataloader(self):
        return td.DataLoader(self.test_dataset,
                             batch_size=self.batch_size,
                             num_workers=self.num_workers)
