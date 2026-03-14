"""In this python module we define class that handles the dataset:
- DatasetClassifier: a custom Pytorch Dataset for classifing images.
- DataModuleClassifier: a Pytorch Lightning Data Module for classifing the images.
"""

import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from download_utils import (
    download_zip_from_gdrive,
)

import pytorch_lightning as pl
import torch.nn as nn

# Definiamo la classe mancante che gestisce il task finale
class Classifier(pl.LightningModule):
    def __init__(self, input_size=768, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        # Il task finale è una semplice testa lineare sui latent di ViT-Base
        self.model = nn.Linear(input_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

# =====================================================
#
#                 DATASETS DEFINITION
#
# =====================================================
class DatasetClassifier(Dataset):
    """A custom implementation of a Pytorch Dataset.

    Args:
        path : Path
            The path to the data.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
        self.input: torch.Tensor
            The absolute representation the decoder.
        self.labels : torch.Tensor
            The labels of the Dataset.
        self.input_size : int
            The size of the input of the network.
        self.num_classes : int
            The size of the number of classes.
    """

    def __init__(
        self,
        path: Path,
    ):
        self.path: Path = path

        # =================================================
        #                 Get the Data
        # =================================================
        rx_blob = torch.load(self.path, weights_only=True)

        # Retrieve the absolute representation from the receiver
        self.input = rx_blob['absolute']

        # Retrieve the labels
        self.labels = rx_blob['labels']

        del rx_blob

        # =================================================
        #         Get the input and the output size
        # =================================================
        self.input_size = self.input.shape[-1]
        self.num_classes = self.labels.unique().shape[-1]

    def __len__(self) -> int:
        """Returns the length of the Dataset.

        Returns:
            int
            Length of the Dataset.
        """
        return len(self.input)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns in a torch.Tensor format the input and the target.

        Args:
            idx : int
                The index of the wanted row.

        Returns:
            (input_i, l_i) : tuple[torch.Tensor, torch.Tensor]
                The inputs and target as a tuple of tensors.
        """
        # Get the i input
        input_i = self.input[idx]

        # Get the label of the element at idx
        l_i = self.labels[idx]

        return input_i, l_i


# =====================================================
#
#                DATAMODULES DEFINITION
#
# =====================================================
class DataModuleClassifier(LightningDataModule):
    """A custom Lightning Data Module to handle a Pytorch Dataset.

    Args:
        dataset : str
            The name of the dataset.
        rx_enc : str
            The name of the receiver encoder.
        batch_size : int
            The size of a batch. Default 128.
        num_workers : int
            The number of workers. Setting it to 0 means that the data will be
            loaded in the main process. Default 0.

    Attributes:
        The self.<arg_name> version of the arguments documented above.
    """

    def __init__(
        self,
        dataset: str,
        rx_enc: str,
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> None:
        super().__init__()

        self.dataset: str = dataset
        self.rx_enc: str = rx_enc
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers

    def prepare_data(self) -> None:
        """This function prepare the dataset (Download and Unzip).

        Returns:
            None
        """
        from dotenv import dotenv_values

        # Get from the .env file the zip file Google Drive ID
        ID = dotenv_values()['DATA_ID']

        # Download and unzip the data
        download_zip_from_gdrive(ID, name='data', path='data')

        return None

    def setup(self, stage: str = None) -> None:
        """This function setups a DatasetRelativeDecoder for our data.

        Args:
            stage : str
                The stage of the setup. Default None.

        Returns:
            None.
        """
        CURRENT = Path('.')
        GENERAL_PATH: Path = CURRENT / 'data/classification' / self.dataset

        self.train_data = DatasetClassifier(
            path=GENERAL_PATH / 'train' / f'{self.rx_enc}.pt'
        )
        self.test_data = DatasetClassifier(
            path=GENERAL_PATH / 'test' / f'{self.rx_enc}.pt'
        )
        self.val_data = DatasetClassifier(
            path=GENERAL_PATH / 'val' / f'{self.rx_enc}.pt'
        )

        assert (
            self.train_data.input_size == self.test_data.input_size
            and self.train_data.input_size == self.val_data.input_size
        ), 'Input size must match between train, test and val data.'
        assert (
            self.train_data.num_classes == self.test_data.num_classes
            and self.train_data.num_classes == self.val_data.num_classes
        ), 'The number of classes must match between train, test and val data.'

        self.input_size = self.train_data.input_size
        self.num_classes = self.train_data.num_classes
        return None

    def train_dataloader(self) -> DataLoader:
        """The function returns the train DataLoader.

        Returns:
            DataLoader
                The train DataLoader.
        """
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """The function returns the test DataLoader.

        Returns:
            DataLoader
                The test DataLoader.
        """
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """The function returns the validation DataLoader.

        Returns:
            DataLoader
                The validation DataLoader.
        """
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """The function returns the predict DataLoader.

        Returns:
            DataLoader
                The predict DataLoader.
        """
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == '__main__':
    pass
