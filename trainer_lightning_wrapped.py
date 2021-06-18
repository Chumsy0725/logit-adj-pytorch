from model.config import *
from model.transforms import *
from model.arch.resnet_lightning_wrapped import ResidualBlock, ResNet
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer

# Define any function from functional wrapper to map to torch.nn package implementation, following the same pattern given.
_mappers = {
    criterion: F.cross_entropy,
}


# Runner for Resnet
class Runner(ResNet):

    def __init__(self, dataset, dataset_root):
        self._m_dataset = dataset
        self._m_dataset_root = dataset_root

        super(Runner, self).__init__(ResidualBlock, [5, 5, 5], 10)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        outputs = self(images)
        loss = _mappers[criterion](outputs, labels)

        tensorboard_logs = {'train_loss': loss}
        # use key 'log'
        return {"loss": loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        train_dataset = self._m_dataset(root=self._m_dataset_root,
                                        train=True,
                                        transform=TRAIN_TRANSFORMS[self._m_dataset.get_identifier()],
                                        download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=train_batch_size,
                                                   shuffle=True)
        return train_loader

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        outputs = self(images)
        loss = _mappers[criterion](outputs, labels)
        tensorboard_logs = {'val_loss': loss}
        # use key 'log'
        return {"Val_loss": loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        test_dataset = self._m_dataset(root=self._m_dataset_root,
                                       train=False,
                                       transform=TEST_TRANSFORMS[self._m_dataset.get_identifier()])

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=test_batch_size,
                                                  shuffle=False)

        return test_loader


def run(dataset, root):
    trainer = Trainer(gpus=1, max_epochs=num_epochs, fast_dev_run=False)
    model = Runner(dataset, root)
    trainer.fit(model)
