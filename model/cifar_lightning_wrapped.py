import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from dataset import CIFAR10LTNPZDataset

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Hyper-parameters
num_epochs = 11
learning_rate = 0.001


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(pl.LightningModule):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        tensorboard_logs = {'train_loss': loss}
        # use key 'log'
        return {"loss": loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        # CIFAR-10 dataset
        train_dataset = CIFAR10LTNPZDataset(root='data',
                                            train=True,
                                            transform=transform)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=100,
                                                   shuffle=True)
        return train_loader

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        tensorboard_logs = {'val_loss': loss}
        # use key 'log'
        return {"Val_loss": loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        # CIFAR-10 dataset
        test_dataset = CIFAR10LTNPZDataset(root='data',
                                           train=False)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=100,
                                                  shuffle=False)

        return test_loader


trainer = Trainer(gpus=1, max_epochs=2, fast_dev_run=False)
model = ResNet(ResidualBlock, [5, 5, 5])
trainer.fit(model)
