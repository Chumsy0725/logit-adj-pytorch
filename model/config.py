import torch
import torch.nn as nn
import torchvision.transforms as transforms
import flags
from model.arch.resnet import Resnet32
from model.utils import BasicLRScheduler
from torch.optim.lr_scheduler import MultiStepLR

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 200
learning_rate = 0.1

# Batch Sizes
train_batch_size = flags.get_flag("train_batch_size")
test_batch_size = flags.get_flag("test_batch_size")

# Model Definition
# Note that this is not used in lightning module and a different implementation is utilized.
model = Resnet32().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

# Learning Rate Scheduler
# lr_scheduler = BasicLRScheduler(optimizer, learning_rate)
lr_scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

# Save path for the model.
model_save_path = "."
