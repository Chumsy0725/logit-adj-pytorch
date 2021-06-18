import torchvision.transforms as transforms

# Pre Processing Config for Train Dataset
TRAIN_TRANSFORMS = {
    "cifar10": transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
    "cifar100": transforms.Compose([]),
    "cifar10-lt": transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
    "cifar100-lt": transforms.Compose([]),
}

# Pre Processing Config for Test Dataset
TEST_TRANSFORMS = {
    "cifar10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
    "cifar100": transforms.Compose([]),
    "cifar10-lt": transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
    "cifar100-lt": transforms.Compose([]),
}
