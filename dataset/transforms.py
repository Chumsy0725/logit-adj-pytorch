import torchvision.transforms as transforms

# data lie between (-0.5 , 0.5 )
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[1.0, 1.0, 1.0])

# Pre Processing Config for Train Dataset
TRAIN_TRANSFORMS = {
    "cifar10": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize
    ]),
    "cifar100": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize
    ]),
    "cifar10-lt": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        normalize
    ]),
    "cifar100-lt": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        normalize]),
}

# Pre Processing Config for Test Dataset
TEST_TRANSFORMS = {
    "cifar10": transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]),
    "cifar100": transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]),

    "cifar10-lt": normalize,
    "cifar100-lt": normalize,
}
