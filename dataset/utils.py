from dataset import *
import flags

_FLAG_DATASET_MAPPINGS = {
    "cifar10": CIFAR10Dataset,
    "cifar100": CIFAR100Dataset,
    "cifar10-lt": CIFAR10LTNPZDataset,
    "cifar100-lt": CIFAR100LTNPZDataset,
}


# Returns the class of the dataset.
def fetch_dataset_from_flags():
    identifier = flags.get_flag("dataset")
    return _FLAG_DATASET_MAPPINGS[identifier]
