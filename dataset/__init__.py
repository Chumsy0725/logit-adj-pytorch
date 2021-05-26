from torch.utils.data import TensorDataset
import numpy as np
import os
import torch


class _CIFARLTNPZDataset(TensorDataset):

    FILE_POSTFIX_DATA = "data"
    FILE_POSTFIX_LABELS = "labels"

    def __init__(self, cifar_prefix: str, root: str, train: bool, transform=None):
        self._m_transform = transform

        file_path_data: str = os.path.join(root, cifar_prefix + "_" + ("train" if train else "test") + "_" + _CIFARLTNPZDataset.FILE_POSTFIX_DATA + ".npz")
        file_path_labels: str = os.path.join(root, cifar_prefix + "_" + ("train" if train else "test") + "_" + _CIFARLTNPZDataset.FILE_POSTFIX_LABELS + ".npz")

        np_data = _CIFARLTNPZDataset._get_np_data_from_file(file_path_data)
        np_labels = _CIFARLTNPZDataset._get_np_data_from_file(file_path_labels)

        tensor_data = torch.Tensor(np_data).to(dtype=torch.float32)
        tensor_labels = torch.Tensor(np_labels).to(dtype=torch.int64)

        super().__init__(tensor_data, tensor_labels)

    @staticmethod
    def _get_np_data_from_file(file_path: str):
        loaded_file_data = np.load(file_path, allow_pickle=True)
        return loaded_file_data["arr_0"]

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        image = image.squeeze()
        image = image.transpose(1, 2).transpose(0, 1)
        if self._m_transform:
            image = self._m_transform(image)

        label = label.numpy()[0]

        return image, label


class CIFAR10LTNPZDataset(_CIFARLTNPZDataset):

    PREFIX_DATASET_TRAIN = "cifar10-lt"
    PREFIX_DATASET_TEST = "cifar10"

    def __init__(self, root: str, train: bool, transform=None):
        super().__init__(CIFAR10LTNPZDataset.PREFIX_DATASET_TRAIN if train else CIFAR10LTNPZDataset.PREFIX_DATASET_TEST, root, train, transform)


class CIFAR100LTNPZDataset(_CIFARLTNPZDataset):

    PREFIX_DATASET_TRAIN = "cifar100-lt"
    PREFIX_DATASET_TEST = "cifar100"

    def __init__(self, root: str, train: bool, transform=None):
        super().__init__(CIFAR100LTNPZDataset.PREFIX_DATASET_TRAIN if train else CIFAR100LTNPZDataset.PREFIX_DATASET_TEST, root, train, transform)
