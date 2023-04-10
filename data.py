import numpy as np
from torch.utils.data.dataset import Subset, ConcatDataset

import datasets


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)


def get_dataset(dataset_name, root, task_list, split='train', download=True, transform=None, seed=0):
    assert split in ['train', 'val', 'test']
    supported_dataset = ['PACS', 'OfficeHome', 'DomainNet', 'VLCS', 'Terra']
    assert dataset_name in supported_dataset

    dataset = datasets.__dict__[dataset_name]

    train_split_list = []
    val_split_list = []
    test_split_list = []
    # we follow DomainBed and split each dataset randomly into two parts, with 80% samples and 20% samples
    # respectively, the former (larger) will be used as training set, and the latter will be used as validation set.
    split_ratio = 0.8
    num_classes = 0

    # under domain generalization setting, we use all samples in target domain as test set
    for task in task_list:
        if dataset_name == 'PACS':
            all_split = dataset(root=root, task=task, split='all', download=download, transform=transform)
            num_classes = all_split.num_classes
        elif dataset_name == 'OfficeHome':
            all_split = dataset(root=root, task=task, download=download, transform=transform)
            num_classes = all_split.num_classes
        elif dataset_name == 'DomainNet':
            train_split = dataset(root=root, task=task, split='train', download=download, transform=transform)
            test_split = dataset(root=root, task=task, split='test', download=download, transform=transform)
            num_classes = train_split.num_classes
            all_split = ConcatDataset([train_split, test_split])
        elif dataset_name == 'VLCS':
            all_split = dataset(root=root, task=task, download=download, transform=transform)
            num_classes = all_split.num_classes
        elif dataset_name == 'Terra':
            all_split = dataset(root=root, task=task, download=download, transform=transform)
            num_classes = all_split.num_classes

        train_split, val_split = split_dataset(all_split, int(len(all_split) * split_ratio), seed)

        train_split_list.append(train_split)
        val_split_list.append(val_split)
        test_split_list.append(all_split)

    train_dataset = ConcatDataset(train_split_list)
    val_dataset = ConcatDataset(val_split_list)
    test_dataset = ConcatDataset(test_split_list)

    dataset_dict = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    return dataset_dict[split], num_classes


def split_dataset(dataset, n, seed=0):
    assert (n <= len(dataset))
    idxes = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(idxes)
    subset_1 = idxes[:n]
    subset_2 = idxes[n:]
    return Subset(dataset, subset_1), Subset(dataset, subset_2)