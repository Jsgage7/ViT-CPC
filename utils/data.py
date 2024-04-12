import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from utils.patches import MakePatches


# from https://github.com/rschwarz15/CPCV2-PyTorch/blob/master/data/data_handlers.py
data_norms = {
    'cifar10': {
        "mean": [0.49139968, 0.48215827, 0.44653124],
        "std": [0.24703233, 0.24348505, 0.26158768],
        "bw_mean": [0.4808616],
        "bw_std": [0.23919088],
    },
    'cifar100': {
        "mean": [0.5070746, 0.48654896, 0.44091788],
        "std": [0.26733422, 0.25643846, 0.27615058],
        "bw_mean": [0.48748648],
        "bw_std": [0.25063065],
    }
}



def make_training_transforms(args, data_norm, fully_supervised):
    """ return a composite transform for all data preprocessing needed for training.
    args = {
      crop_size: 1-d size for the random crop
      crop_padding: padding for random crop
      patch_size: 1-d size for the patch
      p_horiz_flip: probability
      is_grayscale: boolean. if image isnt already grayscale, will be converted.
      make_patches: boolean
    }"""
    print(args)

    print(f'{fully_supervised=}', flush=True)

    trans = [
        transforms.RandomCrop(args['crop_size'], args['crop_padding']),
        transforms.RandomHorizontalFlip(0.5 if args['p_horiz_flip'] is None else args['p_horiz_flip'])
    ]

    # convert to tensor and normalize.
    if args['is_grayscale']:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=data_norm["bw_mean"], std=data_norm["bw_std"]))
    else:
        if not fully_supervised:
            # part of step 3
            trans.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
            # listed as step 4 in patch preprocessing of Appendix A
            trans.append(transforms.RandomGrayscale(p=0.25))
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=data_norm["mean"], std=data_norm["std"]))


    if args['make_patches']:
        trans.append(MakePatches(crop_size=args['crop_size'], patch_size=args['patch_size']))

    trans = transforms.Compose(trans)

    print("training transforms: " + str(trans))

    return trans

def make_test_transforms(args, data_norm):
    """ return a composite transform for all data preprocessing needed for test.
    args = {
      crop_size: 1-d size for the random crop
      patch_size: 1-d size for the patch
      is_grayscale: boolean. if image isnt already grayscale, will be converted.
      make_patches: boolean
    }"""

    trans = [transforms.CenterCrop(args['crop_size'])]

    # convert to tensor and normalize.
    if args['is_grayscale']:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=data_norm["bw_mean"], std=data_norm["bw_std"]))
    else:
        # trans.append(transforms.RandomGrayscale(p=0.25))
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=data_norm["mean"], std=data_norm["std"]))

    if args['make_patches']:
        trans.append(MakePatches(crop_size=args['crop_size'], patch_size=args['patch_size']))


    trans = transforms.Compose(trans)

    return trans

def make_data_subset_sampler(dataset, dataset_pct):
    dataset_size = len(dataset)
    dataset_idxs = list(range(dataset_size))
    dataset_lim = int(dataset_pct * len(dataset))
    np.random.shuffle(dataset_idxs)
    dataset_subset_idxs = dataset_idxs[:dataset_lim]
    dataset_sampler = torch.utils.data.sampler.SubsetRandomSampler(dataset_subset_idxs)
    return dataset_sampler


def make_cifar10_dataloader(batch_size, is_grayscale=False, is_train=True, dataset_pct=1.0, fully_supervised=False):
    num_workers = 2
    download_data = True
    data_root = 'data/cifar10/'

    args = {
      'crop_size': 32,
      'crop_padding': 0,
      'patch_size': 8,
      'p_horiz_flip': 0.5,
      'is_grayscale': is_grayscale,
      'make_patches': True
    }

# <<<<<<< vit
#     transform_train = make_training_transforms(args, data_norms['cifar10'])
#     dataset = datasets.CIFAR10(root=data_root, train=is_train, download=download_data,
#                                 transform=transform_train)

#     if is_train:
#         # generate random subset of data
#         N = len(dataset)
#         num_train_samples = int(N * dataset_pct)
#         dataset_indices = np.random.choice(N, num_train_samples, replace=False)
#         dataset_subset = torch.utils.data.Subset(dataset, dataset_indices)
#         dataset = dataset_subset
        
#         val_size = int(0.1 * num_train_samples)
#         train_size = len(dataset) - val_size
#         dataset_train, dataset_val = random_split(dataset, [train_size, val_size])
        

#         dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
#                                             num_workers=num_workers)
        
#         dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True,
#                                             num_workers=num_workers)
#         return dataloader_train, dataloader_val
# =======
    data_norm = data_norms['cifar10']
    transform = (make_training_transforms(args, data_norm, fully_supervised=fully_supervised) if is_train
        else make_test_transforms(args, data_norm))
    dataset = datasets.CIFAR10(root=data_root, train=is_train, download=download_data,
                                transform=transform)

    dataset_sampler = make_data_subset_sampler(dataset, dataset_pct)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            sampler=dataset_sampler)
    return dataloader


def make_cifar100_dataloader(batch_size, is_grayscale, is_train=True, dataset_pct=1.0):
    num_workers = 2
    download_data = True
    data_root = 'data/cifar100/'

    args = {
      'crop_size': 32,
      'crop_padding': 0,
      'patch_size': 8,
      'p_horiz_flip': 0.5,
      'is_grayscale': is_grayscale,
      'make_patches': True
    }

    data_norm = data_norms['cifar100']
    transform = (make_training_transforms(args, data_norm) if is_train
        else make_test_transforms(args, data_norm))

    dataset = datasets.CIFAR100(root=data_root, train=is_train, download=download_data,
                                transform=transform)

    dataset_sampler = make_data_subset_sampler(dataset, dataset_pct)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            sampler=dataset_sampler)
    return dataloader





# newer dataloader fn, incorporating train/val split
def make_cifar100_dataloader_train_val(batch_size, is_grayscale, dataset_pct=1.0, val_frac=0.2):
    num_workers = 2
    download_data = True
    data_root = 'data/cifar100/'

    args = {
      'crop_size': 32,
      'crop_padding': 0,
      'patch_size': 8,
      'p_horiz_flip': 0.5,
      'is_grayscale': is_grayscale,
      'make_patches': True
    }

    transform = make_training_transforms(args, data_norms['cifar100'])
    dataset = datasets.CIFAR100(root=data_root, train=True, download=download_data,
                                transform=transform)

    # generate random subset of data
    N = len(dataset)
    num_train_samples = int(N * dataset_pct)
    dataset_indices = np.random.choice(N, num_train_samples, replace=False)
    dataset_subset = torch.utils.data.Subset(dataset, dataset_indices)

    # split into train/val
    N_subset = len(dataset_subset)
    V = int(num_train_samples * val_frac)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_subset, [N_subset - V, V])

    print(f'train/val dataset sizes: {N_subset - V}/{V}')

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_train, dataloader_val


def make_cifar10_dataloader_train_val(batch_size, is_grayscale, dataset_pct=1.0, val_frac=0.2, fully_supervised=False):
    num_workers = 2
    download_data = True
    data_root = 'data/cifar10/'

    args = {
      'crop_size': 32,
      'crop_padding': 0,
      'patch_size': 8,
      'p_horiz_flip': 0.5,
      'is_grayscale': is_grayscale,
      'make_patches': True
    }

    transform = make_training_transforms(args, data_norms['cifar10'], fully_supervised)
    dataset = datasets.CIFAR10(root=data_root, train=True, download=download_data,
                                transform=transform)

    # generate random subset of data
    N = len(dataset)
    num_train_samples = int(N * dataset_pct)
    dataset_indices = np.random.choice(N, num_train_samples, replace=False)
    dataset_subset = torch.utils.data.Subset(dataset, dataset_indices)

    # split into train/val
    N_subset = len(dataset_subset)
    V = int(num_train_samples * val_frac)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_subset, [N_subset - V, V])

    print(f'train/val dataset sizes: {N_subset - V}/{V}')

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_train, dataloader_val


def make_cifar10_dataloader_test(batch_size, is_grayscale):
    num_workers = 2
    download_data = True
    data_root = 'data/cifar10/'

    args = {
      'crop_size': 32,
      'patch_size': 8,
      'is_grayscale': is_grayscale,
      'make_patches': True
    }

    transform = make_test_transforms(args, data_norms['cifar10'])
    dataset = datasets.CIFAR10(root=data_root, train=False, download=download_data,
                                transform=transform)

    dataloader_test = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader_test
